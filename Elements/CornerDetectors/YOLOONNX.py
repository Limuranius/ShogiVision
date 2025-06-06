import math
import time

import cv2
import numpy as np
import onnxruntime

from config import paths
from extra.types import ImageNP, Corners
from extra.utils import order_points
from .CornerDetector import CornerDetector
from .yolo_utils import rescale_boxes, xywh2xyxy, nms, draw_detections, sigmoid


class YOLOONNX(CornerDetector):
    def __init__(self):
        self.model = YOLOv8Seg(path=paths.BOARD_SEGMENTATION_ONNX_MODEL_PATH)

    def get_corners(self, image: ImageNP) -> Corners:
        self.model(image)

        if len(self.model.mask_maps) == 0:
            h, w = image.shape[:2]
            return (0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)
        mask = self.model.mask_maps[0].astype(np.uint8)

        # plt.imshow(mask)
        # plt.show()
        # with Timer("cv2.findContours"):
        conts = list(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        conts.sort(key=lambda c: cv2.arcLength(c, True), reverse=True)
        poly = conts[0]
        # poly = np.argwhere(mask)[:, np.newaxis]
        arclen = cv2.arcLength(poly, True)
        poly = cv2.approxPolyDP(poly, 0.02 * arclen, True)
        print(order_points(poly[:, 0]))
        return order_points(poly[:, 0])


class YOLOv8Seg:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(
            path,
            providers=[
                'DmlExecutionProvider',
                'CUDAExecutionProvider',
                'CPUExecutionProvider',
            ]
        )
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        # with Timer("prepare_input"):
        input_tensor = self.prepare_input(image)
        # Perform inference on the image
        # with Timer("inference"):
        outputs = self.inference(input_tensor)
        # with Timer("process_box_output"):
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])
        # with Timer("process_prototype"):
        self.mask_maps = self.process_prototype(mask_pred, outputs[1])
        return self.boxes, self.scores, self.class_ids, self.mask_maps

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        # print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4 + self.num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :self.num_classes + 4]
        mask_coefficients = predictions[..., self.num_classes + 4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_coefficients[indices]

    def process_prototype(self, mask_coefficients, prototype):

        if mask_coefficients.shape[0] == 0:
            return []

        prototype = np.squeeze(prototype)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = prototype.shape  # CHW
        masks = sigmoid(mask_coefficients @ prototype.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = rescale_boxes(self.boxes,
                                    (self.img_height, self.img_width),
                                    (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):
            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                   (x2 - x1, y2 - y1),
                                   interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = rescale_boxes(boxes,
                              (self.input_height, self.input_width),
                              (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_results(self, image, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, self.class_names,
                               mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.num_masks = model_outputs[1].shape[1]
        self.num_classes = model_outputs[0].shape[1] - self.num_masks - 4

        metadata_map = self.session.get_modelmeta().custom_metadata_map
        if 'names' in metadata_map:
            self.class_names = eval(metadata_map['names'])
        else:
            self.class_names = [str(i) for i in range(self.num_classes)]
