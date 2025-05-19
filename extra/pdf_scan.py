import cv2
import pypdfium2
import numpy as np

from extra.types import ImageNP, Box


def __get_big_external_boxes(img_mask: ImageNP) -> list[Box]:
    # Getting external contours
    contours, _ = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Removing small contours (like characters of figures)
    total_area = img_mask.shape[0] * img_mask.shape[1]
    contours = [c for c in contours if cv2.contourArea(c) / total_area > 0.01]

    boxes = []
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        boxes.append((x, y, w, h))
    return boxes


def __get_boxed_images(img: ImageNP, boxes: list[Box]) -> list[ImageNP]:
    return [__crop_image_box(img, box) for box in boxes]


def __crop_image_box(image: ImageNP, box: Box):
    x, y, w, h = box
    return image[y: y + h, x: x + w]


def __is_board_image(img_mask: ImageNP) -> bool:
    """Returns True if whole image is image of board"""
    h, w = img_mask.shape
    min_size = min(h, w)

    # Getting all long straight lines
    r = 1
    theta = np.pi / 180
    threshold = 50
    min_len = int(min_size * 0.98)
    lines = cv2.HoughLinesP(
        img_mask,
        r,
        theta,
        threshold,
        minLineLength=min_len,
        maxLineGap=5
    )

    # Considering image a board if it has at least 10 lines across whole image
    return lines is not None and len(lines) > 10


def __capture_surroundings(border_box: Box) -> Box:
    x, y, w, h = border_box
    margin_left = int(w * 0.125)
    margin_right = int(w * 0.25)
    margin_up = int(h * 0.1)
    margin_down = int(h * 0.1)

    new_x = max(0, x - margin_left)
    new_y = max(0, y - margin_up)
    new_w = w + margin_left + margin_right
    new_h = h + margin_up + margin_down
    return new_x, new_y, new_w, new_h


def __remove_border_pixels(image: ImageNP, width: int) -> ImageNP:
    return image[width: -width, width: -width]


def __extract_boards_borders(img_mask: ImageNP, x0=0, y0=0) -> list[Box]:
    boxes = __get_big_external_boxes(img_mask)
    boxed_masks = __get_boxed_images(img_mask, boxes)
    boards_boxes = []

    for b_mask, box in zip(boxed_masks, boxes):
        if __is_board_image(b_mask):
            boards_boxes.append(box)
        else:
            # Searching for boards inside found boxes
            b_mask = __remove_border_pixels(b_mask, 1)
            box_boards_boxes = __extract_boards_borders(
                b_mask,
                x0=box[0] + x0 + 1,
                y0=box[1] + y0 + 1,
            )
            for bb_box in box_boards_boxes:
                boards_boxes.append((bb_box[0] + x0, bb_box[1] + y0,  # Adjusting for global coordinates
                                     bb_box[2], bb_box[3]))

    return boards_boxes


def extract_boards_boxes(img: ImageNP, surroundings=True) -> list[Box]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = ((gray < 150) * 255).astype(np.uint8)
    board_boxes = __extract_boards_borders(mask)
    board_boxes.sort(key=lambda box: box[1])
    if surroundings:
        board_boxes = [__capture_surroundings(box) for box in board_boxes]
    return board_boxes


def extract_boards_images(img: ImageNP, surroundings=True) -> list[ImageNP]:
    """Returns all images of boards on img"""
    boxes = extract_boards_boxes(img, surroundings=surroundings)
    board_images = []
    for x, y, w, h in boxes:
        board_images.append(img[y: y + h, x: x + w])
    return board_images


def get_pdf_page_image(pdf: pypdfium2.PdfDocument, page_number: int) -> ImageNP:
    page = pdf[page_number]
    image = page.render(scale=4).to_numpy()
    return image
