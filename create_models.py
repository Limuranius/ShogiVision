import json

from ShogiNeuralNetwork.train_model import train_model
from ShogiNeuralNetwork.create_model import create_model
from ShogiNeuralNetwork.CellsDataset import CellsDataset
from extra.image_modes import ImageMode
from config import Paths, GLOBAL_CONFIG
import os
import onnx

dataset = CellsDataset()
dataset.load(Paths.DATASET_PATH)

dataset = dataset.resize((GLOBAL_CONFIG.NeuralNetworkTraining.cell_img_size, GLOBAL_CONFIG.NeuralNetworkTraining.cell_img_size))
dataset = dataset.convert(ImageMode(GLOBAL_CONFIG.NeuralNetworkTraining.image_mode))
train, test = dataset.get_tf_dataset()

model = create_model(
    cell_img_size=GLOBAL_CONFIG.NeuralNetworkTraining.cell_img_size,
    conv_layers_filters_counts=[64, 64, 64, 64],
)
model = train_model(model, train, GLOBAL_CONFIG.NeuralNetworkTraining.epochs, 1)

print(model.evaluate(test, return_dict=True, verbose=1))

model.save(Paths.MODEL_TF_PATH)
os.system("python -m tf2onnx.convert --saved-model {tf_model_path} --output {onnx_model_path}".format(
    tf_model_path=Paths.MODEL_TF_PATH,
    onnx_model_path=Paths.MODEL_ONNX_PATH,
))

# Adding metadata to model
model = onnx.load_model(Paths.MODEL_ONNX_PATH)


def add_meta(key, value):
    m = model.metadata_props.add()
    m.key = key
    m.value = json.dumps(value)


add_meta("cell_img_size", GLOBAL_CONFIG.NeuralNetworkTraining.cell_img_size)
add_meta("image_mode", GLOBAL_CONFIG.NeuralNetworkTraining.image_mode)
onnx.save(model, Paths.MODEL_ONNX_PATH)