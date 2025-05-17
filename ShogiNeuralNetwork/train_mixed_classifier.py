import os
import sys

import numpy as np

sys.path.insert(0, "..")  # in case module is called outside venv

import keras
import sklearn
import tensorflow as tf
from matplotlib import pyplot as plt

import ShogiNeuralNetwork.data_info
from ShogiNeuralNetwork import CellsDataset
from ShogiNeuralNetwork.data_info import CATEGORIES_FIGURE_TYPE, CATEGORIES_DIRECTION
from config import paths


def create_model(
        cell_img_size: int,
) -> keras.Model:
    """Creates keras model to classify figure type and direction of board cells images"""
    x = img_input = keras.Input(shape=(cell_img_size, cell_img_size, 1), name="input")
    for filters in [32, 64, 128, 256]:
        x = keras.layers.Conv2D(filters, 3)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)

    figure_dense = keras.layers.Dense(128, activation="relu")(x)
    figure_pred = keras.layers.Dense(len(CATEGORIES_FIGURE_TYPE), activation="softmax", name="figure")(figure_dense)

    direction_dense = keras.layers.Dense(128, activation="relu")(x)
    direction_pred = keras.layers.Dense(len(CATEGORIES_DIRECTION), activation="softmax", name="direction")(
        direction_dense)

    model = keras.Model(
        img_input,
        outputs=[figure_pred, direction_pred]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={
            "figure": keras.losses.SparseCategoricalCrossentropy(),
            "direction": keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "figure": ["accuracy"],
            "direction": ["accuracy"],
        }
    )

    return model


def train_and_save_model(img_size: int, epochs: int):
    dataset = CellsDataset.CellsDataset.load(paths.CELLS_DATASET_PATH)
    train_dataset, test_dataset = dataset.train_test_split(test_fraction=0.2, random_state=42)
    train = train_dataset.to_tf_dataset(
        cell_image_size=img_size,
        batch_size=64,
        augment=True,
        to_grayscale=True,
        shuffle=True,
    )
    test = test_dataset.to_tf_dataset(
        cell_image_size=img_size,
        batch_size=64,
        augment=False,
        to_grayscale=True,
        shuffle=False,
    )

    model = create_model(img_size)

    history = model.fit(
        train,
        epochs=epochs,
        validation_data=test,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_figure_accuracy",
                mode="max"
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_figure_accuracy",
                mode="max",
                patience=10,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=paths.MIXED_MODEL_KERAS_PATH,
                save_best_only=True,
                monitor="val_figure_accuracy",
                mode="max",
            )
        ],
    ).history

    model.export(paths.MIXED_MODEL_EXPORT_PATH)

    plt.figure(figsize=(5, 10))
    _, ax = plt.subplots(ncols=2)
    ax[0].plot(history["direction_accuracy"])
    ax[0].plot(history["val_direction_accuracy"])
    ax[0].set_title("Direction")
    ax[0].legend(["Train", "Valid"])
    ax[0].grid()
    ax[1].plot(history["figure_accuracy"])
    ax[1].plot(history["val_figure_accuracy"])
    ax[1].set_title("Figure")
    ax[1].legend(["Train", "Valid"])
    ax[1].grid()
    plt.savefig(paths.MODELS_DIR / "report" / "accuracy.png", dpi=200)
    plt.close()

    predictions = model.predict(test)
    figure_predict = predictions[0].argmax(axis=1)
    direction_predict = predictions[1].argmax(axis=1)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot()
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        test_dataset.figure_labels(),
        figure_predict,
        display_labels=[figure.name for figure in ShogiNeuralNetwork.data_info.CATEGORIES_FIGURE_TYPE],
        xticks_rotation='vertical',
        ax=ax,
    )
    plt.savefig(paths.MODELS_DIR / "report" / "test_confusion.png", dpi=200)
    plt.close()

    accuracy = test_dataset.figure_labels() == figure_predict
    print("Balanced accuracy:",
          np.mean([np.mean(accuracy[figure_predict == i]) for i in range(figure_predict.max() + 1)]))


def save_onnx_model():
    os.system("python -m tf2onnx.convert --saved-model {tf_model_path} --output {onnx_model_path}".format(
        tf_model_path=paths.MIXED_MODEL_EXPORT_PATH,
        onnx_model_path=paths.MIXED_MODEL_ONNX_PATH,
    ))


def save_tflite_model():
    model = keras.models.load_model(paths.MIXED_MODEL_KERAS_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(paths.MIXED_MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)


def main():
    train_and_save_model(img_size=64, epochs=100)
    save_onnx_model()
    save_tflite_model()


if __name__ == '__main__':
    main()
