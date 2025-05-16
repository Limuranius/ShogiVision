import os

import keras
import tensorflow as tf
from matplotlib import pyplot as plt

import ShogiNeuralNetwork
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


def train_and_save_model(img_size: int):
    dataset = ShogiNeuralNetwork.CellsDataset.load(paths.CELLS_DATASET_PATH)
    train, test = dataset.to_tf_dataset(
        cell_image_size=img_size,
        test_fraction=0.2,
        batch_size=64,
        augment=True,
        to_grayscale=True,
    )

    model = create_model(img_size)

    figure_weights, direction_weights = dataset.class_weights()

    history = model.fit(
        train,
        epochs=30,
        validation_data=test,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_figure_accuracy", mode="max"),
            # keras.callbacks.EarlyStopping(monitor="val_figure_accuracy", mode="max"),
            keras.callbacks.ModelCheckpoint(
                filepath=paths.MIXED_MODEL_KERAS_PATH,
                save_best_only=True,
                monitor="val_figure_accuracy",
                mode="max",
            )
        ],
        # class_weight={
        #     "figure": figure_weights,
        #     "direction": direction_weights,
        # }
    ).history

    plt.figure(figsize=(10, 5))
    _, ax = plt.subplots(ncols=2)
    ax[0].plot(history["direction_accuracy"])
    ax[0].plot(history["val_direction_accuracy"])
    ax[0].set_title("Direction")
    ax[0].legend(["Train", "Valid"])
    ax[1].plot(history["figure_accuracy"])
    ax[1].plot(history["val_figure_accuracy"])
    ax[1].set_title("Figure")
    ax[1].legend(["Train", "Valid"])
    plt.show()


def save_onnx_model():
    os.system("python -m tf2onnx.convert --saved-model {tf_model_path} --output {onnx_model_path}".format(
        tf_model_path=paths.MIXED_MODEL_KERAS_PATH,
        onnx_model_path=paths.MIXED_MODEL_ONNX_PATH,
    ))


def save_tflite_model():
    model = keras.models.load_model(paths.MIXED_MODEL_KERAS_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(paths.MIXED_MODEL_TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)


def main():
    train_and_save_model(img_size=64)
    save_onnx_model()
    save_tflite_model()


if __name__ == '__main__':
    main()
