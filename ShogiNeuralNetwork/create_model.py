from keras import layers
from .data_info import CATEGORIES_FIGURE_TYPE
import keras


def create_model(
        cell_img_size: int,
        conv_layers_filters_counts: list[int],
):
    img_input = keras.Input(shape=(cell_img_size, cell_img_size, 1), name="input")
    x = img_input
    for filters in conv_layers_filters_counts:
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    figure_dense = layers.Dense(128, activation="relu")(x)
    figure_pred = layers.Dense(len(CATEGORIES_FIGURE_TYPE), activation="softmax", name="figure")(figure_dense)

    direction_dense = layers.Dense(128, activation="relu")(x)
    direction_pred = layers.Dense(3, activation="softmax", name="direction")(direction_dense)

    model = keras.Model(
        img_input,
        outputs=[figure_pred, direction_pred]
    )

    return model
