import sys
import itertools
import tensorflow
import numpy as np
import dlci

from tensorflow import keras
from dlci import deeptrack as dt
from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

TEST_VARIABLES = {
    "generator_depth": [6],
    "generator_breadth": [16],
    "discriminator_depth": [5],
    "batch_size": [8],
    "min_data_size": [1024],
    "max_data_size": [1025],
    "augmentation_dict": [
        {
            "FlipLR": {},
            "Affine": {
                "rotate": "lambda: np.random.rand() * 2 * np.pi",
            },
        }
    ],
    "mae_loss_weight": [0.8],
    "content_loss_weight": [0.5],
    "path_to_dataset": [r"C:/GU/Live-dead-staining (evaluation)/datasets"],
    "dataset": ["Caspase"],
    "content_discriminator": ["DenseNet121"],
}


def model_initializer(
    generator_depth,
    generator_breadth,
    discriminator_depth,
    mae_loss_weight=1,
    content_loss_weight=1,
    content_discriminator="VGG16",
    **kwargs,
):
    generator = dlci.generator(generator_breadth, generator_depth)
    discriminator = dlci.discriminator(discriminator_depth)

    return dt.models.ccgan(
        generator=generator,
        discriminator=discriminator,
        discriminator_loss="mse",
        discriminator_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss=["mse", "mse", "mae"],
        assemble_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss_weights=[1, content_loss_weight, mae_loss_weight],
        content_discriminator=content_discriminator,
    )


# Populate models
_models = []
_generators = []


def append_model(**arguments):
    _models.append((arguments, lambda: model_initializer(**arguments)))


def append_generator(**arguments):

    _generators.append(
        (
            arguments,
            lambda: dlci.DataGenerator(
                **arguments,
            ),
        )
    )


for prod in itertools.product(*TEST_VARIABLES.values()):

    arguments = dict(zip(TEST_VARIABLES.keys(), prod))
    append_model(**arguments)
    append_generator(**arguments)


def get_model(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, model = _models[i]
    return args, model()


def get_generator(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, generator = _generators[i]
    return args, generator()
