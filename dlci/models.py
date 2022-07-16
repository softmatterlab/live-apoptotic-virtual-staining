from . import deeptrack as dt
from tensorflow.keras import layers

from tensorflow.keras.initializers import RandomNormal


def generator(breadth, depth):
    """Creates a u-net generator that
    * Uses concatenation skip steps in the encoder
    * Uses maxpooling for downsampling
    * Uses resnet block for the base block
    * Uses instance normalization and leaky relu.
    Parameters
    ----------
    breadth : int
        Number of features in the top level. Each sequential level of the u-net
        increases the number of features by a factor of two.
    depth : int
        Number of levels to the u-net. If `n`, then there will be `n-1` pooling layers.
    """

    kernel_initializer = RandomNormal(mean=0.0, stddev=0.02)

    activation = layers.LeakyReLU(0.2)

    convolution_block = dt.layers.ConvolutionalBlock(
        activation=activation,
        instance_norm=True,
        kernel_initializer=kernel_initializer,
    )

    base_block = dt.layers.ResidualBlock(
        activation=activation, kernel_initializer=kernel_initializer
    )

    pooling_block = dt.layers.ConvolutionalBlock(
        strides=2,
        activation=activation,
        instance_norm=True,
        kernel_initializer=kernel_initializer,
    )

    upsample_block = dt.layers.StaticUpsampleBlock(
        kernel_size=3,
        instance_norm=True,
        activation=activation,
        with_conv=False,
        kernel_initializer=kernel_initializer,
    )

    return dt.models.unet(
        input_shape=(None, None, 1),
        conv_layers_dimensions=list(
            breadth * 2 ** n for n in range(depth - 1)
        ),
        base_conv_layers_dimensions=(breadth * 2 ** (depth - 1),),
        output_conv_layers_dimensions=(
            breadth,
            breadth // 2,
        ),
        steps_per_pooling=2,
        number_of_outputs=1,
        output_kernel_size=1,
        output_activation="tanh",
        encoder_convolution_block=convolution_block,
        decoder_convolution_block=convolution_block,
        base_convolution_block=base_block,
        pooling_block=pooling_block,
        upsampling_block=upsample_block,
        output_convolution_block=convolution_block,
    )


def discriminator(depth):
    """Creates a patch discriminator according to the specifications in the paper.
    Parameters
    ----------
    depth : int
        Number of levels to the model.
    """

    activation = layers.LeakyReLU(0.2)

    discriminator_convolution_block = dt.layers.ConvolutionalBlock(
        kernel_size=(4, 4),
        strides=1,
        activation=activation,
        instance_norm=lambda x: (
            False
            if x == 16
            else {"axis": -1, "center": False, "scale": False},
        ),
    )

    discriminator_pooling_block = dt.layers.ConvolutionalBlock(
        kernel_size=(4, 4),
        strides=2,
        activation=activation,
        instance_norm={"axis": -1, "center": False, "scale": False},
    )

    return dt.models.convolutional(
        input_shape=[
            (None, None, 1),
            (None, None, 1),
        ],  # shape of the input
        conv_layers_dimensions=[16 * 2 ** n for n in range(depth)],
        dense_layers_dimensions=(),  # number of neurons in each dense layer
        # number of neurons in the final dense step (numebr of output values)
        number_of_outputs=1,
        compile=False,
        output_kernel_size=4,
        dense_top=False,
        convolution_block=discriminator_convolution_block,
        pooling_block=discriminator_pooling_block,
    )
