from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten


def SisterNetwork(
    shape,
    kernel_size_x=3,
    kernel_size_y=3,
    pool_size_x=2,
    pool_size_y=2,
    dropout=0.2,
    filters=(4, 8, 16, 32, 64, 128),
    latent=64,
):

    """Sister network used for feature extraction from images. It is basically
    just a CNN"""

    # Instantiate a Keras tensor (it should be a n_x pixel x n_y pixel x n_channel tensor)
    input = Input(shape=shape)

    # At first, the tensor is simply the input
    x = input

    # Loop over the number of filters to construct CNN recursively
    for f in filters:

        # Apply 2D convolution with ReLU activation function
        x = Conv2D(
            f,
            kernel_size=(kernel_size_y, kernel_size_x),
            activation="relu",
            padding="same",
        )(x)

        # Apply batch normalisation (specify axis = -1 if assuming TensorFlow/channels-last ordering)
        x = BatchNormalization(axis=-1)(x)

        # Apply Max Pooling
        x = MaxPooling2D(pool_size=(pool_size_y, pool_size_x))(x)

    # Add drop-out layer
    x = Dropout(dropout)(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Flatten the tensor before applying the dense layer
    x = Flatten()(x)

    # Apply dense layer (units = dimensionality of the output layer)
    x = Dense(units=latent, activation="relu")(x)

    # Apply batch normalisation
    output = BatchNormalization(axis=-1)(x)

    # Build the model
    sister_net = Model(input, output)

    return sister_net
