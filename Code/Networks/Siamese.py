from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input

from Code.Networks.Resnet import Resnet
from Code.Metric.euclidean import euclidean_distance


def SiameseNetwork(shape: tuple):

    """Siamese network implementation based on a feature extractor defined as the
    Sister Network"""

    # Construct shape of input images
    image_shape = shape

    # Define the inputs of the network
    image_1 = Input(shape=image_shape)
    image_2 = Input(shape=image_shape)

    # Define the feature extractor (i.e. the sister network)
    feature_extractor = Resnet()

    # Preprocess input for the feature extractor
    image_1 = preprocess_input(image_1)
    image_2 = preprocess_input(image_2)

    # Extract features from the two input images
    # Make sure it runs in inference mode
    # This way, the batch norm will not cause harm
    features_1 = feature_extractor(image_1, training=False)
    features_2 = feature_extractor(image_2, training=False)

    # Pass the features into a distance layer
    distance = Lambda(euclidean_distance)([features_1, features_2])

    # Pass the distance in a dense layer with sigmoid activation
    outputs = Dense(1, activation="sigmoid")(distance)

    # Construct the model
    model = Model(inputs=[image_1, image_2], outputs=outputs)

    return model
