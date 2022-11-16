import logging
from tensorflow import keras


def Resnet():

    """Load the InceptionResNetV2 network, with weights from imagenet"""

    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    try:

        # Load the InceptionResNet
        model = keras.applications.InceptionResNetV2(
            include_top=False, weights="imagenet", input_shape=None
        )
        log.info("InceptionResNetV2 loaded successfully")

        # Freeze the model
        model.trainable = False

    except Exception as e:
        log.info(f"InceptionResNetV2 failed to load, reason: {e}")
