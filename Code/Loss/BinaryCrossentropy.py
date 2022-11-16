import logging
from tensorflow.keras.losses import BinaryCrossentropy


def loss(**kwargs):

    """Create Binary Crossentropy loss"""

    try:
        # Create loss if correct keywords are passed
        Loss = BinaryCrossentropy(**kwargs)

        # Log the loading
        logging.info("Binary Crossentropy loss has been loaded")

        # Return the optimizer
        return Loss

    except Exception as e:
        # Raise an exception if wrong keywords are passed
        raise e
