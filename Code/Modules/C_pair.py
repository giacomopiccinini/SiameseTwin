import numpy as np
import logging
from Code.Techniques.Pairing.pair import make_pairs


def pair_dataset(train_set, validation_set, test_set):

    """Create paired train, validation  and test set"""
    
    # Set logger
    logging.basicConfig(level=logging.NOTSET)
    log = logging.getLogger(__name__)

    # Extract images and labels from training set
    train_images = np.array([image.tensor for image in train_set.data])
    train_labels = np.array([label.label for label in train_set.labels])

    # Split images and labels in pairs of positive and negative
    log.info("Pairing training set")
    paired_train_images, paired_train_labels = make_pairs(train_images, train_labels)

    # Split in positive and negative
    paired_train_images = [paired_train_images[:, 0], paired_train_images[:, 1]]

    # Extract image and labels from validation set
    validation_images = np.array([image.tensor for image in validation_set.data])
    validation_labels = np.array([label.label for label in validation_set.labels])

    # Split images and labels in pairs of positive and negative
    log.info("Pairing validation set")
    paired_validation_images, paired_validation_labels = make_pairs(
        validation_images, validation_labels
    )

    # Split in positive and negative
    paired_validation_images = [
        paired_validation_images[:, 0],
        paired_validation_images[:, 1],
    ]

    # Extract images and labels from test set
    test_images = np.array([image.tensor for image in test_set.data])
    test_labels = np.array([label.label for label in test_set.labels])

    # Split images and labels in pairs of positive and negative
    log.info("Pairing test set")
    paired_test_images, paired_test_labels = make_pairs(test_images, test_labels)

    # Split in positive and negative
    paired_test_images = [paired_test_images[:, 0], paired_test_images[:, 1]]

    return (
        paired_train_images,
        paired_train_labels,
        paired_validation_images,
        paired_validation_labels,
        paired_test_images,
        paired_test_labels,
    )
