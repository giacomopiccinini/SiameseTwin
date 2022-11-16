import os
import tensorflow as tf

from Code.CometML.track_experiment import track
from Code.Callbacks.callback import list_callbacks


def train(
    Siamese,
    paired_train_images,
    paired_train_labels,
    paired_validation_images,
    paired_validation_labels,
    args,
):

    """Train Siamese"""
    # Ensure GPU is visible
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # TensorFlow routine
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Load experiment
    experiment = track(args)

    # Load callbacks
    callbacks = list_callbacks(experiment)

    # Train CNN
    Siamese.fit(
        x=paired_train_images,
        y=paired_train_labels,
        validation_data=(paired_validation_images, paired_validation_labels),
        epochs=args["Train"].epochs,
        callbacks=callbacks,
    )

    return experiment
