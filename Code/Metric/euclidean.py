import tensorflow.keras.backend as K


def euclidean_distance(vectors):

    """Compute Euclidean distance between vectors using Keras backend"""

    # Unpack the vectors
    (v, w) = vectors

    # Compute the sum of squared distances between the vectors
    sum_squared = K.sum(K.square(v - w), axis=1, keepdims=True)

    # Return the euclidean distance between the vectors, epsilon is a fudge factor
    # to prevent Nan or infs
    return K.sqrt(K.maximum(sum_squared, K.epsilon()))
