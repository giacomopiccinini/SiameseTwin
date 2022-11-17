import numpy as np
from tqdm import tqdm


def make_pairs(images, labels, bound=50000):

    """Pair images and labels"""

    # Initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pair_images = []
    pair_labels = []

    # Calculate the total number of classes present in the dataset
    classes = np.unique(labels)

    # Build a list of indices for each class label that
    # provides the indices for all examples with a given label
    idx = [np.where(labels == i)[0] for i in classes]
    
    # Initialise counter to prevent memory allocation error
    counter = 0

    for current_image, label in tqdm(zip(images, labels)):
        
        # Break if we exceeed the bound
        if counter > bound:
            break
     
        # Fetch index for class of interest
        class_idx = np.where(classes == label)[0][0]
        
        # Randomly pick an image that belongs to the *same* class
        idx_b = np.random.choice(idx[class_idx])
        pos_image = images[idx_b]

        # Find indices for negative classes
        neg_idx = np.where(labels != label)[0]

        # Randomly select an image from the negative class
        neg_image = images[np.random.choice(neg_idx)]

        # Update images and labels for positive class
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])

        # Update images and labels for negative class
        pair_images.append([current_image, neg_image])
        pair_labels.append([0])
        
        # Increase counter
        counter += 1

    return np.array(pair_images), np.array(pair_labels)
