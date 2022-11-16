import pandas as pd
from sklearn.model_selection import train_test_split
from Code.Loaders.ImageLoader import ImageLoader
from Code.Classes.Label import Label


def split(args):

    """Split dataset into train, test and validation.

    Variables and default values:

    test_size=0.2
    validation_size=0.2
    seed=42

    """

    # Read from Excel file containing labels and images names
    df = pd.read_excel("Input/Labels/labels.xlsx")

    # Create paths to data
    image_paths = [f"Input/Images/{image}" for image in df["Image"]]

    # Retrieve labels
    labels = list(map(Label, df["Image"], df["Label"]))

    # Separate test and train set
    images_train, images_test, labels_train, labels_test = train_test_split(
        image_paths, labels, test_size=args.test_size, random_state=args.seed
    )

    # Separate train and validation set
    images_train, images_validation, labels_train, labels_validation = train_test_split(
        images_train,
        labels_train,
        test_size=args.validation_size,
        random_state=args.seed,
    )

    # Initialise datasets
    train_set = ImageLoader(
        images_train, labels_train, set_type="train", batch_size=args.batch
    )

    validation_set = ImageLoader(
        images_validation,
        labels_validation,
        set_type="validation",
        batch_size=args.batch,
        maximum=train_set.maximum,
        minimum=train_set.minimum,
    )

    test_set = ImageLoader(
        images_test,
        labels_test,
        set_type="test",
        batch_size=args.batch,
        maximum=train_set.maximum,
        minimum=train_set.minimum,
    )

    # Save splitting in .yaml files
    train_set.save_split()
    validation_set.save_split()
    test_set.save_split()

    return train_set, validation_set, test_set
