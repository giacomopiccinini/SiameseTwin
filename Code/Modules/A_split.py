import pandas as pd
from yaml import safe_load
from Code.Loaders.ImageLoader import ImageLoader
from Code.Classes.Label import Label


def split(args):

    """
    Split dataset into train, test and validation.

    Default split for CelebA is used
    """

    # Read from Excel file containing labels and images names
    df = pd.read_excel("Input/Labels/labels.xlsx")

    # Load canonical splits
    with open("Split/splits.yaml", "r") as file:
        sets = safe_load(file)
        train_set = sets["train"]
        validation_set = sets["validation"]
        test_set = sets["test"]

    # Split in train, test and validation
    images_train = df.query("Image in @train_set")["Image"]
    labels_train = df.query("Image in @train_set")["Label"]
    labels_train = list(map(Label, images_train, labels_train))
    images_train = [f"Input/Images/{image}" for image in images_train]

    images_validation = df.query("Image in @validation_set")["Image"]
    labels_validation = df.query("Image in @validation_set")["Label"]
    labels_validation = list(map(Label, images_validation, labels_validation))
    images_validation = [f"Input/Images/{image}" for image in images_validation]

    images_test = df.query("Image in @test_set")["Image"]
    labels_test = df.query("Image in @test_set")["Label"]
    labels_test = list(map(Label, images_test, labels_test))
    images_test = [f"Input/Images/{image}" for image in images_test]

    # Initialise datasets
    train_set = ImageLoader(
        images_train,
        labels_train,
        set_type="train",
        batch_size=args.batch,
        maximum=255,
        minimum=0,
    )

    validation_set = ImageLoader(
        images_validation,
        labels_validation,
        set_type="validation",
        batch_size=args.batch,
        maximum=255,
        minimum=0,
    )

    test_set = ImageLoader(
        images_test,
        labels_test,
        set_type="test",
        batch_size=args.batch,
        maximum=255,
        minimum=0,
    )

    return train_set, validation_set, test_set
