import argparse
from argparse import ArgumentParser


def parse():

    """Parse command line arguments"""

    # Initiate argparser
    parser = ArgumentParser()

    # Add group for general info on the project
    project_group = parser.add_argument_group(
        "Project", "Arguments for project options"
    )

    # Add arguments
    project_group.add_argument(
        "--project",
        const="SiameseNetwork",
        default="SiameseNetwork",
        nargs="?",
        type=str,
        help="String representing the name of the project (for logging purposes",
    )

    # Add group for splitting
    split_group = parser.add_argument_group("Split", "Arguments for splitting options")

    # Add arguments
    split_group.add_argument(
        "--test_size",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Float in (0,1) representing the percentage of sample to be kept for testing",
    )
    split_group.add_argument(
        "--validation_size",
        const=0.3,
        default=0.3,
        nargs="?",
        type=float,
        help="Float in (0,1) representing the percentage of sample to be kept for validation",
    )
    split_group.add_argument(
        "--seed",
        const=42,
        default=42,
        nargs="?",
        type=int,
        help="Integer representing the seed used in train/test/validation splitting",
    )
    split_group.add_argument(
        "--batch",
        const=32,
        default=32,
        nargs="?",
        type=int,
        help="Integer for batch size",
    )

    # Add group for preparation
    prepare_group = parser.add_argument_group(
        "Prepare", "Arguments for network preparation options"
    )

    # Add arguments
    prepare_group.add_argument(
        "--loss",
        const="BinaryCrossentropy",
        default="BinaryCrossentropy",
        nargs="?",
        type=str,
        choices=["BinaryCrossentropy"],
        help="Loss function to be used at training time",
    )
    prepare_group.add_argument(
        "--optimizer",
        const="Adam",
        default="Adam",
        nargs="?",
        type=str,
        choices=["Adam", "SGD"],
        help="Optimizer to be used at training time",
    )
    prepare_group.add_argument(
        "--learning_rate",
        const=0.001,
        default=0.001,
        nargs="?",
        type=float,
        help="Learning rate",
    )
    prepare_group.add_argument(
        "--filters",
        const=(4, 8),
        default=(4, 8),
        nargs="?",
        type=tuple,
        help="Filters to apply in feature extractor",
    )
    prepare_group.add_argument(
        "--dropout",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Float for dropout percentage at the end of feature extractor network",
    )

    prepare_group.add_argument(
        "--kernel_size_x",
        const=3,
        default=3,
        nargs="?",
        type=float,
        help="Integer for x-dimension of convolutional kernel",
    )

    prepare_group.add_argument(
        "--kernel_size_y",
        const=3,
        default=3,
        nargs="?",
        type=float,
        help="Integer for y-dimension of convolutional kernel",
    )

    prepare_group.add_argument(
        "--pool_size_x",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Integer for x-dimension of pooling kernel",
    )

    prepare_group.add_argument(
        "--pool_size_y",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Integer for y-dimension of  pooling kernel",
    )

    prepare_group.add_argument(
        "--latent",
        const=64,
        default=64,
        nargs="?",
        type=int,
        help="Integer for latent space dimension",
    )

    # Add group for Adam
    adam_group = parser.add_argument_group("Adam", "Arguments for Adam optimizer")

    # Add arguments
    adam_group.add_argument(
        "--beta_1",
        const=0.2,
        default=0.2,
        nargs="?",
        type=float,
        help="Beta_1 parameter for Adam",
    )
    adam_group.add_argument(
        "--beta_2",
        const=0.4,
        default=0.4,
        nargs="?",
        type=float,
        help="Beta_2 parameter for Adam",
    )
    adam_group.add_argument(
        "--epsilon",
        const=1e-8,
        default=1e-8,
        nargs="?",
        type=float,
        help="Epsilon to prevent division by zero",
    )

    # Add group for Adam
    sgd_group = parser.add_argument_group("SGD", "Arguments for SGD optimizer")

    # Add arguments
    sgd_group.add_argument(
        "--momentum",
        const=0.9,
        default=0.9,
        nargs="?",
        type=float,
        help="Momentum for SGD otpimizer",
    )

    # Add group for train
    train_group = parser.add_argument_group("Train", "Arguments for training options")

    # Add arguments
    train_group.add_argument(
        "--epochs",
        const=50,
        default=50,
        nargs="?",
        type=int,
        help="Integer for training epochs",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialise dictionary for groups of arguments
    arg_groups = {}

    # Add arguments to relevant group
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    return arg_groups
