import logging

logging.getLogger("everett").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import comet_ml

from Code.Parser.parser import parse

from Code.Modules.A_split import split
from Code.Modules.B_prepare import prepare
from Code.Modules.C_pair import pair_dataset
from Code.Modules.D_train import train
from Code.Modules.E_test import test

if __name__ == "__main__":

    logging.basicConfig(level=logging.NOTSET)

    logging.info("Parsing requests")
    args = parse()

    logging.info("Loading datasets")
    train_set, validation_set, test_set = split(args["Split"])

    logging.info("Preparing network")
    Siamese = prepare(args=args, shape=train_set.shape)

    logging.info("Pairing datasets")
    (
        paired_train_images,
        paired_train_labels,
        paired_validation_images,
        paired_validation_labels,
        paired_test_images,
        paired_test_labels,
    ) = pair_dataset(train_set, validation_set, test_set)

    logging.info("Training network")
    experiment = train(
        Siamese,
        paired_train_images,
        paired_train_labels,
        paired_validation_images,
        paired_validation_labels,
        args,
    )

    logging.info("Testing network")
    test(
        Siamese,
        experiment,
        paired_test_images,
        paired_test_labels,
    )
