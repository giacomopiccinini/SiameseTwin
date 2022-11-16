from Code.Selector.Selector import Selector
from Code.Networks.Siamese import SiameseNetwork


def prepare(args, shape):

    # Load selectors of Loss function and Optimizer
    Loss = Selector("loss").select(args["Prepare"].loss)()
    Optimizer = Selector("optimizer").select(args["Prepare"].optimizer)(
        learning_rate=args["Prepare"].learning_rate,
        **vars(args[args["Prepare"].optimizer])
    )

    if len(shape) == 2:
        new_shape = (shape[0], shape[1], 1)
    else:
        new_shape = shape

    # Load Regression CNN
    Siamese = SiameseNetwork(shape=new_shape)

    # Compile and summarise model
    Siamese.compile(optimizer=Optimizer, loss=Loss, metrics=["accuracy"])
    Siamese.summary()

    return Siamese
