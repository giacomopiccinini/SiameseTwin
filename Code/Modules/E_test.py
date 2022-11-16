import numpy as np
import pandas as pd
from Code.Metric.Metric import Metric


def test(Siamese, experiment, paired_test_images, paired_test_labels):

    # Load best model
    Siamese.load_weights(f"Output/{experiment.get_name()}/Checkpoints/model.hdf5")

    with experiment.test():

        # Get ground truth
        y_truth = paired_test_labels.reshape(-1)

        # Predict values
        y_pred = Siamese.predict(paired_test_images).reshape(-1)

        # Instantiate metric object
        metric = Metric(y_truth=y_truth, y_pred=y_pred)

        # Evaluate metrics
        metrics = metric.evaluate()

        # Log metrics
        experiment.log_metrics(metrics)

        # End experiment
        experiment.end()

        # Clip y_pred
        y_pred = np.where(y_pred > 0.5, 1, 0)

        # Create pandas data frame
        df = pd.DataFrame({"Ground Truth": y_truth, "Prediction": y_pred})

        # Write results
        df.to_excel(
            f"Output/{experiment.get_name()}/predictions.xlsx",
            sheet_name="sheet1",
            index=False,
        )
