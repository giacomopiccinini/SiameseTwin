import numpy as np


class Metric:

    """Class for evaluating results of predictions in classification"""

    def __init__(self, y_truth, y_pred):

        # Store ground truths and predictions
        self.y_truth = y_truth
        self.y_pred = y_pred

    def compute_binary_crossentropy(self):

        """Compute binary cross entropy"""

        self.y_pred = np.clip(self.y_pred, 1e-7, 1 - 1e-7)

        # Compute first term
        term_0 = (1 - self.y_truth) * np.log(1 - self.y_pred + 1e-7)

        # Compute second term
        term_1 = self.y_truth * np.log(self.y_pred + 1e-7)

        # Compute total cross entropy
        self.binary_crossentropy = -np.mean(term_0 + term_1)

    def compute_classification_metric(self):

        """Compute accuracy"""

        # Clip to 0 or 1
        y_pred = np.where(self.y_pred > 0.5, 1, 0)

        y_truth = self.y_truth

        # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
        TP = np.sum(np.logical_and(y_pred == 1, y_truth == 1))

        # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
        TN = np.sum(np.logical_and(y_pred == 0, y_truth == 0))

        # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
        FP = np.sum(np.logical_and(y_pred == 1, y_truth == 0))

        # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
        FN = np.sum(np.logical_and(y_pred == 0, y_truth == 1))

        # Compute metrics
        self.accuracy = (TP + TN) / (TP + TN + FP + FN)
        self.precision = TP / (TP + FP)
        self.recall = TP / (TP + FN)
        self.f1 = 2 / (1 / self.precision + 1 / self.recall)

    def evaluate(self):

        """Evaluate all metrics"""

        # Compute all metrics
        self.compute_binary_crossentropy()
        self.compute_classification_metric()

        # Store all metrics in a dictionary
        metrics = {}

        metrics["binary_crossentropy"] = self.binary_crossentropy
        metrics["accuracy"] = self.accuracy
        metrics["precision"] = self.precision
        metrics["recall"] = self.recall
        metrics["f1"] = self.f1

        return metrics
