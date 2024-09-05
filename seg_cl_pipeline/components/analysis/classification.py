import os
import json
import logging

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, accuracy_score


class Classification:
    """
    Classify the subjects based on the ROI intensities using optimal cutoffs, calculate AUC and ACC.
    """

    def __init__(self, intensities, config, outpath):
        self.results_dict = intensities
        self.config = config
        self.outpath = outpath

    def classify(self, subjects, labels):
        """
        subjects: list of subject IDs to classify. The IDs should be present in the results dictionary.
        labels: list of labels corresponding to the subjects.
        """

        # Check if all subjects are present in the results dictionary
        assert all(
            str(key) in self.results_dict for key in subjects
        ), "Results dictionary doesn't contain all the subjects"

        # Create a map of labels to subject groups
        label_groups = {0: [], 1: []}
        for subject, label in zip(subjects, labels):
            label_groups[label].append(subject)

        # Extract corresponding ROI intensities for the target scan
        def get_intensities(group):
            return [
                self.results_dict[str(subject)].get(str(self.config.target_scan))
                for subject in group
                if str(self.config.target_scan) in self.results_dict[str(subject)]
            ]

        intensities_healthy_target = get_intensities(label_groups[0])
        intensities_patient_target = get_intensities(label_groups[1])

        # Log missing intensities
        for group_name, group_size, intensities in [
            ("healthy", len(label_groups[0]), intensities_healthy_target),
            ("patient", len(label_groups[1]), intensities_patient_target),
        ]:
            if len(intensities) < group_size:
                logging.warning(
                    f"Missing intensities for {group_size - len(intensities)} {group_name} subjects in the target scan"
                )

        # Create label vector
        y_true = (
            [1] * len(intensities_patient_target)
            + [0] * len(intensities_healthy_target)
            if self.config.invert
            else [0] * len(intensities_patient_target)
            + [1] * len(intensities_healthy_target)
        )

        # Combine the intensities of the two groups
        y_scores = intensities_patient_target + intensities_healthy_target

        # calculate ROC curve, auc and cutoff value
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        cutoff_value = thresholds[np.argmax(tpr - fpr)]

        # predict the labels based on the cutoff value, calculate accuracy
        y_pred = np.array(y_scores) > cutoff_value
        y_pred = [int(x) for x in y_pred]
        accuracy = accuracy_score(y_true, y_pred)

        # plot the ROC curve
        self._plot_roc(fpr, tpr, roc_auc, cutoff_value)

        # export the results as json
        results_dict = {
            "roc_auc": roc_auc,
            "accuracy": accuracy,
        }
        with open(res_path := os.path.join(self.outpath, "cl_results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)

        logging.info(f"ROC AUC: {roc_auc:.3f}")
        logging.info(f"Accuracy: {accuracy:.3f}")
        logging.info(f"Classification done. Results saved to {self.outpath}\n")

        return [res_path]

    def _plot_roc(self, fpr, tpr, roc_auc, cutoff_value):
        """
        Plot the ROC curve and save it as a png file.
        fpr: list, false positive rates
        tpr: list, true positive rates
        roc_auc: float, area under the ROC curve
        cutoff_value: float, optimal cutoff value
        """

        # create ROC curve as plotly figure
        fig = go.Figure()

        # roc trace
        roc_trace = go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.3f})"
        )
        fig.add_trace(roc_trace)

        # random trace
        random_trace = go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")
        )
        fig.add_trace(random_trace)

        # add AUC and cutoff value as annotations
        fig.update_xaxes(title_text="FPR")
        fig.update_yaxes(title_text="TPR")
        fig.add_annotation(x=0.8, y=0.1, text=f"AUC={roc_auc:.3f}", showarrow=False)
        fig.add_annotation(
            x=0.8, y=0.0, text=f"Cutoff={cutoff_value:.3f}", showarrow=False
        )
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=False,
            title=f"{self.config.chromo}: {self.config.title}",
        )

        outfile = os.path.join(
            self.outpath,
            f"{self.config.chromo}_roc_curve.png",
        )

        fig.write_image(outfile)
