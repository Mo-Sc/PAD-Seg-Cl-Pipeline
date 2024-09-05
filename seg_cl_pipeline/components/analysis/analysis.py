import os
import json
import logging

import numpy as np
from skimage.transform import resize
from tqdm import tqdm


from ..component import PipelineComponent
from .classification import Classification


class Analysis(PipelineComponent):
    """
    Class to extract ROI intensities from OA scans and perform different types of analysis.
    """

    file_ending = "json"

    def __init__(self, config):
        super().__init__(config)

        self.roi_intensities = None

    def extract_intensities(self, input_files, overwrite=True):
        """
        Extract MSOT intensities from the OA scans in the ROIs.
        """

        # does it make sense to extract intensities for all input files or rather just subjects?

        # roi intensities are saved to a json file
        res_path = os.path.join(self.outpath, "roi_intensities.json")

        # in case of multiple downstream analysis tasks, this step can be skipped if the intensities are already extracted
        if self.roi_intensities is not None and not overwrite:
            logging.info(
                "ROI intensities already extracted and overwrite=False. Skipping."
            )
            return [res_path]

        roi_intensities = {}

        pbar = tqdm(total=len(input_files))

        # data loaders for the OA scans and the ROIs
        scan_loader = self.get_scan_loader()
        roi_loader = self.get_roi_loader()

        for filepath in input_files:

            roi_filename = os.path.basename(filepath).split(".")[0]

            subject_id, scan_id = roi_filename.split("_")

            pbar.set_description(f"Extracting ROI intensity from {roi_filename}")

            # load the roi mask
            roi_mask = roi_loader.load_data(filepath)

            # load the correpsponding OA scan
            oa_scan_filepath = (
                os.path.join(self.config.src_path, roi_filename)
                + f".{self.config.src_format}"
            )

            # load the OA scan
            oa_scan = scan_loader.load_data(oa_scan_filepath)

            # extract the intensity in the ROI from the OA scan based on the specified metric
            roi_intensity = self._summarize_roi_intensity(roi_mask, oa_scan)

            # add intensity to dict with subject_id and scan_id as keys. If subject_id doesn't exist, create it
            if subject_id not in roi_intensities:
                roi_intensities[subject_id] = {}
            roi_intensities[subject_id][scan_id] = float(roi_intensity)

            pbar.update(1)
        pbar.close()

        logging.info(f"ROI intensities extracted from {len(input_files)} ROIs")

        # save the results to a json file
        with open(res_path, "w") as f:
            json.dump(roi_intensities, f, indent=4)

        logging.info(f"ROI intensities saved to {res_path}")

        self.roi_intensities = roi_intensities

        # return path as list for consistency with other components
        return [res_path]

    def _summarize_roi_intensity(self, mask, scan):
        """
        Overlaps the ROI mask with the OA scan and summarizes intensity values in ROI.
        """

        roi_mask = mask.copy()
        oa_scan = scan.copy()

        # resize the mask to the scan size if necessary
        if roi_mask.shape != oa_scan.shape:
            roi_mask = resize(
                roi_mask,
                oa_scan.shape,
                anti_aliasing=False,
            )

        roi_mask = roi_mask.astype(bool)

        # set intensities outside the ROI to 0
        oa_scan[~roi_mask] = 0

        if self.config.clip_negative and np.min(oa_scan) < 0:
            # negative intensities are not valid
            logging.info("Negative ROI intensiites clipped to 0")
            oa_scan = np.clip(oa_scan, 0, np.inf)

        if self.config.exclude_zeros:
            # exclude zero intensities from the statistics
            oa_scan = oa_scan[oa_scan != 0]

        if np.sum(roi_mask) == 0:
            logging.warning("No intensities in ROI")
            return 0

        # summarize the intensities based on the specified metric
        if self.config.metric == "mean":
            return np.nanmean(oa_scan)
        elif self.config.metric == "median":
            return np.nanmedian(oa_scan)
        elif self.config.metric == "sum":
            return np.nansum(oa_scan)
        elif self.config.metric == "max":
            return np.nanmax(oa_scan)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def analyze(self, subjects, labels):
        """
        Perform analysis on the extracted ROI intensities.
        """

        logging.info(f"Performing analysis: {self.config.task}")

        # create output directory for the specific analysis task
        outpath = os.path.join(self.outpath, self.config.task)
        os.makedirs(outpath, exist_ok=True)

        # choose the analysis task based on the config, perform the analysis and return the results
        if self.config.task == "classification":
            # classify the subjects based on the ROI intensities using optimal cutoffs, calculate AUC and ACC
            classifier = Classification(
                self.roi_intensities, self.config.task_config, outpath
            )
            return classifier.classify(subjects, labels)
        elif self.config.analysis_type == "group_comparison":
            # Compare the ROI intensities between the groups using boxplots, perform statistical tests
            raise NotImplementedError(
                "Group comparison analysis not moved to this project yet"
            )
        elif self.config.analysis_type == "time_curve":
            # plot the MSOT intensities over time for each subject. Summarize time curves for each group
            raise NotImplementedError(
                "Time curve analysis not moved to this project yet"
            )
        else:
            raise ValueError("Unknown analysis type")
