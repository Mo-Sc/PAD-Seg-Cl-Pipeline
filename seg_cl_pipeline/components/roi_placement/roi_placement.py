import os
import json
import logging

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..component import PipelineComponent
from .roi_shapes import ShapeFactory


class ROIPlacement(PipelineComponent):
    """
    Places the ROI on the segmentation mask.
    """

    file_ending = "npy"

    def __init__(self, config):
        super().__init__(config)

        # create the ROI object of the specified type
        self.roi_shape = ShapeFactory.create_shape(config)

    def place_roi(self, input_filepaths):
        """
        Places the ROI on the segmentation mask.
        Returns the filepaths of the saved ROIs.
        """

        pbar = tqdm(total=len(input_filepaths))
        outfile_paths = []

        for npy_file in input_filepaths:

            # required for loading the US image for overlay when in debug mode
            study_id = os.path.basename(npy_file).split("_")[0]
            scan_id = os.path.basename(npy_file).split("_")[1].split(".")[0]

            pbar.set_description(f"ROI-Placement {os.path.basename(npy_file)}")

            mask = np.load(npy_file)
            outfile_path = os.path.join(self.outpath, os.path.basename(npy_file)[:-4])

            # get the ROI as mask and dict from the specified shape
            roi_mask, roi_dict = self.roi_shape.get_roi(mask)

            # if in debug mode, save the overlay of the US image and the ROI as png
            if os.environ["MODE"] == "DEBUG":
                preprocessed_us_file = os.path.join(
                    os.environ["OUTPUT_BASEPATH"],
                    self.config.preprocessed_us_folder,
                    f"{study_id}_{scan_id}.npy",
                )
                us_image = np.load(preprocessed_us_file)
                self._plot_us_overlay(
                    us_image,
                    roi_mask,
                    outfile_path + ".png",
                )

            # export into json format (in m)
            if self.config.json_export:

                if self.config.roi_type != "ellipse":
                    raise NotImplementedError(
                        f"JSON export is only supported for 'ellipse' ROI type, not {self.config.roi_type}"
                    )
                with open(outfile_path + ".json", "w") as f:
                    json.dump(roi_dict, f, indent=4)

            # export into format readable by ithera iLabs software (in m)
            if self.config.iannotation_export:
                if self.config.roi_type != "ellipse":
                    raise NotImplementedError(
                        f"iAnnotation export is only supported for 'ellipse' ROI type, not {self.config.roi_type}"
                    )
                self._export_iannotation(roi_dict, outfile_path + ".iannotation")

            # only ROI should be passed on -> remove segmask (1), keep ROI (2)
            roi_mask_roi_only = np.where(roi_mask == 2, 1, 0)

            np.save(outfile := f"{outfile_path}.{self.file_ending}", roi_mask_roi_only)

            outfile_paths.append(outfile)

            pbar.update(1)
        pbar.close()

        logging.info(
            f"ROI placement done. {len(outfile_paths)} ROIs saved to {self.outpath}\n"
        )

        return outfile_paths

    @staticmethod
    def _plot_us_overlay(image, mask, outfile, fg_color=(0, 1, 0), roi_color=(1, 0, 0)):
        """
        Save the US image with the ROI overlay as png.
        image: np.array, US image
        mask: np.array, ROI mask
        outfile: str, path to save the image
        fg_color: tuple, color of the foreground (default: green)
        roi_color: tuple, color of the ROI (default: red)
        """

        plt.imshow(image, cmap="gray")

        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        for i in range(3):
            overlay[..., i] = np.where(mask == 1, fg_color[i], 0)
            overlay[..., i] = np.where(mask == 2, roi_color[i], overlay[..., i])

        plt.imshow(overlay, alpha=0.5)
        plt.axis("off")
        plt.title("US - Segmentation Mask - ROI Overlay: " + os.path.basename(outfile))

        plt.savefig(outfile)
        plt.close()

    @staticmethod
    def _export_iannotation(roi_dict, outfile, frames=[12], scan_hash=""):
        """
        Export the ROI data into a iannotation file (json structure) that can be read by the iLabs software.
        frames: list of frames / sweeps that the annotation is valid for (default: [12])
        """

        rois = roi_dict["rois"]
        roi_types = roi_dict["roi_types"]
        annotated_frames = roi_dict["annotated_frames"]
        annotation_source = roi_dict["annotation_source"]

        if not isinstance(annotated_frames, list):
            annotated_frames = frames

        roi_list_ian = []

        # Iterate through ROIs
        for roi, type in zip(rois, roi_types):
            pos = roi[0]
            size = roi[1]

            roi_ian = {
                "__classname": "iROI",
                "type": type,
                "pos": f"Point({pos[0]}, {pos[1]})",
                "size": f"Point({size[0]}, {size[1]})",
            }
            roi_list_ian.append(roi_ian)

        annotation = {
            "__classname": "iAnnotation",
            "ROIList": [roi_list_ian],
            "Sweeps": annotated_frames,
            "Source": annotation_source,
        }

        iannotation_dict = {
            "__classname": "iAnnotationListWrapper",
            "Annotations": [annotation],
            "ScanHash": scan_hash,
        }

        with open(outfile, "w") as f:
            json.dump(iannotation_dict, f, indent=4)
