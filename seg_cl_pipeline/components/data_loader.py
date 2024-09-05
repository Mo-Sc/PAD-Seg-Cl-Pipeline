import os
import json
from abc import ABC, abstractmethod
import logging

import numpy as np
import nrrd
import nibabel as nib
import cv2

from . import ithera_roi_to_px, calculate_mso2


# --- data loaders for OA and US scans ---
class DataLoader(ABC):
    """
    Base class for data loaders.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load_data(self, filepath):
        """
        Load the scan data from different file formats.
        """
        pass

    def load_channel(self, scan_data):
        """
        Extracts the desired channel from the loaded scan data based on the channel names in the config.
        """

        if self.config.src_type == "mSO2":
            assert (
                len(scan_data.shape) == 3
            ), "scan has to have channel dim for mso2 calculation"
            # calculate mSO2 from Hb and HbO2
            scan_img = calculate_mso2(
                scan_data[self.config.src_channel_names.index("Hb")],
                scan_data[self.config.src_channel_names.index("HbO2")],
            )
        else:
            # in case of multiple channels, find dimension for desired channel
            if len(scan_data.shape) == 2:
                scan_img = scan_data
            else:
                scan_img = scan_data[
                    self.config.src_channel_names.index(self.config.src_type)
                ]

        return scan_img


class NpyLoader(DataLoader):
    """
    Data loader for npy files.
    """

    def load_data(self, filepath):
        try:
            scan_data = np.load(filepath)
            scan_img = self.load_channel(scan_data)
        except:
            raise ValueError(f"Couldnt load npy file {filepath}")

        return scan_img


class NrrdLoader(DataLoader):
    """
    Data loader for nrrd files.
    """

    def load_data(self, filepath):
        try:
            scan_data, _ = nrrd.read(filepath)
            scan_img = self.load_channel(scan_data)
        except:
            raise ValueError(f"Couldnt load nrrd file {filepath}")
        return scan_img


class NiftiLoader(DataLoader):
    """
    Data loader for nifti files.
    """

    def load_data(self, filepath):
        try:
            scan_data = nib.load(filepath).get_fdata()

            # workaround cause currently nii files have channel dim at the end
            scan_data = np.moveaxis(scan_data, -1, 0)

            if scan_data.shape[0] > 20:
                logging.warning(
                    f"Nifti Scan seems to have more than 20 channels. Maybe wrong order of dimensions? Shape: {scan_data.shape}"
                )

            scan_img = self.load_channel(
                scan_data,
            )
        except:
            raise ValueError(f"Couldnt load nifti file {filepath}")
        return scan_img


class ScanLoaderFactory:
    """
    Returns the appropriate data loader based on the file format.
    """

    @staticmethod
    def get_loader(file_format, config):
        if file_format == "npy":
            return NpyLoader(config)
        elif file_format == "nrrd":
            return NrrdLoader(config)
        elif file_format in ["nii", "nii.gz"]:
            return NiftiLoader(config)
        else:
            raise ValueError(f"Unsupported file format for OA/US: {file_format}")


# --- data loaders for ROIs ---


class NpyROILoader(DataLoader):
    """
    Data loader for npy ROI files.
    """

    def load_data(self, filepath):
        try:
            scan_data = np.load(filepath)
        except:
            raise ValueError(f"Couldnt open npy file {filepath}")
        return scan_data


class JsonROILoader(DataLoader):
    """
    Data loader for JSON ROI files.
    """

    def load_data(self, filepath):
        """
        Loads the ROI coordinates from the JSON file.
        Creates a binary mask with the ROI shape.
        Currently only supports Ellipse ROI.
        """

        try:
            rois_dict = json.load(open(filepath))
            # should contain only one ROI. List is only for iLabs compatibility
            roi = rois_dict["rois"][0]
            roi_type = rois_dict["roi_types"][0]
            # annotated_frames = rois_dict["annotated_frames"]
            # annotation_source = rois_dict["annotation_source"]
        except:
            raise ValueError(f"Couldnt load ROI from json file {filepath}")

        if roi_type != "Ellipse":
            raise ValueError(f"Unsupported ROI type for JSON: {roi_type}")

        else:

            # convert from ithera format to px coordinates
            roi_center, roi_size, _ = ithera_roi_to_px(roi)

            # Create an empty image with the desired size
            mask_size = (
                self.config.img_size,
                self.config.img_size,
            )  # is there another way of getting the size?
            roi_mask = np.zeros((mask_size), dtype=np.uint8)

            # draw the ellipse on the mask
            cv2.ellipse(roi_mask, roi_center, roi_size[0:2], 0, 0, 360, 255, -1)

            return roi_mask


class ROILoaderFactory:
    """
    Returns the appropriate data loader for the ROIs based on the file format.
    """

    @staticmethod
    def get_loader(file_format, config):
        if file_format == "npy":
            return NpyROILoader(config)
        elif file_format == "json":
            return JsonROILoader(config)
        else:
            raise ValueError(f"Unsupported file format for ROI: {file_format}")
