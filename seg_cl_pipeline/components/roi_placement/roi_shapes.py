import os
from abc import abstractmethod, ABC

import numpy as np
import cv2

from ...components import px_roi_to_ithera


class ShapeFactory:
    """
    Factory class to create different ROI shapes.
    Supported shapes: Ellipse, Polygon
    """

    @staticmethod
    def create_shape(config):
        if config.roi_type == "ellipse":
            return Ellipse(config)
        elif config.roi_type == "polygon":
            return Polygon(config)
        else:
            raise ValueError(f"Unknown shape type: {config.roi_type}")


class ROIShape(ABC):
    """
    Base class for ROI shapes.
    """

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_roi(self, mask):
        """
        Places the ROI in the mask.
        Returns mask with ROI as np.array and ROI coordinates as dict.
        """
        pass


class Ellipse(ROIShape):
    """
    Class to place an ellipse as ROI on the segmentation mask.
    """

    def get_roi(self, mask):
        """
        Get the ellipse ROI as mask and annotation dictionary.
        """

        # Place the ellipse in the mask
        mask_with_ellipse, center, axes = self._place_ellipse_in_mask(mask)

        roi = np.array([[center[0], center[1], 0.0], [axes[0], axes[1], 0.0]])

        # Convert the ROI to Ithera format for the iannotation file export
        ithera_roi_center, ithera_roi_axes, _ = px_roi_to_ithera(
            roi, px_size=self.config.px_size, image_size=mask.shape
        )

        # Create the annotation dictionary, structured similar to the Ithera format
        annotation_dict = {
            "annotated_frames": "default",
            "annotation_source": f"SEG-CL-Pipeline_{os.environ['RUN_ID']}",
            "rois": [
                [
                    [ithera_roi_center[0], ithera_roi_center[1], 0.0],
                    [ithera_roi_axes[0], ithera_roi_axes[1], 0.002],
                ]
            ],
            "roi_types": ["Ellipse"],
        }

        return mask_with_ellipse, annotation_dict

    def _place_ellipse_in_mask(self, mask):
        """
        Place an ellipse into the binary mask. The ellipse is centered on the horizontal axis
        and placed as far up as possible within the foreground mask, with an optional margin.
        mask: A binary numpy array where the foreground is labeled with 1s and the background with 0s.
        Returns copy of the original mask with the ellipse drawn on it.
        """

        # Ensure mask is binary
        mask = mask.astype(np.uint8)

        # Convert the ROI size from the config from m to pixels
        ellipse_width_px = int(self.config.roi_ellipse_size[0] / self.config.px_size)
        ellipse_height_px = int(self.config.roi_ellipse_size[1] / self.config.px_size)

        # Convert the margin from the config from m to pixels
        margin_px = int(self.config.margin / self.config.px_size)

        # Find the topmost point of the foreground band along the central vertical axis
        center_x = mask.shape[1] // 2
        for center_y in range(mask.shape[0]):
            if mask[center_y, center_x] == 1:
                break

        # Adjust the center_y by the given margin
        center_y += margin_px

        # Calculate ellipse parameters
        center = (
            center_x,
            center_y + ellipse_height_px // 2,
        )  # Centered on horizontal axis, placed as far up as possible with margin
        axes = (
            ellipse_width_px // 2,
            ellipse_height_px // 2,
        )  # Use the provided width and height

        # Create a copy of the mask to draw the ellipse on
        mask_with_ellipse = mask.copy()

        # Draw the ellipse on the mask
        cv2.ellipse(
            mask_with_ellipse,
            center,
            axes,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=2,  # Use a distinct color value for the ellipse
            thickness=-1,
        )
        return mask_with_ellipse, center, axes


class Polygon(ROIShape):
    """
    Class to place a polygon as ROI on the segmentation mask.
    """

    def get_roi(self, mask):
        """
        Get the polygon ROI as mask and annotation dictionary.
        """

        # sensitivity map is the region in the scan where the sensitivity of the scanner
        # is assumed to be reasonably constant. The ROI can only be placed in this region.
        sensitivity_mask = np.zeros_like(mask, dtype=bool)
        y_min, y_max, x_min, x_max = [
            int(value / self.config.px_size) for value in self.config.sensitivity_map
        ]
        sensitivity_mask[y_min:y_max, x_min:x_max] = True

        # Trim the mask to the sensitivity region
        trimmed_mask = np.where(sensitivity_mask, mask, False)

        # Trim the ROI to the desired height. This should keep the size of the ROI approximately constant.
        trimmed_mask = self._trim_roi_to_height(trimmed_mask)

        # overlay seg mask (1) and generated ROI (2)
        seg_mask_roi_overlay = np.where(trimmed_mask, 2, mask)

        return seg_mask_roi_overlay, {}

    def _trim_roi_to_height(self, mask):
        """
        Trims the roi object in the segmentation mask to a given height.
        mask: Segmentation mask containing roi (True) and background (False).
        Returns Modified segmentation mask with the roi trimmed.
        """

        # Convert the desired height from m to pixels
        height_px = int(self.config.roi_height / self.config.px_size)

        # Find the vertical center of the image
        center_column = mask.shape[1] // 2

        # Determine the starting row of the foreground object in the center column
        foreground_rows = np.where(mask[:, center_column])[0]
        if len(foreground_rows) == 0:
            return mask  # No foreground found, return the original mask

        start_row = foreground_rows[0]
        end_row = start_row + height_px

        # Trim the foreground object
        trimmed_mask = mask.copy()
        trimmed_mask[end_row:, :] = False

        return trimmed_mask
