# import os
# import numpy as np
# import h5py
import math
import numpy as np


def px_roi_to_ithera(roi, px_size=0.0001, image_size=(400, 400)):
    """
    Used for exporting the roi to ithera format
    Takes ROI in pixel and cv2 ellipse coords and converts it to m and ithera coordinate system
    ithera coordinates: relative to the center of the image (top right is positive)

    """

    # Make sure values are floats
    roi_center = [float(value) for value in roi[0]]
    roi_axes_length = [float(value) for value in roi[1]]

    # shift ellipse center so it is relative to the center of the image
    roi_center = [roi_center[0] - image_size[0] / 2, roi_center[1] - image_size[1] / 2]

    # Calculate the position and size of the ROI
    roi_pos = [roi_center[0] - roi_axes_length[0], roi_center[1] - roi_axes_length[1]]
    roi_size = [roi_axes_length[0] * 2, roi_axes_length[1] * 2]

    # Convert the values to m
    roi_pos = [value * px_size for value in roi_pos]
    roi_size = [value * px_size for value in roi_size]

    area = math.pi * roi_size[0] / 2 * roi_size[1] / 2

    return roi_pos, roi_size, area


def ithera_roi_to_px(roi, px_size=0.0001, image_size=(400, 400), top_left=False):
    """
    Used for exporting the roi to ithera format
    Takes ROI in m and ithera coordinate system and converts it to pixel and cv2 ellipse coords
    roi: list of two lists / 2x3 np.array, first list contains the position of the ROI, second list contains the size of the ROI
    """

    # TODO: optional: shift back from center to top left corner

    # Make sure values are floats
    roi_pos = [float(value) for value in roi[0][0:2]]
    roi_size = [float(value) for value in roi[1][0:2]]

    # Calculate the center and half axes length of the ellipse
    roi_center = [roi_pos[0] + roi_size[0] / 2, roi_pos[1] + roi_size[1] / 2]
    roi_axes_length = [roi_size[0] / 2, roi_size[1] / 2]

    area = math.pi * roi_axes_length[0] * roi_axes_length[1]

    # Convert the values to pixels
    roi_center = [int(value / px_size) for value in roi_center]
    roi_axes_length = [int(value / px_size) for value in roi_axes_length]

    # shift ellipse center so it is relative to the top left corner of the image
    if top_left:
        roi_center = [
            roi_center[0] + image_size[0] / 2,
            roi_center[1] + image_size[1] / 2,
        ]

    return roi_center, roi_axes_length, area


def calculate_mso2(hb_scan, hbo2_scan, nan_invalid=True):
    """
    Calculate the mso2 value from the scan as mso2 = HbO2 / (Hb + HbO2)
    if nan_invalid is set to True, all values outside the range [0, 1] are set to nan
    """

    thb = hb_scan + hbo2_scan
    thb[thb == 0] = np.nan  # to avoid div by 0

    mso2 = hbo2_scan / thb

    if nan_invalid:
        mso2[mso2 > 1] = np.nan
        mso2[mso2 < 0] = np.nan

    mso2[np.isnan(mso2)] = 0

    return mso2
