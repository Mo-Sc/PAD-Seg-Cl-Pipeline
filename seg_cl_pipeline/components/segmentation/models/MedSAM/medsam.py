import os
import logging

import torch
import numpy as np
from skimage.transform import resize

# workaround for sam code to work without changes
# sys.path.append(os.path.join(os.path.dirname(__file__), "assets"))

from .assets.medsam_model import MedSAMModel
from .assets.segment_anything import sam_model_registry
from ..segmentation import Segmentation


class MedSAM(Segmentation):
    """
    Interface to MedSAM segmentation model.
    """

    def initialize(self):

        sam_model = sam_model_registry[self.config.model_config.configuration](
            checkpoint=self.weights_path
        )
        medsam_model = MedSAMModel(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        ).to(self.device)

        logging.info(f"Loaded MedSAM model from {self.weights_path}")

    def preprocess(self, img_path):
        """
        Adapted from preprocessing in MedSAM repo (pre_grey_rgb.py).
        todo: check whats necessary
        """

        image_data = np.load(img_path)

        if self.orig_img_shape is None:
            self.orig_img_shape = image_data.shape

        # resize to model input size
        image_data = resize(
            image_data,
            (self.config.model_config.img_size, self.config.model_config.img_size),
            order=3,
            mode="constant",
            preserve_range=True,
            anti_aliasing=True,
        )

        # some scaling and normalization
        if np.max(image_data) > 255.0:
            image_data = np.uint8(
                (image_data - image_data.min())
                / (np.max(image_data) - np.min(image_data))
                * 255.0
            )

        # add third dimension for compatibility
        if len(image_data.shape) == 2:
            image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)

        # intensity_cutoff
        lower_bound, upper_bound = np.percentile(
            image_data[image_data > 0], 0.5
        ), np.percentile(image_data[image_data > 0], 99.5)
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        image_data_pre[image_data == 0] = 0
        image_data_pre = np.uint8(image_data_pre)

        image_data_pre = image_data_pre / 255.0

        # keep image for visualizations
        if os.environ["MODE"] == "DEBUG":
            self.img = image_data_pre

        # Convert to tensor and add batch dimension
        image_data_pre = (
            torch.from_numpy(image_data_pre)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        return image_data_pre

    def inference(self, img_pre):

        # Define dummy bounding box as whole image
        # This is a placeholder, adjust as needed
        dummy_box = np.array([[0, 0, img_pre.shape[2], img_pre.shape[3]]])

        with torch.no_grad():
            mask = self.model(img_pre, dummy_box)
            # Apply sigmoid to get probabilities
            mask = torch.sigmoid(mask)

        return mask

    def postprocess(self, mask):

        # Remove batch and channel dimensions
        mask = mask.squeeze().cpu().numpy()
        #  Binarize the mask
        mask = (mask > 0.5).astype(np.uint8)

        outfile = os.path.join(
            self.outpath,
            f"{self.study_id}_{self.scan_id}.npy",
        )

        if os.environ["MODE"] == "DEBUG":
            self.mask_post = mask

        # resize back to original shape
        if self.orig_img_shape != mask.shape:
            mask = resize(
                mask,
                self.orig_img_shape,
                order=0,
                mode="constant",
                preserve_range=True,
                anti_aliasing=False,
            )
        np.save(outfile, mask)

        return outfile
