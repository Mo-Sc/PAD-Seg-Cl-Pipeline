from abc import abstractmethod
import os
import logging

from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from ...component import PipelineComponent


class Segmentation(PipelineComponent):
    """
    Base class for segmentation models.
    """

    file_ending = "npy"
    orig_img_shape = None

    def __init__(self, config):
        super().__init__(config)

        # check if model weights are available
        if not os.path.exists(config.model_config.model_weights):
            raise FileNotFoundError(
                f"Path to model weights not found for {config.model_config.model_weights}"
            )
        self.weights_path = config.model_config.model_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def initialize(self):
        """
        Load the model weights, set required parameters, etc.
        """
        pass

    @abstractmethod
    def preprocess(self, data):
        """
        Model specific preprocessing.
        Returns preprocessed image path(s) as list.
        data: list, contains image filepath(s)
        """
        pass

    def inference(self, data):
        """
        Perform inference on preprocessed data
        Returns segmentation mask path(s) as list.
        data: list, contains preprocessed image filepath(s)
        """
        pass

    @abstractmethod
    def postprocess(self, data):
        """
        Model specific postprocessing.
        Returns postprocessed image path(s) as list.
        data: list, contains segmentation mask filepath(s)
        """
        pass

    def segment(self, data):
        """
        Segment the images.
        Applies model specific preprocessing, inference and postprocessing.
        Returns path(s) to postprocessed segmentation masks.
        Can be used in instance or batch mode.
        instance: send image to segmentation model one by one
        batch: send all images to segmentation model at once
        For speedup, use batch mode (if supported by model).
        data: list, contains input image filepath(s)
        """

        # collect filepaths of postprocessed segmentation masks
        result_filepaths = []

        if self.config.model_config.mode == "instance":
            # process images one by one

            pbar = tqdm(total=len(data))

            for img_path in data:

                pbar.set_description(f"Segmenting {os.path.basename(img_path)}")

                # might be needed by model
                self.study_id = os.path.basename(img_path).split("_")[0]
                self.scan_id = os.path.basename(img_path).split("_")[1].split(".")[0]

                result_filepaths.extend(self._run([img_path]))

                pbar.update(1)
            pbar.close()

        elif self.config.model_config.mode == "batch":
            # process all images at once

            result_filepaths.extend(self._run(data))

        else:
            raise ValueError(
                "Invalid data processing mode. Choose 'instance' or 'batch'."
            )

        if os.environ["MODE"] == "DEBUG":
            # save the image, mask and overlay for debugging as png
            self._plot_result(data, result_filepaths)

        logging.info(
            f"Segmentation done. {len(result_filepaths)} masks saved to {self.outpath}\n"
        )

        return result_filepaths

    def _run(self, data):
        """
        Perform one segmentation step, i.e. preprocess, inference and postprocess.
        data: list, contains image filepath(s)
        """
        preprocessed_data = self.preprocess(data)
        segmask_data = self.inference(preprocessed_data)
        postprocessed_data = self.postprocess(segmask_data)

        return postprocessed_data

    def _plot_result(self, data, result_filepaths):
        """
        Save the image, mask and overlay as png for debugging.
        data: list, contains image filepath(s)
        result_filepaths: list, contains segmentation mask filepath(s)
        """
        for img_path, mask_path in zip(data, result_filepaths):
            assert (fn := os.path.basename(img_path)) == os.path.basename(mask_path)
            assert os.path.exists(img_path)
            assert os.path.exists(mask_path)

            fig, axs = plt.subplots(1, 3, figsize=(9, 3))
            fig.suptitle(f"{self.config.model} Segmentation Result: {fn}")

            # load the us image
            axs[0].imshow(img := np.load(img_path), cmap="gray")
            axs[0].set_title("Ultrasound Image")
            axs[0].axis("off")

            # load the mask
            axs[1].imshow(mask := np.load(mask_path), cmap="viridis")
            axs[1].set_title("Predicted Mask")
            axs[1].axis("off")

            # overlay the mask on the image
            axs[2].imshow(img, cmap="gray")
            axs[2].imshow(mask, alpha=0.5, cmap="viridis")
            axs[2].set_title("Segmentation Mask Overlay")
            axs[2].axis("off")

            plt.tight_layout()
            # save to png
            plt.savefig(
                os.path.join(
                    self.outpath,
                    f"{os.path.basename(img_path).replace('.npy', '.png')}",
                )
            )
            plt.close()
