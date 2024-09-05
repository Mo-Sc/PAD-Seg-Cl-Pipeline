import os
import logging

from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import numpy as np

from ..component import PipelineComponent


class PostProcessor(PipelineComponent):
    """
    Postprocesses the segmentation masks
    """

    file_ending = "npy"

    def __init__(self, config):
        super().__init__(config)

    def process(self, input_filepaths):

        pbar = tqdm(total=len(input_filepaths))

        outfile_paths = []

        for npy_file in input_filepaths:

            pbar.set_description(f"Postprocessing {os.path.basename(npy_file)}")

            mask = np.load(npy_file)

            # Perform connected component analysis to keep only the largest connected region
            mask_post, _ = self._connected_component_analysis(mask, mode="islands")

            outfile_path = os.path.join(self.outpath, os.path.basename(npy_file)[:-4])

            # if in debug mode, save the original and postprocessed masks as images
            if os.environ["MODE"] == "DEBUG":
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(mask)
                plt.title("Original Mask")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(mask_post)
                plt.title("Postprocessed Mask")
                plt.axis("off")
                plt.savefig(outfile_path + ".png")
                plt.close()

            np.save(outfile := f"{outfile_path}.{self.file_ending}", mask_post)
            outfile_paths.append(outfile)

            pbar.update(1)
        pbar.close()

        logging.info(
            f"Postprocessing done. {len(outfile_paths)} masks saved to {self.outpath}\n"
        )

        return outfile_paths

    @staticmethod
    def _connected_component_analysis(mask, mode):
        """
        Perform connected component analysis on the mask.
        Keeps only the largest connected region in the mask.
        Returns a mask with only the largest component and a boolean indicating if the mask has been modified.
        """

        assert mode in ["holes", "islands"], "Mode must be 'holes' or 'islands'"

        correct_holes = mode == "holes"

        # XOR the mask with correct_holes to get the working mask
        working_mask = (correct_holes ^ mask).astype(np.uint8)

        # Perform connected component analysis
        n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, 8)

        if n_labels <= 1:
            # Only the background is present, nothing to keep
            return mask, False

        # Find the label of the largest component (excluding the background)
        largest_label = np.argmax(stats[1:, -1]) + 1

        # Create a mask with only the largest component
        mask = regions == largest_label

        return mask, True
