import os
import logging

import numpy as np
from tqdm import tqdm
from skimage.transform import resize

from ..component import PipelineComponent


class Preprocessor(PipelineComponent):
    """
    Preprocesses the input data.
    """

    file_ending = "npy"

    def __init__(self, config):
        super().__init__(config)

    def process(self, input_filepaths):

        pbar = tqdm(total=len(input_filepaths))

        outfile_paths = []

        scan_loader = self.get_scan_loader()

        for filepath in input_filepaths:

            pbar.set_description(f"Preprocessing {os.path.basename(filepath)}")

            # Load the scan data
            scan_data = scan_loader.load_data(filepath)

            # flip / mirror / rotate the scan data if necessary
            if self.config.flipud:
                scan_data = np.flipud(scan_data)

            if self.config.fliplr:
                scan_data = np.fliplr(scan_data)

            if self.config.rotate:
                scan_data = np.rot90(scan_data, self.config.rotate)

            # resize image if necessary (always assumed to be square)
            if self.config.img_size != scan_data.shape[0]:
                scan_data = resize(
                    scan_data,
                    (self.config.img_size, self.config.img_size),
                    order=3,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=True,
                )

            outfile_path = os.path.join(self.outpath, os.path.basename(filepath)[:-4])

            np.save(outfile := f"{outfile_path}.{self.file_ending}", scan_data)

            outfile_paths.append(outfile)

            pbar.update(1)
        pbar.close()

        logging.info(
            f"Preprocessing done. {len(outfile_paths)} scans saved to {self.outpath}\n"
        )

        return outfile_paths
