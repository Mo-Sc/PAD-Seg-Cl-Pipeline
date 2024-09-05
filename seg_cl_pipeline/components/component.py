from abc import ABC
import logging
import os
from .data_loader import ScanLoaderFactory, ROILoaderFactory


class PipelineComponent(ABC):
    """
    Base class for all pipeline components.
    """

    file_ending = None

    def __init__(self, config):
        self.config = config
        self.outpath = os.path.join(os.environ["OUTPUT_BASEPATH"], config.step_name)
        os.makedirs(self.outpath, exist_ok=True)
        logging.info(f"--- {self.config.step_name.upper()} ---")

    def skip(self, overwrite_path=None):
        """
        Skip this step. Return a list of files in the given path or in the default output path.
        overwrite_path: path to use instead of the default output path.
        """

        if self.file_ending is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'file_ending'"
            )

        logging.info(
            f"{self.config.step_name} Skipped. Using files from {overwrite_path if overwrite_path else self.outpath}\n"
        )
        if overwrite_path:
            return [
                os.path.join(overwrite_path, file)
                for file in os.listdir(overwrite_path)
                if file.endswith(self.file_ending)
            ]
        else:
            return [
                os.path.join(self.outpath, file)
                for file in os.listdir(self.outpath)
                if file.endswith(self.file_ending)
            ]

    def get_scan_loader(self, src_format=None):
        """
        Returns a scan loader to load us / oa scans from different file formats, as defined in the config.
        src_format can be used to override the config setting.
        """

        loader = ScanLoaderFactory.get_loader(
            self.config.src_format if src_format is None else src_format, self.config
        )

        return loader

    def get_roi_loader(self, src_format=None):
        """
        Returns a scan loader to load roi masks from different file formats, as defined in the config.
        src_format can be used to override the config setting.
        """
        loader = ROILoaderFactory.get_loader(
            self.config.roi_src_format if src_format is None else src_format,
            self.config,
        )
        return loader
