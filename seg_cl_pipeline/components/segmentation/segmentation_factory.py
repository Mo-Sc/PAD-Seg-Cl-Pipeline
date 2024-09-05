import logging
from .models.nnUNet import set_nnunet_env_vars


class SegmentationFactory:
    """
    Select and create the segmentation model based on the configuration.
    """

    @staticmethod
    def create_model(config):
        if config.model == "MedSAM":

            if config.model_config.mode == "batch":
                raise ValueError(
                    "MedSAM does not support batch mode. Please use instance mode."
                )

            from .models.MedSAM.medsam import MedSAM

            return MedSAM(config)

        elif config.model == "nnUNet":

            if config.model_config.mode == "instance":
                logging.warning(
                    "Running nnUNet in instance mode. This is extremely slow. Consider using batch mode."
                )

            # Set environment variables required by nnUNet
            set_nnunet_env_vars(config)

            from .models.nnUNet.nnunet import nnUNet

            return nnUNet(config)
        # Add more models if needed
        else:
            raise ValueError(f"Unknown model type: {config.model}")
