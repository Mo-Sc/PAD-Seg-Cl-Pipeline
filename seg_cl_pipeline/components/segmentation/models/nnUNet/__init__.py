import os


def set_nnunet_env_vars(seg_config):
    """
    Set environment variables required by nnUNet.
    """

    run_outpath = os.environ["OUTPUT_BASEPATH"]

    os.environ["nnUNet_raw"] = os.path.join(
        run_outpath, seg_config.step_name, "temp", "nnUNet_raw"
    )
    os.environ["nnUNet_preprocessed"] = os.path.join(
        run_outpath, seg_config.step_name, "temp", "nnUNet_preprocessed"
    )
    os.environ["nnUNet_results"] = os.path.join(
        run_outpath, seg_config.step_name, "temp", "nnUNet_results"
    )
