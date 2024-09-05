import os
import logging
import shutil
from datetime import datetime

from seg_cl_pipeline.utils import (
    load_subjects,
    check_pipeline_config,
    get_filenames,
)
from seg_cl_pipeline.components.preprocessing.preprocessor import Preprocessor
from seg_cl_pipeline.components.segmentation.segmentation_factory import (
    SegmentationFactory,
)
from seg_cl_pipeline.components.postprocessing.postprocessor import PostProcessor
from seg_cl_pipeline.components.roi_placement.roi_placement import ROIPlacement
from seg_cl_pipeline.components.analysis.analysis import Analysis

import seg_cl_pipeline.config as pl_config


def setup():

    check_pipeline_config(pl_config)

    # run_id is a timestamp, also used to create output folder.
    run_id = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if pl_config.RUN_ID is None
        else pl_config.RUN_ID
    )

    run_outpath = os.path.join(pl_config.OUTPUT_BASEPATH, run_id)
    os.makedirs(run_outpath, exist_ok=True)

    if pl_config.LOGGING:
        # if logging is enabled, write logs to file
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(run_outpath, "pipeline.log")),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    # set some env vars that are used all over the project
    os.environ["MODE"] = pl_config.MODE
    os.environ["RUN_ID"] = run_id
    os.environ["OUTPUT_BASEPATH"] = run_outpath

    return run_id, run_outpath


def main():

    run_id, run_outpath = setup()

    logging.info(f"##### STARTING SEG-CL-PIPELINE #####")
    logging.info(f"RUN ID: {run_id}")
    logging.info(f"LOGGING: {pl_config.LOGGING}")
    logging.info(f"MODE: {pl_config.MODE}")
    logging.info(f"OUTPUT BASEPATH: {pl_config.OUTPUT_BASEPATH}\n")

    # Copy config file to output directory
    shutil.copy(os.path.join(os.getcwd(), "seg_cl_pipeline/config.py"), run_outpath)
    logging.info(f"Config file copied to {run_outpath}")

    # Load subjects and labels from file
    subjects, labels = load_subjects(
        pl_config.SUBJECT_ID_SOURCE, pl_config.SUBJECTS_FILE
    )

    logging.info(
        f"Loaded {len(subjects)} subject_ids and {len(labels) if labels is not None else 0} labels from {pl_config.SUBJECT_ID_SOURCE} (Source: {pl_config.SUBJECTS_FILE})"
    )

    # load corresponding raw us filenames
    raw_files, not_found_subjects = get_filenames(
        subjects,
        pl_config.US_NPY_DATASET_PATH,
        pl_config.PreProcessorConfig.src_format,
        target_scans=pl_config.TARGET_SCANS,
    )

    logging.info(
        f"Loaded {len(raw_files)} raw US files from {pl_config.US_NPY_DATASET_PATH}. Couldnt find files for subject_ids {str(not_found_subjects)}\n\n"
    )

    # --- Preprocessing ---
    preprocessor = Preprocessor(pl_config.PreProcessorConfig)
    if pl_config.PreProcessorConfig.enabled:
        preprocessed_files = preprocessor.process(raw_files)
    else:
        preprocessed_files = preprocessor.skip()

    # --- Segmentation ---
    seg_model = SegmentationFactory.create_model(pl_config.SegmentationConfig)
    if pl_config.SegmentationConfig.enabled:
        seg_model.initialize()
        segmented_files = seg_model.segment(preprocessed_files)
    else:
        segmented_files = seg_model.skip()

    # --- Postprocessing ---
    postprocessor = PostProcessor(pl_config.PostProcessorConfig)
    if pl_config.PostProcessorConfig.enabled:
        postprocessed_files = postprocessor.process(segmented_files)
    else:
        postprocessed_files = postprocessor.skip()

    # --- ROI Placement ---
    roiplacer = ROIPlacement(pl_config.ROIPlacementConfig)
    if pl_config.ROIPlacementConfig.enabled:
        roi_files = roiplacer.place_roi(postprocessed_files)
    else:
        roi_files = roiplacer.skip()

    # --- Analysis ---
    analyzer = Analysis(pl_config.AnalysisConfig)
    if pl_config.AnalysisConfig.enabled:
        analyzer.extract_intensities(roi_files)
        analyzer.analyze(subjects, labels)
    else:
        analyzer.skip()

    logging.info("Pipeline finished\n")


if __name__ == "__main__":
    main()
