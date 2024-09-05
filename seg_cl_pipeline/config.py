import os

work_dir = os.getcwd()

SUBJECT_ID_SOURCE = "JSON"  # can be "METAFILE_1" / "METAFILE_2" (to load scans and labels from clinic xlx files) or "JSON" (to define scans and labels directly in a json file)
SUBJECTS_FILE = os.path.join(work_dir, "data/demo/demo_subjects.json")
TARGET_SCANS = None  # list of scans, like [8, 9] or None for all available scans

# path to raw msot scans
US_NPY_DATASET_PATH = os.path.join(work_dir, "data/demo/us")
US_CHANNEL_NAMES = ["US"]  # list of channel names in the US scans

# path to raw msot scans
OA_NPY_DATASET_PATH = os.path.join(work_dir, "data/demo/oa")
OA_CHANNEL_NAMES = ["Hb", "HbO2"]  # list of channel names in the OA scans

OUTPUT_BASEPATH = os.path.join(work_dir, "data")

IMG_SIZE = 400  # desired image size of the US images (in px) (not size of input images, they will be resized)
PX_SIZE = 0.04 / IMG_SIZE  # pixel size in m, image is 4cm x 4cm

LOGGING = True  # if True, logs will be written to logfile
MODE = "DEBUG"  # if DEBUG, imgs will be written out as pngs, intermediate results will be saved

RUN_ID = None  # can be used to fix run_id, eg for debugging. Overwrites existing files. Is necessary if steps should be skipped


class PreProcessorConfig:

    step_name = "1_preprocessing"
    enabled = True

    src_type = "US"  # has to be in src_channel_names
    src_channel_names = US_CHANNEL_NAMES
    src_format = "npy"  # can be "npy", "nrrd", "nifti", "hdf5"
    src_path = US_NPY_DATASET_PATH

    flipud = False  # depending on file format, image might be upside down
    fliplr = False  # depending on file format, image might be mirrored
    rotate = 0  # rotate image (np.rot90) 1,2,3 times
    img_size = IMG_SIZE  # images will be resized to this size (in px)


class SegmentationConfig:

    step_name = "2_segmentation"
    enabled = True

    # model specific config file
    from seg_cl_pipeline.components.segmentation.models.model_config import nnUNetConfig

    # from seg_cl_pipeline.components.segmentation.models.model_config import MedSAMConfig

    model_config = nnUNetConfig
    model = "nnUNet"  # MedSAM or nnUNet


class PostProcessorConfig:

    step_name = "3_postprocessing"
    enabled = True


class ROIPlacementConfig:

    step_name = "4_roi_placement"
    enabled = True

    roi_type = "polygon"  # can be "ellipse" or "polygon"

    px_size = PX_SIZE  # in m

    # --- for ellipse ---
    roi_ellipse_size = [0.00960219, 0.00244162]  # as used by clinic (in m)
    margin = 0 * px_size  # How deep is ellipse in muscle (in m)

    # --- for polygon ---
    roi_height = 30 * px_size  # in m

    # specifies a region in which the roi can be placed. [y_min, y_max, x_min, x_max] (in m)
    sensitivity_map = [
        100 * px_size,
        300 * px_size,
        130 * px_size,
        270 * px_size,
    ]

    iannotation_export = False  # exports rois into ilabs format. Doesnt work for polygon rois (cant be viewed in ilabs anyways)
    json_export = False  # exports rois as json additionally to npy masks (doesnt work yet for polygon rois)

    # required for debugging (to create the US overlay images)
    preprocessed_us_folder = PreProcessorConfig.step_name


class AnalysisConfig:
    step_name = "5_analysis"
    enabled = True

    # OA Dataloading Stuff
    src_type = "mSO2"  # Has to be in src_channel_names or "mSO2".
    src_channel_names = OA_CHANNEL_NAMES
    src_path = OA_NPY_DATASET_PATH
    src_format = "npy"  # for OA images, can be "npy", "nrrd", "nifti", "hdf5"

    # ROI Dataloading Stuff
    roi_src_format = "npy"  # npy or json
    img_size = IMG_SIZE  # needed for creating mask from json ROI

    # Intensity Extraction Stuff
    metric = "mean"  # "mean", "median", "sum", "max"
    clip_negative = (True,)  # clip negative intensities to 0
    exclude_zeros = True  # exclude zero values from calculation

    # load task specific config
    from seg_cl_pipeline.components.analysis.analysis_config import (
        ClassificationConfigDatasetA,
        ClassificationConfigDatasetB,
        ClassificationConfigDemo,
    )

    task = "classification"
    task_config = ClassificationConfigDemo
