import os

work_dir = os.getcwd()


class MedSAMConfig:
    configuration = "vit_b"  # configuration of medsam model that was used for training

    model_weights = os.path.join(
        work_dir,
        "seg_cl_pipeline/components/segmentation/models/MedSAM/assets/checkpoint_best_weights_only.pth",
    )  # path to trained medsam model .pth file
    img_size = (
        1024  # images will be resized to this size (in px) as in original medsam model
    )

    mode = "instance"  # instance or batch. Whether the segmentation model receives a single image or a batch of images for inference


class nnUNetConfig:

    mode = "batch"  # instance or batch. Whether the segmentation model receives a single image or a batch of images for inference

    evaluation_mode = False  # In case of multiple folds including multiple trained models, use only the fold that wasnt trained on any image of the patient that is currently being processed. Default: False (use ensemble of all models)

    model_weights = os.path.join(
        work_dir,
        "seg_cl_pipeline/components/segmentation/models/nnUNet/assets/nnunet_5foldcv_trained.zip",
    )  # path to trained nnunet zip file (exported using nnUNetv2_export_model_to_zip command)

    img_size = 210  # images will be resized to this size (in px) as in nnunet training

    dataset_name = (
        "Dataset044-msot-ic-2-us-segmentation"  # name of the dataset used for training
    )
    configuration = "2d"  # configuration of the nnunet model that was used for training
    folds = (0, 1, 2, 3, 4)  # folds that were used for training
    checkpoint_selector = (
        "checkpoint_final.pth"  # which checkpoint to use for inference
    )
