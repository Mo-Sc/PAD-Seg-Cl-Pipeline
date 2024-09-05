import os
import zipfile
import logging

import nrrd
import numpy as np
import torch
from tqdm import tqdm
from skimage.transform import resize

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.file_path_utilities import get_output_folder

# from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

from ..segmentation import Segmentation


class nnUNet(Segmentation):
    """
    Interface to nnUNet segmentation model.
    """

    def initialize(self):

        # create folders for env vars
        self.nnUNet_raw = os.environ["nnUNet_raw"]
        self.nnUNet_preprocessed = os.environ["nnUNet_preprocessed"]
        self.nnUNet_results = os.environ["nnUNet_results"]
        os.makedirs(self.nnUNet_raw, exist_ok=True)
        os.makedirs(self.nnUNet_preprocessed, exist_ok=True)
        os.makedirs(self.nnUNet_results, exist_ok=True)

        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.device = torch.device("cuda")

        # load model from zip. Will be saved in nnUNet_results.
        with zipfile.ZipFile(self.weights_path, "r") as zip_ref:
            zip_ref.extractall(self.nnUNet_results)

        if self.config.model_config.evaluation_mode:
            # just for evaluation purposes. Load folds so that the fold that contains the image that is being processed can be excluded from inference
            self.patients_folds = self.load_folds(
                os.path.join(
                    self.nnUNet_results,
                    self.config.model_config.dataset_name,
                    "nnUNetTrainer__nnUNetPlans__2d",
                )
            )

        logging.info(f"Loaded nnUNet from {self.weights_path}")

    def preprocess(self, data):
        # load all the files in img_paths (npy files) and save them as nrrd files into the nnunet nnUNet_preprocessed dir
        # return the paths to the nrrd files

        assert isinstance(data, list), "Data must be a list of file paths."

        # if self.config.model_config.evaluation_mode:
        #     # flush nnUNet_preprocessed folder
        #     for file in os.listdir(self.nnUNet_preprocessed):
        #         os.remove(os.path.join(self.nnUNet_preprocessed, file))

        outfiles = []

        for img_path in (pbar := tqdm(data)):

            pbar.set_description(f"Preprocessing {os.path.basename(img_path)}")
            image_data = np.load(img_path)

            # save shape of original image for resizing back later
            if self.orig_img_shape is None:
                self.orig_img_shape = image_data.shape

            # resize to model input size if necessary
            if image_data.shape != (
                img_size := (
                    self.config.model_config.img_size,
                    self.config.model_config.img_size,
                )
            ):
                image_data = resize(
                    image_data,
                    img_size,
                    order=3,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=True,
                )

            # save to nnUNet_preprocessed folder
            outpath = os.path.join(
                self.nnUNet_preprocessed,
                os.path.basename(img_path).replace(".npy", "_0000.nrrd"),
            )
            nrrd.write(outpath, image_data)
            outfiles.append(outpath)

        logging.info(
            f"Preprocessing done. {len(outfiles)} images saved to {self.nnUNet_preprocessed}\n"
        )

        return outfiles

    def inference(self, data):
        # adapted from nnunetv2/inference/predict_from_raw_data.py
        # params that are not specified in the model_config are set to default values

        assert isinstance(data, list), "Data must be a list of file paths."

        if self.config.model_config.evaluation_mode:
            # just for evaluation purposes. Load folds so that the fold that contains the image that is being processed can be excluded from inference
            # find the fold that contain the image being processed in their validation set
            if int(self.study_id) in self.patients_folds:
                folds = tuple(self.patients_folds[int(self.study_id)])
            else:
                logging.warning(
                    f"Study {self.study_id} not found in folds. Using all folds."
                )
                folds = self.config.model_config.folds

        else:
            # default: use all folds specified in model_config
            folds = self.config.model_config.folds

        logging.info(f"Using folds {folds} for inference.")

        # output is saved to out folder in nnUNet_results
        outpath = os.path.join(
            self.nnUNet_results, self.config.model_config.dataset_name, "out"
        )

        os.makedirs(outpath, exist_ok=True)

        # get path to trained model
        model_folder = get_output_folder(
            dataset_name_or_id=self.config.model_config.dataset_name,
            configuration=self.config.model_config.configuration,
        )

        # initialize nnUNetPredictor
        predictor = nnUNetPredictor(device=self.device, allow_tqdm=True)

        predictor.initialize_from_trained_model_folder(
            model_folder,
            folds,
            checkpoint_name=self.config.model_config.checkpoint_selector,
        )
        # run inference
        data_list_of_lists = [[d] for d in data]
        predictor.predict_from_files(
            data_list_of_lists,
            outpath,
            num_processes_preprocessing=3,
            num_processes_segmentation_export=3,
        )

        # collect output files, make sure they exist

        outfiles = []

        for infile in data:
            # filename of outfile
            filename = os.path.basename(infile).replace("_0000.nrrd", ".nrrd")
            outfile = os.path.join(outpath, filename)
            assert os.path.exists(
                outfile
            ), f"Segmentation output file {outfile} corresponding to input file {infile} not found."
            outfiles.append(outfile)

        logging.info(f"Inference done. {len(outfiles)} masks saved to {outpath}\n")

        return outfiles

    def postprocess(self, data):
        # load all the masks, resize them back to pipeline size
        # save them as npy files

        assert isinstance(data, list), "Data must be a list of file paths."

        outfiles = []

        for mask_path in (pbar := tqdm(data)):
            pbar.set_description(f"Postprocessing {os.path.basename(mask_path)}")
            mask_data, _ = nrrd.read(mask_path)

            # resize back to original shape
            if self.orig_img_shape != mask_data.shape:
                mask_data = resize(
                    mask_data,
                    self.orig_img_shape,
                    order=0,
                    mode="constant",
                    preserve_range=True,
                    anti_aliasing=False,
                )
            outpath = os.path.join(
                self.outpath, os.path.basename(mask_path).replace(".nrrd", ".npy")
            )

            # save as npy to outpath
            np.save(outpath, mask_data)
            outfiles.append(outpath)

        logging.info(
            f"Postprocessing done. {len(outfiles)} masks saved to {self.outpath}\n"
        )

        return outfiles

    def __del__(self):
        if not os.environ["MODE"] == "DEBUG":
            # cleanup nnUNet temp folder
            os.rmdir(tf := os.path.join(self.outpath, "temp"))
            logging.info(f"Deleted nnunet temp folder: {tf}")

    @staticmethod
    def load_folds(nnune_results_path, num_folds=5):
        """
        Loads the summary.json file from each trained nnunet fold and creates a dictionary with all the study_ids as key and the folds in which they were included as value.
        """
        import json

        patients_folds = {}

        for fold in range(num_folds):
            summary_file = os.path.join(
                nnune_results_path, f"fold_{fold}", "validation", "summary.json"
            )
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    summary_dict = json.load(f)
                    cases = summary_dict["metric_per_case"]

                    for case in cases:
                        study_scan_str = (
                            os.path.basename(case["reference_file"])
                            .split("_")[-1]
                            .split(".")[0]
                        )
                        study_id, scan_id = int(study_scan_str[:3]), int(
                            study_scan_str[3:]
                        )
                        if study_id in patients_folds:
                            # append fold to list of folds
                            patients_folds[study_id].append(fold)
                        else:
                            # create new list with fold
                            patients_folds[study_id] = [fold]

            else:
                raise ValueError(f"Summary file {summary_file} not found.")

        return patients_folds
