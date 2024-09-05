import os

import pandas as pd
import numpy as np


def check_pipeline_config(config):
    """
    Check if the pipeline configuration is valid:
    If RUN_ID is not set to constant, all the steps have to be enabled (Because otherwise intermediate files are missing).
    If analysis is enabled, labels have to be provided.
    """
    if config.RUN_ID is None:
        enabled_flags = [
            config.PreProcessorConfig.enabled,
            config.SegmentationConfig.enabled,
            config.PostProcessorConfig.enabled,
            config.ROIPlacementConfig.enabled,
        ]
        if not all(enabled_flags):
            raise ValueError(
                "Some intermediate pipeline steps are disabled, but RUN_ID is not constant (Intermediate data will be missing)."
            )

    # Check if labels are provided if analysis is enabled (Do all analysis tasks require labels?)
    if config.AnalysisConfig.enabled:

        subjects, labels = load_subjects(config.SUBJECT_ID_SOURCE, config.SUBJECTS_FILE)

        assert labels is not None, "Labels are missing, but analysis is enabled"
        assert len(subjects) == len(
            labels
        ), "Number of subjects and labels do not match"


def load_subjects(subjects_source, filepath):
    """
    Load the subject id and their labels if available from the given source.
    source can be "JSON" or "METAFILE_1" or "METAFILE_2"
    JSON: subjects and labels are defined in a json file
    METAFILE_1: subjects and labels are loaded from the clinical study documentation xlsx file (Dataset A)
    METAFILE_2: subjects and labels are loaded from the clinical study documentation xlsx file (Dataset B)
    """

    if subjects_source == "JSON":
        import json

        with open(filepath) as f:
            data = json.load(f)

        subjects = np.array(data["subjects"])
        if "labels" in data:
            labels = np.array(data["labels"])
        else:
            labels = None

    elif subjects_source == "METAFILE_2":
        hvs, ics, _, _ = load_meta_data_dsa(filepath)
        subjects = np.array(hvs + ics)
        labels = np.array([0] * len(hvs) + [1] * len(ics))

    elif subjects_source == "METAFILE_1":
        raise NotImplementedError("METAFILE_1 not implemented yet")
    else:
        raise ValueError("Invalid subjects source")

    return subjects, labels


def load_meta_data_dsa(metafile):
    """
    Load the meta data from the clinical study documentation xlsx file of Dataset A.
    """

    # load the meta data
    meta_df = pd.read_excel(metafile)

    hvs = meta_df.loc[meta_df["gesund"] == 1, ["Study_Number", "NOTIZ"]]
    ics = meta_df.loc[meta_df["gesund"] == 0, ["Study_Number", "NOTIZ"]]

    hvs_study_numbers = hvs["Study_Number"].values
    ics_study_numbers = ics["Study_Number"].values
    hvs_notes = hvs["NOTIZ"].values
    ics_notes = ics["NOTIZ"].values

    hvs_int = [int(num) for num in hvs_study_numbers]
    ics_ints = [int(num) for num in ics_study_numbers]

    return hvs_int, ics_ints, hvs_notes, ics_notes


def get_filenames(subjects, basepath, file_format, target_scans=None):
    """
    Get all the filepaths of the existing files in the basepath for the given subjects.
    Optionally filter for specific target scans.
    Otherwise all files for the subject are returned.
    """

    filepaths = []
    not_found = []

    for subject in subjects:

        if target_scans is not None:

            # get all files in the basepath that start with the subject id and end with the file format
            subject_files = [
                os.path.join(basepath, file)
                for file in os.listdir(basepath)
                if int(file.split("_")[0]) == subject
                and int(file.split("_")[1].split(".")[0]) in target_scans
                and file.endswith(file_format)
            ]
        else:
            # get all files in the basepath that start with the subject id and end with the file format
            subject_files = [
                os.path.join(basepath, file)
                for file in os.listdir(basepath)
                if int(file.split("_")[0]) == subject and file.endswith(file_format)
            ]

        if len(subject_files) == 0:
            not_found.append(subject)

        filepaths.extend(subject_files)

    return filepaths, not_found
