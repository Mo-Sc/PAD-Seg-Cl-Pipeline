# OA-US Analysis Pipeline Documentation

This is an extended and updated version of the pipeline used in the previous publication:

> **Automatic Muscle Segmentation for the Diagnosis of Peripheral Artery Disease Using Multispectral Optoacoustic Tomography**  
> [https://doi.org/10.1117/12.3049067](https://doi.org/10.1117/12.3049067)

Samples from the old version can be found [here](SPIE_Pub_Samples.md).

**Warning: this repository is under constant development and modification. Therefore it is recommended to contact the author before using it.**

Contact: moritz.schillinger@fau.de

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Core Concepts](#core-concepts)
   - [Subject](#subject)
   - [HDF5 Data Storage](#hdf5-data-storage)
   - [Configuration System](#configuration-system)
5. [Pipeline Components](#pipeline-components)
   - [Preprocessing](#preprocessing)
   - [Segmentation](#segmentation)
   - [Postprocessing](#postprocessing)
   - [ROI Placement](#roi-placement)
   - [Extraction](#extraction)
   - [Analysis](#analysis)
6. [Additional Modules](#additional-modules)
   - [iAnnotation Loader](#iannotation-loader)
   - [Dataset Loader](#dataset-loader)
   - [Summary Plot](#summary-plot)
7. [Configuration Reference](#configuration-reference)
8. [Creating a Pipeline](#creating-a-pipeline)
9. [Running the Pipeline](#running-the-pipeline)
10. [Sample Summary Plot](#sample-summary-plot)

---

## Overview

Configurable pipeline for automatic ultrasound muscle segmentation and MSOT (Multispectral Optoacoustic Tomography) ROI analysis. The pipeline:

- Loads OA + US data from various formats (HDF5, NIfTI, NPY)
- Preprocesses OA and US frames (resizing, normalization, flipping)
- Segments muscle tissue using pluggable segmentation backends (default: custom U-Net)
- Postprocesses segmentation masks (connected components, cleanup)
- Places ROIs of various shapes (ellipse, polygon, static) and maps them to OA images
- Extracts ROI intensities using radiomics features
- Performs configurable downstream analysis (group comparison, depth profiles, image quality metrics)

---

## Installation

### Requirements (might be outdated)

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.3.1+ (for segmentation models)
- NumPy, Pandas (data handling)
- scikit-image, OpenCV (image processing)
- pyradiomics (feature extraction)
- h5py (HDF5 file handling)
- Matplotlib, Plotly (visualization)

---

## Project Structure

```
oa_us_pipeline/
├── v2/
│   ├── __init__.py          # StepNames enum, utilities
│   ├── configs.py           # All configuration dataclasses
│   ├── pipeline.py          # Demo pipeline definition
│   ├── runner.py            # Pipeline execution entry point
│   │
│   ├── components/          # Pipeline components
│   │   ├── pipeline_component.py      # Base class
│   │   ├── preprocessing/
│   │   ├── segmentation/
│   │   ├── postprocessing/
│   │   ├── roi_placement/
│   │   ├── extraction/
│   │   └── analysis/
│   │
│   ├── data/                # Data handling
│   │   ├── subject.py       # Subject class
│   │   ├── tags.py          # HDF5 group/dataset tags
│   │   ├── constants.py     # Dataset constants (wavelengths)
│   │   └── load_subjects.py # Custom loading functions
│   │
│   ├── pipelines/    # Pre-defined pipeline configurations
│   │
│   └── utils/               # Utility modules
│       ├── channels.py      # mSO2, ratio calculations
│       ├── dataset_loader.py
│       ├── iannotation_loader.py
│       ├── ithera.py        # iThera format conversion
│       ├── summary_plot.py
│       ├── statistics.py
│       └── visualizations.py
│
├── misc/                    # Demo data and figures
├── requirements.txt
├── setup.py
└── README.md
```

---

## Core Concepts

### Subject

The `Subject` class (`v2/data/subject.py`) represents a single sample in the pipeline. Each subject links to an HDF5 file storing all pipeline outputs.

**Key attributes:**
- `input_file`: Source file path
- `group_id`: Group/class label (e.g., 0=PAD, 1=Healthy)
- `study_id`: Study identifier
- `scan_id`: Scan number within study
- `frame`: Target frame index or name
- `subject_id`: Generated unique ID: `{group_id:03d}-{study_id:03d}-{scan_id:03d}-{frame:03d}`


### HDF5 Data Storage

All intermediate and final outputs are stored in per-subject HDF5 files. Structure:

```
{subject_id}.hdf5
├── RAW_DATA/
│   ├── US                   # (F, H, W) ultrasound
│   └── OA                   # (F, WAW, H, W) optoacoustic
├── PREPROCESSING/
│   └── US                   # Preprocessed US
├── SEGMENTATION/
│   └── MASK                 # Raw segmentation mask
├── POSTPROCESSING/
│   └── MASK                 # Cleaned segmentation mask
├── ROI_PLACEMENT/
│   └── MASK                 # ROI mask
└── EXTRACTION/
    ├── OA                   # Masked OA data
    └── ROI_FEATURES         # Extracted features (tabular)
```

**HDF5 Tags** (`v2/data/tags.py`):
- Group names: `RAW_DATA`, `PREPROCESSING`, `SEGMENTATION`, `POSTPROCESSING`, `ROI_PLACEMENT`, `EXTRACTION`
- Dataset names: `OA`, `US`, `MASK`, `ROI_FEATURES`

### Configuration System

All configurations inherit from `Config` base class (`v2/configs.py`), providing automatic serialization for JSON export.

---

## Pipeline Components

All components inherit from `PipelineComponent` (`v2/components/pipeline_component.py`).

**Base class interface:**
```python
class PipelineComponent:
    def __init__(self, name, config, subjects, source_overrides=None, target_overrides=None)
    def run()                    # Execute component
    def _run_component()         # Implement in subclass
    def _default_sources()       # Define input (group, dataset) pairs
    def _default_targets()       # Define output (group, dataset) pairs
```

### Preprocessing

**File:** `v2/components/preprocessing/preprocessing_component.py`

**Purpose:** Prepares input images for segmentation.

**Operations:**
- Flip vertically/horizontally (`flipud`, `fliplr`)
- Rotate by 90° multiples (`rotate90`)
- Resize to target size (`img_size`)

**Config (`PreprocessingConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `flipud` | bool | False | Flip image upside down |
| `fliplr` | bool | False | Flip image left-right |
| `rotate90` | int | 0 | Rotate 90° × n times |
| `img_size` | int | 400 | Target image size (square) |

**I/O:**
- Source: `RAW_DATA/US`
- Target: `PREPROCESSING/US`

### Segmentation

**File:** `v2/components/segmentation/segmentation_component.py`

**Purpose:** Segments anatomical structures in ultrasound images.

**Supported architectures:**
- `cUNet`: Custom U-Net implementation ([sendero-us-segmentation](https://github.com/Mo-Sc/sendero-us-segmentation))
- `nnUNet`: Not implemented in v2
- `MedSAM`: Not implemented in v2

**Segmentation labels** (default, 10 classes):
| ID | Label | Description |
|----|-------|-------------|
| 0 | background | Background |
| 1 | Haut | Skin |
| 2 | Faszie1 | Fascia 1 |
| 3 | Muskel1 | Muscle 1 |
| 4 | Muskel2 | Muscle 2 |
| 5 | Membran | Membrane |
| 6 | Faszie2 | Fascia 2 |
| 7 | Vorlaufstrecke | Standoff |
| 8 | SAT | Subcutaneous adipose tissue |
| 9 | Gel | Coupling gel |

**Config (`SegmentationConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `arch` | str | "cUNet" | Architecture name |
| `model_config` | Config | cUNetConfig() | Model-specific config |

**cUNet Config (`cUNetConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | "batch" | Processing mode |
| `model_weights` | str | Path | Path to trained weights |
| `img_size` | int | 224 | Input image size |
| `ds_mean` | float | Dataset | Dataset mean for normalization |
| `ds_std` | float | Dataset | Dataset std for normalization |
| `n_classes` | int | 10 | Number of segmentation classes |
| `png_export` | bool | True | Export segmentation as PNG |

**I/O:**
- Source: `PREPROCESSING/US`
- Target: `SEGMENTATION/MASK`

### Postprocessing

**File:** `v2/components/postprocessing/postprocessing_component.py`

**Purpose:** Cleans and refines segmentation masks.

**Operations (in order):**
1. Keep largest connected component per specified class
2. Reassign freed pixels based on row majority
3. Combine class groups (merge related classes)
4. Remove small objects per class
5. Reassign remaining freed pixels using neighbor majority

**Config (`PostprocessingConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keep_largest_per_class` | List[int] | [1,3,7,8] | Classes to keep largest component |
| `combine_class_groups` | List[List[int]] | [[2,6],[3,4]] | Classes to merge (first = target) |
| `remove_small_objects_config` | List[Tuple[int,int]] | [...] | (class_id, min_size) pairs |

**I/O:**
- Source: `SEGMENTATION/MASK`
- Target: `POSTPROCESSING/MASK`

### ROI Placement

**File:** `v2/components/roi_placement/roi_placement_component.py`

**Purpose:** Places region of interest within segmented tissue.

**ROI Types:**
1. **Ellipse**: Fixed-size ellipse placed at top of target class along center axis
2. **Polygon**: Trimmed version of segmentation mask with specified height/width
3. **Static**: Pre-defined mask loaded from file

**Config (`ROIPlacementConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_class_id` | int | 3 | Class in which to place ROI |
| `roi_type` | str | "ellipse" | ROI shape type |
| `roi_params` | dict | {...} | Shape-specific parameters |
| `ilabs_export` | bool | False | Export iThera iAnnotation format |

**Ellipse parameters:**
- `roi_ellipse_size`: [width, height] in meters
- `margin`: Depth offset from top edge in meters

**Polygon parameters:**
- `roi_height`: Height in meters
- `roi_width`: Width in meters

**Static parameters:**
- `roi_mask`: Path to .npy file or numpy array
- `in_target_class`: Only place within target class if True

**I/O:**
- Source: `POSTPROCESSING/MASK`
- Target: `ROI_PLACEMENT/MASK`

### Extraction

**File:** `v2/components/extraction/extraction_component.py`

**Purpose:** Extracts quantitative features from OA images within ROI.

**Features:** Uses pyradiomics for feature extraction. Default: firstorder statistics (Mean, Median, etc.).

**Derived channels:**
- `mSO2`: Calculated as HbO2 / (Hb + HbO2)
- Ratio channels: e.g., `"700/850"` for wavelength ratios

**Config (`ExtractionConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_class_id` | int | 2 | ROI class ID for extraction |
| `feature_classes` | List[str] | ["firstorder"] | Radiomics feature classes |
| `derived_channels` | List[str] | [] | Channels to compute (mSO2, ratios) |
| `positive_only` | bool | False | Exclude zero/negative values |
| `xlsx_export` | bool | True | Export combined Excel file |

**I/O:**
- Sources: `ROI_PLACEMENT/MASK`, `RAW_DATA/OA`
- Targets: `EXTRACTION/ROI_FEATURES`, `EXTRACTION/OA`

### Analysis

**File:** `v2/components/analysis/analysis_component.py`

**Purpose:** Performs downstream analysis on extracted features.

**Analysis Strategies:**

#### 1. Group Comparison (`group_comparison`)

Compares extracted features between groups (e.g., PAD vs. Healthy).

**Outputs:**
- Boxplots comparing groups
- Scatter plots with trend lines
- ROC curves (if 2 groups)
- Classification metrics (accuracy, F1, AUC)

**Config (`GroupComparisonConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `groups` | List[int] | [1,2,...,6] | Group IDs to compare |
| `group_labels` | List[str] | [...] | Display labels |
| `target_features` | List[str] | [Mean, Median] | Features to analyze |
| `target_channels` | List[str] | [Hb, HbO2, mSO2] | Channels to analyze |
| `combine_studies` | bool | True | Average per study ID |
| `stat_test` | str | "ttest_student" | Statistical test |
| `classification_metrics` | bool | False | Compute classification metrics |
| `invert` | List[bool] | [...] | Invert prediction per channel |

#### 2. Depth Profile (`depth_profile`)

Plots feature values against ROI depth.

**Config (`DepthProfileConfig`):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `target_channels` | List[str] | Channels to plot |
| `plot_trendline` | bool | Show trend line |
| `profile_labels` | List[str] | Label schemes (GT, CM, unlabelled) |

#### 3. Image Quality Metrics (`image_quality_metrics`)

Computes reconstruction quality metrics (MSE, PSNR, SSIM) between ground truth and reconstructed images.

**Config (`ImageQualityMetricsConfig`):**
| Parameter | Type | Description |
|-----------|------|-------------|
| `recon_group/dataset` | str | Reconstruction data location |
| `gt_group/dataset` | str | Ground truth data location |
| `metrics` | List[str] | Metrics to compute (MSE, PSNR, SSIM, DIFF) |

---

## Additional Modules

### iAnnotation Loader

**File:** `v2/utils/iannotation_loader.py`

Loads ROI masks from iThera iAnnotation files (HDF5 format).

**Config (`IAnnotationLoaderConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | Path | "data/input" | Directory containing annotations |
| `anno_index` | int | -1 | Annotation index (last by default) |
| `roi_index` | int | -1 | ROI index within annotation |
| `img_size` | int | 400 | Output mask size |
| `px_size` | float | 0.0001 | Pixel size in meters |

### Dataset Loader

**File:** `v2/utils/dataset_loader.py`

Loads arbitrary datasets into the pipeline HDF5 structure.

**Config (`DatasetLoaderConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | Path | "data/input" | Base directory |
| `target_group` | str | "RAW" | HDF5 group name |
| `target_dataset` | str | "DATASET" | HDF5 dataset name |
| `channel_names` | List[str] | [...] | Channel labels |
| `naming_scheme` | str | "{study_id}_{scan_id}.nrrd" | File naming pattern |
| `file_format` | str | "nrrd" | File format (nrrd, hdf5, npy) |

### Summary Plot

**File:** `v2/utils/summary_plot.py`

Creates per-subject PNG visualization of all pipeline steps.

**Config (`SummaryPlotConfig`):**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plotable_groups` | List[str] | [...] | HDF5 groups to plot |
| `oa_channel_name` | str | "Hb" | OA channel to display |
| `plot_size` | int | 400 | Plot size in pixels |

---

## Configuration Reference

### GlobalConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | Path | "data/output" | Base output directory |
| `logging_level` | str | "INFO" | Logging verbosity |
| `overwrite` | bool | True | Overwrite existing outputs |
| `random_seed` | int | 42 | Random seed |
| `run_id` | str | None | Custom run identifier (auto-generated timestamp if None) |

### DataConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dir` | Path | "data/input" | Input data directory |
| `metadata_file` | Path | "data/metadata.csv" | Metadata CSV path |
| `target_scans` | List[int] | None | Scan indices to process |
| `target_frames` | List/str | None | Frame indices/names or "annotated" |
| `hdf5_tags_us` | List[str] | [...] | HDF5 path to US data |
| `hdf5_tags_oa` | List[str] | [] | HDF5 path to OA data |
| `target_channels_us` | List | [12] | US channel indices |
| `target_channels_oa` | List | None | OA channel indices |
| `px_size` | float | 0.0001 | Pixel size in meters |

---

## Creating a Pipeline

Define a pipeline as a list of components in a Python file ([pipeline.py](v2/pipelines/pipeline.py)):

```python
from v2 import StepNames
from v2.components.preprocessing.preprocessing_component import PreprocessingComponent
from v2.components.segmentation.segmentation_component import SegmentationComponent
from v2.components.postprocessing.postprocessing_component import PostprocessingComponent
from v2.components.roi_placement.roi_placement_component import ROIPlacementComponent
from v2.components.extraction.extraction_component import ExtractionComponent
from v2.components.analysis.analysis_component import AnalysisComponent
from v2.utils.summary_plot import SummaryPlot
from v2.configs import *
from v2.components.segmentation.model_configs import cUNetConfig
from v2.components.analysis.strategy_configs import GroupComparisonConfig

# Load subjects (implement custom loader or use existing)
from v2.data.load_subjects import collect_subjects_demo
subjects = collect_subjects_demo(data_config, file_ending=".hdf5")

# Define configs
global_config = GlobalConfig(output_dir="data/output", overwrite=True)
data_config = DataConfig(input_dir="data/input", px_size=0.0001)
segmentation_config = SegmentationConfig(arch="cUNet", model_config=cUNetConfig())
analysis_config = AnalysisConfig(strategy_name="group_comparison", strategy_config=GroupComparisonConfig())

# Build pipeline
pipeline = [
    PreprocessingComponent(name=StepNames.PREPROCESSING, config=PreprocessingConfig(), subjects=subjects),
    SegmentationComponent(name=StepNames.SEGMENTATION, config=segmentation_config, subjects=subjects),
    PostprocessingComponent(name=StepNames.POSTPROCESSING, config=PostprocessingConfig(), subjects=subjects),
    ROIPlacementComponent(name=StepNames.ROI_PLACEMENT, config=ROIPlacementConfig(), subjects=subjects),
    ExtractionComponent(name=StepNames.EXTRACTION, config=ExtractionConfig(), subjects=subjects),
    AnalysisComponent(name=StepNames.ANALYSIS, config=analysis_config, subjects=subjects),
    SummaryPlot(name=StepNames.SUMMARY_PLOT, config=SummaryPlotConfig(), subjects=subjects),
]
```

### Source/Target Overrides

Components support custom source/target mappings:

```python
ROIPlacementComponent(
    name=StepNames.ROI_PLACEMENT,
    config=ROIPlacementConfig(),
    subjects=subjects,
    source_overrides={"primary": (HDF5Tags.CUSTOM_GROUP, HDF5Tags.SEG)},
    target_overrides={"primary": (HDF5Tags.ROI, HDF5Tags.SEG)},
)
```

---

## Running the Pipeline

### Via runner.py

1. Import your pipeline config in `runner.py`:
```python
from v2.pipeline import (
    pipeline, global_config, data_config
)
```

2. Execute:
```bash
python v2/runner.py
```

### Output Structure

```
{output_dir}/{run_id}/
├── pipeline.json          # Full configuration dump
├── pipeline.log           # Execution log
└── data/
    ├── {subject_id}.hdf5  # Per-subject data
    ├── .preprocessing/    # Component outputs
    ├── .segmentation/
    ├── .postprocessing/
    ├── .roi_placement/
    ├── .extraction/
    ├── .analysis/
    └── .summary_plot/
```

---

## Sample Summary Plot (Demo Image)

<img src="misc/figures/summary_plot_demo_v2.png" alt="Summary Plot - Demo" width="80%"/>
