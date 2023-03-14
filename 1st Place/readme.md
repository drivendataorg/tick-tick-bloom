# Solution Documentation

## 1. Overview
This documentation intend to guide reader to reproduce the submited solution.

### 1.1 folder and files

* inputs: official `metadata.csv`, `submission_format.csv` and `train_labels.csv`
* codes: Jupyter notebook `nrrr.ipynb` and `Sentinel.ipynb` for extra data downloading.
* downloaded_data: Predownloaded data for NRRR and Sentinel.
* outputs: folder for generated files
* notebooks: `Model.ipynb` for final modeling, `Sentinels_features.ipynb` for Sentinels feature extraction.
* `reproduce.sh`: shell script to reproduce submission.csv
* `environment_*.yaml`: conda environment config file.

## 2. Environment Setup
### 2.1 Hardware
* CPU: AMD Ryzen R9 5950x
* RAM: 128 GB
* OS: ubuntu server 20.04

I use this setup to train the model and download the extra dataset, Any system with more than 4 Cores and 16GM RAM should be
okay for training, but higher machine is still recommanded.

### 2.2 Software
To setup the environment, please install conda first, then create env via the config yaml.
``` shell
conda env create --file environment_base.yaml --name tickrep
conda activate tickrep
```

> _DrivenData note: If you do not have libGL installed on your system, you may need to install it for opencv. You can do so with the following commands:_
>
> ```bash
> # CentOS/Amazon Linux
> yum install mesa-libGL
>
> # Ubuntu/Debian
> apt-get install libgl1
> ```

if you want to re-download SENTINEL-2 data, please create environment for downloading

``` shell
conda env create --file environment_download.yaml --name tickdown
conda activate tickdown
```


## 3. Reproduce

> _DrivenData note: predownloaded data is not included in this repository. You will need to have the competition data files in `inputs/` and follow the instructions in Section 4 to download the Sentinel and HRRR datasets.

To get the submission.csv with pre-downloaded data, run reproduce.sh (mask sure tickenv is activated)

```
/bin/bash reproduce.sh
```

This shell script includes two step by executing two notebooks:

1. `notebooks/Sentinels_features.ipynb`: extract water color features from Sentinels (if available)
2. `notebooks/Model.ipynb`: Train a lightGBM and KNN model and predict test set

## 4. Extra dataset download

I use two additional dataset, `NOAA High-Resolution Rapid Refresh (HRRR) Data` and `SENTINEL-2 Data`. Since the online data query and retrival maybe unstable I provide the downloaded data on the `downloaded_data` assets, two scripts are ready for the redownloading.

### 4.1 NRRR data

to download NRRR dataset, please use `codes/nrrr.ipynb`.

### 4.2 SENTINEL-2

This dataset is relative big and require a good Internet connection, Please rerun `codes/Sentinel.ipynb` to download datasets.

## 5. inference with current weight
* to make prediction, please use the `single_infer.ipynb` in codes
