# Predicting Algae Bloom's

This is the repo associated with user `apwheele` in the [DrivenData Algae Bloom prediction competition](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/649/).

## Python Environment

To set up the python environment, I use Anaconda. In particular here is my initial set up (I have a difficult time with geopandas, so make them go first):

    git clone https://github.com/apwheele/algaebloom.git
    cd ./algaebloom
    conda create --name bloom python=3.9 pip geopandas
    conda activate bloom
    pip install -r requirements.txt
    ipython kernel install --name "bloom_jpy" --user

Where `requirements.txt` has the necessary libraries for data download/manipulation and statistical modeling.

I saved the final built versions via `pip list > final_env.txt`, which is also uploaded to the repo.

## Quickstart

Once downloading this repository, to simply replicate the final models, you can do:

    # everything should be run from root directory
    cd ./algaebloom 
    python main_prepdata.py

This prepares the local sqlite database with the data needed to run the models. Then if you run:

    python main_preds.py

To estimate the model, generate predictions, and cache the final model object.

If you want to replicate downloading the original data from the planetary computer, see below. 

## Downloading Data

I have saved the final files I used in the competition as CSV files in the `./data` folder. These include:

 - `elevation_dem.csv`, data obtained from Planetary computers DEM source, see `get_dem.py`
 - `spat_lag.csv`, spatial lags in space/time (only from input data), see `get_lag.py`
 - `sat.csv`, feature engineering of satellite imagery, see `get_sat.py`
 - `split_pred.csv`, weights used for train/test splits, see `get_split.py` (needs to be run after `get_dem.py`)

In addition to the competition data csv files provided, `train_labels.csv`, `metadata.csv`, and `submission_format.csv` are all expected to be saved in the data folder.

If you have already run `python main_prepdata.py`, these will not work, as the tables are already populated. Below downloading the original data only works for if you start fresh from an empty database.

Note that for each of these scripts, they involved downloading (and caching) the data in a local sqlite database. On my personal machine they could take a very long time, and if your internet goes out could result in errors. The scripts are written so you can just "rerun" them again, and it will attempt to fill in the missing information and add in more data. E.g., if you are in the root of the project, you can run:

    python get_dem.py

See the output, and then if some data is missing, rerun the exact same script:

    python get_dem.py

To attempt to download more data. In the end I signed up for the Planetary Computer Hub, running the scripts on their machines went a bit faster than on my local machine.

## Running Models

To run the final model, you can do:

    python main_preds.py

This will save a model in the `./models` folder with the current date. You should get a print out showing how it is the same as the final winning solution (which I have saved in the repo), submission `./submissions/sub_2023_02_07.csv`.

## Modeling Notes

In addition to this, I have in the root folder `main_hypertune.py`, hypertuning experiments. And these results are saved in `hypertune_results.txt` (e.g. by running `python main_hypertune.py > hypertune_results.txt`. These helped guided the final models that I experiemented with, but the final ones are due to more idiosyncratic experimentation uploading every day.

To go over the modeling strategy, see the notebook `model_strategy.ipynb`. 