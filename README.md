[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[![2017 algal bloom in Lake Erie seen as bright green splotches, captured by the NASA/USGS Landsat mission](https://drivendata-public-assets.s3.amazonaws.com/competition_cyano_banner.jpeg)](https://www.drivendata.org/competitions/143/tick-tick-bloom/)

# Tick Tick Bloom: Harmful Algal Bloom Detection Challenge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8329929.svg)](https://doi.org/10.5281/zenodo.8329929)
[![Tick Tick Bloom Challenge](https://img.shields.io/badge/DrivenData-Tick%20Tick%20Bloom-white?logo=data:image/png;base64,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)](https://www.drivendata.org/competitions/143/tick-tick-bloom/)

## Goal of the Competition

Inland water bodies provide a variety of critical services for both human and aquatic life, including drinking water, recreational and economic opportunities, and marine habitats. A significant challenge water quality managers face is the formation of **harmful algal blooms (HABs). One of the major types of HABs is cyanobacteria.** HABs produce toxins that are poisonous to humans and their pets, and threaten marine ecosystems by blocking sunlight and oxygen. Manual water sampling, or “in situ” sampling, is generally used to monitor cyanobacteria in inland water bodies. In situ sampling is accurate, but time intensive and difficult to perform continuously.

**The goal in this challenge was to use satellite imagery to detect and classify the severity of cyanobacteria blooms in small, inland water bodies.** The resulting algorithms will help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs. Ultimately, more accurate and more timely detection of algal blooms helps keep both the human and marine life that rely on these water bodies safe and healthy.

## What's in this Repository

This repository contains code from winning competitors in the [Tick Tick Bloom: Harmful Algal Bloom Detection](https://www.drivendata.org/competitions/143/tick-tick-bloom/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | [sheep](https://www.drivendata.org/users/sheep/) | 0.7554 | 0.7608 | For the Midwest and Northeast, trains a gradient boosted decision tree on temperature and satellite imagery water color. For the West and South, where water bodies tended to be smaller and data lower quality, trains a KNN model on location.
2   | [apwheele](https://www.drivendata.org/users/apwheele/) | 0.7476 | 0.7616 | Trains three regression models (LGBM, XGBoost, and catboost) on date, location, elevation data, and satellite imagery. Uses K-means segmentation to identify the lake area in images.
3   | [karelds](https://www.drivendata.org/users/karelds/) | 0.7698 | 0.7844 | Trains a separate gradient boosted regression tree for each combination of satellite dataset (Landsat 8, Landsat 9, Sentinel) and region, and ensembles to get a final prediction. Features are temperature, humidity, and satellite imagery color statistics.

### Method Write-up Bonus

Place | Team or User | Place in Prediction Competition | Link
--- | --- | ---   | ---
1 | [karelds](https://www.drivendata.org/users/karelds/) | 3rd | [Read the report](https://github.com/drivendataorg/tick-tick-bloom/blob/main/3rd%20Place/reports/3rd-Place_DrivenData-Competition-Winner-Documentation.pdf)
2 | [sheep](https://www.drivendata.org/users/sheep/) | 1st | [Read the report](https://github.com/drivendataorg/tick-tick-bloom/blob/main/1st%20Place/reports/1st-Place_DrivenData-Competition-Winner-Documentation.pdf)

**Winners Announcement: [Meet the winners of the Tick Tick Bloom challenge](https://drivendata.co/blog/tick-tick-bloom-challenge-winners)**

**Benchmark Blog Post: [How to predict harmful algal blooms using LightGBM and satellite imagery](https://drivendata.co/blog/tick-tick-bloom-benchmark)**
