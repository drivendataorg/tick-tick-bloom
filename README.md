[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![2017 algal bloom in Lake Erie seen as bright green splotches, captured by the NASA/USGS Landsat mission](https://drivendata-public-assets.s3.amazonaws.com/competition_cyano_banner.jpeg)

# Tick Tick Bloom: Harmful Algal Bloom Detection Challenge

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
