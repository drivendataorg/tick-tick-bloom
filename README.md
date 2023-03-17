[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

![2017 algal bloom in Lake Erie seen as bright green splotches, captured by the NASA/USGS Landsat mission](https://drivendata-public-assets.s3.amazonaws.com/competition_cyano_banner.jpeg)

# Tick Tick Bloom: Harmful Algal Bloom Detection Challenge

## Goal of the Competition

Inland water bodies provide a variety of critical services for both human and aquatic life, including drinking water, recreational and economic opportunities, and marine habitats. A significant challenge water quality managers face is the formation of **harmful algal blooms (HABs). One of the major types of HABs is cyanobacteria.** HABs produce toxins that are poisonous to humans and their pets, and threaten marine ecosystems by blocking sunlight and oxygen. Manual water sampling, or “in situ” sampling, is generally used to monitor cyanobacteria in inland water bodies. In situ sampling is accurate, but time intensive and difficult to perform continuously.

**The goal in this challenge was to use satellite imagery to detect and classify the severity of cyanobacteria blooms in small, inland water bodies.** The resulting algorithm will help water quality managers better allocate resources for in situ sampling, and make more informed decisions around public health warnings for critical resources like drinking water reservoirs. Ultimately, more accurate and more timely detection of algal blooms helps keep both the human and marine life that rely on these water bodies safe and healthy.

## What's in this Repository

This repository contains code from winning competitors in the [Tick Tick Bloom: Harmful Algal Bloom Detection](https://www.drivendata.org/competitions/143/tick-tick-bloom/page/649/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place |Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | sheep | 0.7554 | 0.7608 | <Description from the 1st place's writeup>
2   | apwheele | 0.7476 | 0.7616 | Three regression models (LGBM, XGBoost, and catboost) are trained on date, location, elevation data, and satellite imagery. K-means segmentation is used to identify the lake area in images.
3   | BrandenKMurray | 0.7614 | 0.7745 | <Description from the 3rd place's writeup>

### Model Write-up Bonus

Team or User | Public Score | Private Score | Summary of Model
--- | ---   | ---   | ---
user_name | 0.858 | 0.859 | <Description from the 1st place's writeup>
user_name | 0.857 | 0.857 | <Description from the 2nd place's writeup>

Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Benchmark Blog Post: [How to predict harmful algal blooms using LightGBM and satellite imagery](https://drivendata.co/blog/tick-tick-bloom-benchmark)**