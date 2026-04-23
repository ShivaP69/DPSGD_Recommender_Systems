# Differentially Private Recommender Systems

This project implements four different recommender system models with support for Local Differential Privacy (LDP) and Differential Privacy using DP-SGD (DPSGD). 
Including: Beysian Personalized rankig, Variational Auto-encoder, Singular Value decomposition and Neural Colaborative Filtering. 
We evaluate these models on various metrics, including:

- NDCG, Recall
- Calibration (KLD)
- Coverage
- DPF (Disparate Popularity Fairness)
- Exposure
- Popularity Loss (PL)
- Novelty

# Dataset

This project uses two main datasets: MovieLens 1M and the Yelp Open Dataset.

## MovieLens 1M Dataset

Please download the following files from the official GroupLens website (https://grouplens.org/datasets/movielens/1m/):

    movies.csv

    ratings.dat

Make sure these files are placed in the model's working directory to ensure proper execution of the code.

## Yelp Open Dataset

You will need the following files from the Yelp Open Dataset (https://business.yelp.com/data/resources/open-dataset/):

    yelp_academic_dataset_review.json

    yelp_academic_dataset_business.json

Additionally, our project uses a supplementary file:

    result.json, located in the Yelp_relevant_dataset/ directory.

This file contains crawled category data. If needed, you can regenerate it using our code by calling the read_df_item_and_df_business function found in main/training.py.

    ⚠️ Note: Crawling this data requires internet access and may be affected by changes to the structure of the Yelp category list website (https://blog.yelp.com/businesses/yelp_category_list/). For reproducibility and to avoid potential issues, we recommend using the provided result.json file.



# Project Structure

Important: Many scripts and utility functions (e.g., metric evaluators, calibration functions) are shared across multiple models and datasets. 

#  Key Features

- 4 recommender system models
- Support for both LDP and DPSGD
- Metrics-based evaluation
- Organized for clean reproducibility
- CSV exports of evaluation metrics

# visualized results

All plots in the paper are reproducible using the codes provided in the visualized_results directory. 

