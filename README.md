# BUSA3020 Group Assignment: Predicting Airbnb Listing Prices in Brisbane

This repository contains the Jupyter Notebook and associated project files for the **BUSA3020 Advanced Analytics Techniques** group assignment (Semester 1, 2026). The goal of this project is to build machine learning models to forecast the nightly listing prices for Airbnb properties in Brisbane, Australia, as part of a Kaggle competition.

## Project Overview

Accurate price forecasting helps hosts set competitive nightly rates to balance occupancy and revenue, while assisting platforms in offering dynamic pricing suggestions. To address this, we developed a supervised machine learning regression pipeline that learns patterns from historic listings, including location, property attributes, host characteristics, and review activity.

Kaggle Competition Link: [Brisbane Airbnb Price Prediction](https://www.kaggle.com/t/554f205a43a94920943d5227ae9dcbdb)

## Team Information
- **Team Name on Kaggle:** `BUSA3020 Duo Carry`
- **Team Leader / Member 1:** `Ba Duc Thanh Nguyen`
- **Team Member 2:** `Viet Hoai Bac Huynh`

## Tasks Breakdown

The assignments and Notebook follow three primary structured deliverables:

### Task 1: Problem Description and Exploratory Data Analysis (EDA)
- Defining the target goal, business logic and the chosen metric.
- Describing the evaluation metric **Mean Absolute Percentage Error (MAPE)** and evaluating its business relevance.
- Diagnosing missing values and providing comprehensive summary statistics for categorical and numerical features.
- Handling outlier evaluations.

### Task 2: Data Cleaning, Missing Observations, and Feature Engineering
- Constructing the feature sets and applying consistent transformation tasks for Train and Test sets.
- Replacing data outliers and filling/imputing sparse and missing values.
- One-hot encoding categorical variables and mapping ordinal data (e.g. `host_response_time`), while applying scaling to numeric data (e.g., standardisation).
- Crafting relevant pricing markers like location premium patterns, listing capacities, and demand proxies.

### Task 3: Model Fitting, Tuning, and Prediction
- Fitting models using 5-fold cross-validation and searching hyperparameters with `GridSearchCV` optimized to minimize MAPE.
- We compared three core algorithms on the curated features:
  1. **Ridge Regression:** A linear method with $L_2$ regularization for swift and interpretable base predictions.
  2. **Random Forest Regression:** An ensemble method capturing varying non-linear interactions across features.
  3. **HistGradientBoostingRegressor:** An advanced tree-based boosting algorithm capable of efficient handling of non-linear patterns on larger sets of features.
- Evaluating out-of-fold performance against validations to synthesize the final model used to generate predictions on Kaggle.

## Repository Contents

- `busa3020_groupproject.ipynb`: The primary notebook file containing the end-to-end workflow (EDA, transformations, and predictive logic).
- `train.csv`, `test.csv`: Challenge data files (handled inside the notebook logic).
- `metaData.csv`: Attribute descriptive metadata provided by the Kaggle competition mapping variables representation.
- `Kaggle-Prize-Pipeline/`: Directory holding an alternative modeling approach or winning pipelines with advanced text extraction features.

## Setup and Running

1. **Prerequisites:** 
Make sure you have standard Python data science libraries such as `numpy`, `pandas`, `scikit-learn`, `matplotlib`, and `seaborn`.

     ```bash
     pip install pandas numpy scikit-learn matplotlib seaborn
     ```

2. **Execution:** 
Open the `busa3020_groupproject.ipynb` in Jupyter Notebook, JupyterLab, or VS Code and run exactly top to bottom to witness the EDA generation, preprocessing transformations, and evaluation metrics mapped back to outputs formatting.
