# Predicting NFL Point Spread Coverage Using Machine Learning

**STAT 301-2: Data Science 2 - Final Project**

**Author:** Thomas McCabe

## Overview

This project builds a binary classification model to predict whether the closing favorite in an NFL game will cover the point spread. Using team-level offensive and defensive statistics, betting lines, Elo ratings, and game conditions from 2012–2024 NFL regular season games, eight model types are compared across three preprocessing recipes to identify the best predictive approach.

## Repository Structure

```
├── scripts/                                # R scripts (run in numbered order)
│   ├── 00_project_setup.R                  # Shared setup, libraries, helper functions
│   ├── 01_build_model_data.R               # Data loading, cleaning, feature engineering
│   ├── 02_prepare_modeling_objects.R        # Train/test split, CV folds, recipes
│   ├── 03_define_workflows.R               # Workflow definitions (calls 03a–03d)
│   ├── 03a_define_baseline_logistic.R       # Null + logistic workflows
│   ├── 03b_define_regularized_models.R      # Lasso, Ridge, Elastic Net workflows
│   ├── 03c_define_tree_models.R             # Random Forest + XGBoost workflows
│   ├── 03d_define_knn_models.R              # K-Nearest Neighbors workflows
│   ├── 04_fit_workflows.R                   # Fit all workflows (calls 04a–04d)
│   ├── 04a_fit_baseline_logistic.R          # Fit baseline + logistic
│   ├── 04b_fit_regularized_models.R         # Fit regularized models
│   ├── 04c_fit_tree_models.R                # Fit tree-based models
│   ├── 04d_fit_knn_models.R                 # Fit KNN models
│   ├── 05_collect_results.R                 # Collect and compare CV results
│   ├── 06_fit_expanded_workflows.R          # Fit expanded/curated recipe workflows
│   ├── 07_audit_and_select_final_model.R    # Select best model, test set evaluation
│   └── 08_make_report_assets.R              # Generate figures and report assets
├── data/                                    # Raw datasets + codebook (see data/README.md)
├── objects/                                 # Saved RDS model objects
├── results/                                 # CSV result tables
├── figures/                                 # Generated plot PNGs
├── memos/                                   # Progress memos 1 & 2
├── McCabe_Thomas_Final_Report.qmd           # Final report (renders to HTML)
├── McCabe_Thomas_Executive_Summary.qmd      # Executive summary
└── final-project-2-tommccabe9933.Rproj      # RStudio project file
```

## How to Reproduce

1. Open `final-project-2-tommccabe9933.Rproj` in RStudio
2. Run scripts in `scripts/` in numbered order (00 through 08)
3. Render `McCabe_Thomas_Final_Report.qmd` and `McCabe_Thomas_Executive_Summary.qmd`

## Data Sources

- [NFL Team Data 2003-2023](https://www.kaggle.com/datasets/nickcantalupa/nfl-team-data-2003-2023) (Cantalupa)
- [NFL Scores and Betting Data](https://www.kaggle.com/datasets/tobycrabtree/nfl-scores-and-betting-data) (Spreadspoke)
- [NFL Stats 2012-2024](https://www.kaggle.com/datasets/philiphyde1/nfl-stats-2012-2024) (Hyde)


