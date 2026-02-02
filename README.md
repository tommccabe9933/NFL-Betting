
# Final Project 2 – STAT 301-2

**Predicting NFL Point Spread Coverage**
Tom McCabe

## Overview

This project predicts whether an NFL team will **cover the closing point spread** using pre-game team performance, aggregated player offense, and betting context. The task is a **binary classification problem** (cover vs no cover) designed to test whether structured data contains predictive signal beyond betting market baselines.

## Repository Structure

data/    # Raw NFL team, player, and betting datasets  
memos/   # Quarto progress memo(s) (no visible code)  
*.Rproj  # RStudio project file  
README.md

## Data Sources

Public NFL datasets from Kaggle:

* Player-level offense (Philip Hyde)
* Team offense and defense (Nick Cantalupa)
* Scores and betting data (Spreadspoke)

The current analysis uses the 2025 season sample; future work will extend to multiple seasons.

## Target Variable

**Spread coverage**, defined as whether a team covers the closing point spread. Pushes are rare and excluded, yielding a clean binary outcome.

## Reproducibility

* Analysis conducted in R using tidyverse and Quarto
* Memo outputs contain no visible code, per course guidelines
* Project is fully reproducible when opened as an RStudio project

---
