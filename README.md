# First-sample-ML-Project
My first end-to-end ML project, built to learn the full pipeline: data loading, cleaning, EDA, model training, evaluation, and interpretation — without relying on any helper frameworks.
 
---
 
## Overview
 
This notebook predicts Chicago taxi fares using linear regression, built from scratch without any helper libraries. It walks through a complete machine learning pipeline — from raw data to live predictions — using the Chicago Taxi Trips dataset from Google ML Education. Two models are trained and compared to show how adding features improves performance.
 
---
 
## Dataset
 
| Property | Value |
|---|---|
| Source | [Chicago Taxi Trips — Google ML Education / BigQuery](https://cloud.google.com/bigquery/public-data) |
| Size | 31,694 rows × 18 columns |
| Target variable | `FARE` (trip fare in USD) |
 
---
 
## Project Structure
 
The notebook is organized into 8 sections:
 
1. **Load the Data** — read and preview the raw dataset
2. **Exploratory Data Analysis** — distributions, correlations, outlier detection
3. **Data Cleaning** — handle missing values and remove invalid trips
4. **Feature Engineering** — derive `TRIP_MINUTES`, `TRIP_SPEED`, and other useful columns
5. **Model A** — simple linear regression using `TRIP_MILES` only
6. **Model B + Comparison** — multi-feature model with side-by-side evaluation
7. **Final Test Evaluation** — held-out test set results for the best model
8. **Predictions on New Rides** — run the model on hypothetical trips
---
 
## Results
 
| | Model A | Model B |
|---|---|---|
| Features | `TRIP_MILES` | `TRIP_MILES`, `TRIP_MINUTES`, `TRIP_SPEED` |
| MAE | $1.57 | $1.19 |
| RMSE | $3.69 | $3.79 |
| R² | 0.9514 | 0.9513 |
 
Model B predicts fares with an average error of just **$1.19**, explaining **95.1% of fare variance**.
![Result]("Result.png")

---
 
## Requirements
 
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```
 
Install all at once:
 
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
 
---
 
## How to Run
 
1. Clone or download the repo
2. Open `linear_regression.ipynb` in VS Code or Jupyter
3. Run all cells top to bottom
---
 
## Key Takeaways
 
- Adding trip duration and speed meaningfully reduces prediction error over mileage alone
- Linear regression coefficients are directly interpretable as each feature's contribution to fare
- Potential next steps: time-of-day features, pickup zone encoding, tree-based models, or neural networks
