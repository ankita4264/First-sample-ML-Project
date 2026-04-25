# First-sample-ML-Project
When I first started learning Machine Learning I started with this basic fundamental ML project deploying Linear Regression Algorithm. 
This is a beginner ML project that predicts Chicago taxi fares using linear regression, built from scratch without any helper libraries.
This notebook walks through a complete machine learning pipeline — from raw data to live predictions — using the Chicago Taxi Trips dataset from Google ML Education. Two models are trained and compared to show how adding features improves performance.

Dataset I used: 
-- Source: Chicago Taxi Trips (Google ML Education / BigQuery)

-- Size: 31,694 rows × 18 columns

-- Target variable: FARE (trip fare in USD)


Project Structure:
The notebook is organized into 8 sections:
1] Load the Data — read and preview the raw dataset
2] Exploratory Data Analysis — distributions, correlations, outlier detection
3] Data Cleaning — handle missing values and remove invalid trips
4] Feature Engineering — derive TRIP_MINUTES, TRIP_SPEED, and other useful columns
5] Model A — simple linear regression using TRIP_MILES only
6] Model B + Comparison — multi-feature model (TRIP_MILES, TRIP_MINUTES, TRIP_SPEED) with side-by-side evaluation
7] Final Test Evaluation — held-out test set results for the best model
8] Predictions on New Rides — run the model on hypothetical trips

Results: Model B explains 95.1% of fare variance, with predictions off by just $1.19 on average.

Requirements:

-- pandas

-- numpy

-- matplotlib

-- seaborn

-- scikit-learn


How to Run:

-- Clone or download the repo

-- Open linear_regression.ipynb in VS Code or Jupyter

-- Run all cells top to bottom


Key Takeaways:
Adding trip duration and speed meaningfully reduces prediction error over mileage alone
Linear regression coefficients are directly interpretable as each feature's contribution to fare
Potential next steps: time-of-day features, pickup zone encoding, tree-based models, or neural networks

About:
This was my first end-to-end ML project, built to learn the full pipeline: data loading, cleaning, EDA, model training, evaluation, and interpretation — without relying on any helper frameworks.Sonnet 4.6
