# Energy Consumption Prediction

A machine learning project that predicts energy consumption using **Linear Regression** and **Gradient Boosting**, built and trained in Google Colab.

## Overview

This project explores two regression approaches to forecast building energy consumption (kWh) from environmental and occupancy data:

- **Linear Regression** — a baseline model to capture linear trends in the data
- **Gradient Boosting Regressor** — an ensemble model, tuned via `GridSearchCV` over `n_estimators`, `learning_rate`, and `max_depth`, to capture more complex, non-linear relationships

Both models were trained and evaluated on the same held-out test set using RMSE and R².

## What's Inside

- **Exploratory Data Analysis:** distribution of energy consumption, average consumption by hour of day and day of week, holiday vs. non-holiday usage patterns, correlation heatmap, and pairwise feature relationships
- **Feature engineering:** extracted `hour`, `month`, and `day_of_week` from the raw timestamp
- **Preprocessing pipeline:** `ColumnTransformer` with `StandardScaler` on numerical features and `OneHotEncoder` on categorical features (HVAC usage, lighting usage, holiday), wrapped in a scikit-learn `Pipeline`
- **Model training & tuning:** Linear Regression baseline + Gradient Boosting tuned with grid search and cross-validation

## Tech Stack

- Python
- Google Colab
- pandas / NumPy (data handling)
- scikit-learn (preprocessing pipeline, Linear Regression, Gradient Boosting, GridSearchCV, evaluation metrics)
- Matplotlib / Seaborn (visualization)

## Dataset

Features include: Temperature, Humidity, SquareFootage, Occupancy, HVAC Usage, Lighting Usage, Holiday, and Timestamp (used to derive hour/month/day-of-week). Target variable: Energy Consumption (kWh).

> **Note:** The dataset used in this notebook is loaded from Google Drive and is not included in this repository.
> To run the notebook yourself:
> 1. Add your own copy of the dataset to your Google Drive.
> 2. Update the file path in the data-loading cell of the notebook to point to your dataset.

## Project Structure

```
energy-consumption-prediction/
├── README.md
├── energy_consumption_prediction.ipynb   # Main notebook: data prep, training, evaluation
├── requirements.txt                      # Python dependencies
└── .gitignore
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/shadowBit01/energy-consumption-prediction.git
   ```
2. Open `energy_consumption_prediction.ipynb` in Google Colab or Jupyter Notebook.
3. Update the dataset path to point to your own copy.
4. Run all cells to preprocess the data, train both models, and view evaluation results.

## Results

| Model               | Accuracy (approx.) |
|---------------------|---------------------|
| Linear Regression    | ~60%                |
| Gradient Boosting    | ~60%                |

## Future Improvements

- Hyperparameter tuning for Gradient Boosting
- Feature engineering to improve accuracy
- Trying additional models (Random Forest, XGBoost)
- Cross-validation for more robust evaluation

## Author

**Divyansh Agrawal**
[GitHub](https://github.com/shadowBit01)
