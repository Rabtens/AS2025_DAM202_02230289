# Weather Prediction using Machine Learning

## Project Overview

This project develops a machine learning model to predict the weather category — **Sunny**, **Cloudy**, **Rainy**, or **Stormy** — for the next 6–12 hours using historical weather time-series data. The work was completed as part of the **DAM202 – Assignment II**.

## Objective

The main goals are:

- Build a model that can forecast short-term weather conditions
- Compare different machine learning models
- Use time-series validation to avoid data leakage
- Evaluate the models using proper classification and regression metrics

## Dataset

The dataset contains historical weather readings including:

- **Atmospheric Pressure** (hPa)
- **Wind Direction** (°)
- **Cloud Cover** (%)
- **Dew Point Temperature** (°C)
- **Solar Radiation** (W/m²)
- **Visibility** (km)
- **Sea Level Pressure**

These variables were selected because they are important indicators for short-term weather forecasting.

> **Note:** If hourly data was not available, daily data was processed to create approximate short-term predictions using interpolation and derived features.

## Data Preparation

Before training the models, several preprocessing steps were applied:

1. **Loading and Inspecting** — Initial dataset exploration and structure validation
2. **Cleaning Missing Values** — Using interpolation or filling techniques
3. **Handling Class Imbalance** — Resampling methods like SMOTE or class weights
4. **Feature Engineering** — Created temporal features including:
   - Lagged values
   - Rolling means
   - Time of day features
   - Pressure change rate
   - Wind direction shifts
5. **Target Variable Creation** — Weather categories generated based on pressure, cloud cover, and other meteorological parameters

## Model Development

Multiple machine learning models were implemented and compared:

- **Random Forest Classifier**
- **Gradient Boosting Classifier**
- **Logistic Regression**
- **Random Forest Regressor** (for continuous prediction comparison)

Each model was trained and evaluated using **Time Series Split cross-validation** to maintain the correct chronological order of weather data.

## Evaluation Metrics

The models were assessed using several metrics:

### Classification Metrics
- Accuracy
- F1-Score
- Precision
- Recall
- ROC-AUC
- Confusion Matrix

### Regression Metrics (for comparison)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

This approach helped identify both categorical prediction accuracy and numerical trend performance.

## Results Summary

Key findings from the analysis:

-  Models were able to classify weather categories with good accuracy
-  Random Forest and Gradient Boosting performed the best among all models
-  Time-Series validation ensured reliable evaluation results
-  Dataset quality and feature selection strongly affected prediction performance
-  Limitations exist when using daily data for 6–12-hour forecasts, but models still showed consistent short-term trends

![alt text](<Screenshot from 2025-10-20 09-14-28.png>)

## Recommendations

To further improve performance:

1. Include hourly weather data if available for better short-term forecasts
2. Add persistence and climatology baseline models for comparison
3. Add more pressure-based and temporal features (e.g., pressure change over the last 3 hours)
4. Visualize feature importance and confusion matrices to interpret results clearly

## Conclusion

The project successfully demonstrated how machine learning can predict short-term weather categories using time-series data. All core requirements were achieved:

-  Target variable creation
-  Multiple ML models compared
-  Time-series cross-validation
-  Proper evaluation metrics
-  Feature engineering and data cleaning

While the model performs well, using hourly atmospheric pressure data and more advanced features could further improve accuracy, especially for predicting Rainy and Stormy conditions.

## Acknowledgement

This assignment was carried out for **DAM202** under the guidance of the module tutor. All code, model explanations, and results are documented in the notebook file:

```
Assignment2_DAM202.ipynb
```

---

**Project Type:** Machine Learning Classification | **Course:** DAM202 | **Assignment:** II