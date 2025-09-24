# Simple RNN Practical – Exercise Guide

This notebook is a **hands-on exercise** to learn how to use a Simple Recurrent Neural Network (RNN) for time-series data (weather prediction).  
It belongs to the DAM202 – Sequence Models course, focusing on practical implementation of sequence models for real-world applications.

## Overview

In this practical, you'll work with weather data to predict temperature values using a Simple RNN. This exercise demonstrates how RNNs can capture temporal patterns in sequential data, making them ideal for time-series forecasting tasks.

### Why Weather Prediction?
- Real-world application of sequence modeling
- Rich temporal dependencies in weather patterns
- Multiple correlated features (humidity, wind speed, etc.)
- Clear evaluation metrics for prediction accuracy

### What You'll Build
A complete end-to-end machine learning pipeline that:
1. Processes time-series weather data
2. Trains a Simple RNN model
3. Makes future temperature predictions
4. Evaluates and visualizes the results

---

## Learning Goals

By finishing this notebook, you will:

- Load and prepare a weather dataset for RNN training.
- Build a Simple RNN model in TensorFlow/Keras.
- Train, test and evaluate the model.
- Try out changes (like sequence length, hidden units, dropout) to see their effect.

---

## Exercise Steps

Follow these steps in the notebook:

1. **Setup**  
   - Install the required Python packages.  
   - Mount Google Drive if using Colab.  

2. **Load Data**  
   - Use the given `load_weather_data()` function to load the CSV file.  
   - Check shape, date range and columns of the dataset.  

3. **Pre-process Data**  
   - Scale features with MinMaxScaler.  
   - Split into train and test sets.  
   - Create input sequences and labels for the RNN.  

4. **Build the Simple RNN Model**  
   - Add a `SimpleRNN` layer with hidden units.  
   - Add a `Dense` output layer.  
   - Compile with Adam optimizer and a loss function.  

5. **Train the Model**  
   - Fit the model on training data.  
   - Use callbacks like EarlyStopping or ModelCheckpoint.  

6. **Evaluate the Model**  
   - Predict on test data.  
   - Calculate metrics: MSE, MAE, R².  
   - Plot predicted vs. actual values.  

7. **Experiment (Main Exercise)**  
   - Change sequence length (e.g., 3, 5, 7, 10).  
   - Change number of hidden units or dropout.  
   - Observe how results and plots change.  
   - Write down your observations.

---

## Files

Simple_RNN_practical3.ipynb # The main exercise notebook
weather_data.csv # Dataset (Bangladesh weather)

---

## Technical Requirements

### Software Requirements
- Python 3.8 or higher
- Jupyter Notebook or Google Colab
- Git (for cloning the repository)

### Required Python Packages
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
```

### Hardware Recommendations
- At least 4GB RAM
- CPU or GPU (GPU will significantly speed up training)
- Approximately 500MB free disk space

### Development Environment Setup
1. Clone the repository
2. Create a virtual environment (recommended)
3. Install the required packages
4. Launch Jupyter Notebook or open in Colab

## Data Description

The CSV file should have columns like:

| Year | Day | Wind_Speed | Specific_Humidity | Relative_Humidity | Precipitation | Temperature |

The notebook converts Year + Day into a date index for time-series work.

## Expected Outputs & Deliverables

### Model Artifacts
- Trained Simple RNN model (.h5 format)
- Saved data scalers for future predictions
- Model architecture summary

### Performance Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of determination)
- Training and validation loss curves

### Visualizations
1. Time series plots of actual vs predicted values
2. Error distribution histogram
3. Feature correlation heatmap
4. Training history plots (loss vs epochs)

### Documentation
- Written analysis of model performance
- Effects of parameter changes on predictions
- Suggestions for model improvements
- Key observations and insights

## Troubleshooting Tips

Common issues and solutions:
1. **Memory Errors**
   - Reduce batch size
   - Decrease sequence length
   - Use data generators

2. **Poor Model Performance**
   - Check data normalization
   - Adjust sequence length
   - Increase model capacity
   - Try different learning rates

3. **Overfitting**
   - Add dropout layers
   - Reduce model complexity
   - Use early stopping
   - Increase training data

## Credits & References

- Prepared for DAM202 – Sequence Models course
- Weather data sourced from Bangladesh Meteorological Department
- Based on TensorFlow/Keras RNN documentation
- Inspired by real-world time series forecasting applications

## Additional Resources

1. [TensorFlow RNN Guide](https://www.tensorflow.org/guide/keras/rnn)
2. [Time Series Forecasting Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
3. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
4. [Weather Forecasting with Deep Learning](https://www.nature.com/articles/s41586-021-03854-z)

Prepared as a practical exercise for DAM202 – Sequence Models to practice building and testing Simple RNNs.