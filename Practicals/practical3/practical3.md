# Simple RNN Practical – Exercise Guide

This notebook is a **hands-on exercise** to learn how to use a Simple Recurrent Neural Network (RNN) for time-series data (weather prediction).  
It belongs to the DAM202 – Sequence Models course.

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

## Requirements

Install these packages:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn
```

Use Python 3 with Jupyter or Google Colab.

## Input Data Format

The CSV file should have columns like:

| Year | Day | Wind_Speed | Specific_Humidity | Relative_Humidity | Precipitation | Temperature |

The notebook converts Year + Day into a date index for time-series work.

## Expected Output

- A trained Simple RNN model.
- Metrics (MSE, MAE, R²).
- Plots comparing predicted and actual values.
- Notes on how changes in parameters affect model performance.

## Credits

Prepared as a practical exercise for DAM202 – Sequence Models to practice building and testing Simple RNNs.