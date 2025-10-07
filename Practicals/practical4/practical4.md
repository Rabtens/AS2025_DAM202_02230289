# Weather Prediction using RNNs

This notebook demonstrates how to build and compare different Recurrent Neural Network (RNN) architectures (SimpleRNN, LSTM, and GRU) for time series weather prediction. The goal is to predict the daily temperature based on historical weather data.

## Data

The dataset used is `weather_data.csv`. It contains daily weather information including:
- Year
- Day
- Wind Speed
- Specific Humidity
- Relative Humidity
- Precipitation
- Temperature

## Methodology

The notebook follows these steps:

1.  **Setup and Imports**: Import necessary libraries and mount Google Drive to access the data.
2.  **Data Preprocessing**:
    *   Load the data and format it, setting the Date as the index.
    *   Perform exploratory data analysis, including visualizing time series trends for each feature.
    *   Clean the data by handling missing values using forward fill, backward fill, and linear interpolation.
    *   Engineer new features like day of the year, month, season, and cyclical features (sin/cos of day of year).
    *   Add moving averages and lag features for 'Temperature', 'Wind_Speed', and 'Relative_Humidity'.
    *   Drop unnecessary columns and rows with remaining missing values.
3.  **Train-Test Split**: Split the data into training, validation, and test sets while preserving the time series order.
4.  **Scaling and Sequence Creation**:
    *   Normalize the features and target variable ('Temperature') using `MinMaxScaler` fitted only on the training data to prevent data leakage.
    *   Create sequences of historical data (defined by `sequence_length`) as input (X) and the corresponding future temperature value as the target (y) for the RNN models.
5.  **Model Architecture, Training, and Evaluation**:
    *   Define a function to build SimpleRNN, LSTM, and GRU models with configurable hidden units, dropout rate, and learning rate.
    *   Train each model using the training and validation data. Early stopping is used to prevent overfitting.
    *   Evaluate each trained model on the unseen test set using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R2 Score.
6.  **Model Evaluation and Comparison**:
    *   Plot the training and validation loss curves for each model to visualize learning progress and identify potential overfitting.
    *   Plot segments of the actual vs. predicted temperature values from the test set for a visual comparison of model performance.
    *   Generate a comparison table summarizing the evaluation metrics for all models.

## Results

The notebook trains and evaluates SimpleRNN, LSTM, and GRU models. The comparison table shows the performance of each model on the test data, including metrics like Test Loss (MSE), RMSE, MAE, and R2 Score. The GRU model demonstrates the best performance across these metrics for this dataset. The learning curve plots illustrate the training progress, and the prediction plots show how well each model predicts the actual temperature values.


## How to Run

1.  Upload the `weather_data.csv` file to your Google Drive.
2.  Update the `data_path` variable in the first code cell to the correct path of the file in your Google Drive.
3.  Run all the code cells in the notebook sequentially.
4.  The output will include data exploration details, training progress for each model, learning curves, prediction plots, and a final performance comparison table.

## Dependencies

The notebook requires the following libraries:

*   numpy
*   pandas
*   matplotlib
*   seaborn
*   tensorflow
*   sklearn
*   itertools