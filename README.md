# Assignment 1 - Laser Measurement Predictions

This repository contains all the files necessary for one-step prediction on laser intensity using a Long Short-Term Memory (LSTM) model. 

## Requirements
```bash
pip install -r requirements.txt
```

## Models Implemented
- Long Short Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Usage
```bash
python main.py
```

## Project Structure
- `main.py` — Manage overall flow of the models to implement forecasting task
- `Models` — Used models for the laser measurement prediction
    - `lstm.py` — Main module class for the Long Short Term Memory model. Inlcudes basic model functions as create, compile, fit, summary and fit
    - `gru.py` — Main module class for the Gated Recurrent Unit model. Inlcudes basic model functions as create, compile, fit, summary and fit
- `Data` — Folder to store dataset files
    - `Xtrain.mat` — Original train dataset file
    - `Xtest.mat` — Original test dataset file
    - `xtrain.csv` — Train dataset Xtrain.mat but transformed to .csv file
- `Utils` — Utils needed for the implementation of the laser measurement prediction
    - `transform_data.py` — Prepare data for train and test, scale them and build sliding windows for the sequences
    - `plotting_helpers.py` — Plots for understanding original dataset, predictions, autocorreleation and training history
    - `forecasting.py` — Predict the next 200 datapoints once the model is trained
    - `tune_lookback.py` — Tune hyperparameter of the models with window lookback
    - `predict_test.py` — Evaluation of testdataset
- `Images` — Stores images of the plots made with the plotting helpers and forecast

## Authors
- Andrei Medesan (4799526)
- Kumkum Singh (0917656)
- Bharath Kumar (8874476)
- Diana Luna (9876847)

