# Assignment 1 - Laser Measurement Prediction

## Requirements
```bash
pip install -r requirements.txt
```

## Models Implemented
- Long Short Term Memory (LSTM)

## Usage
```bash
python main.py
```

## Project Structure
- `main.py` — 
- `Models` — Used models for the laser measurement prediction
    - `lstm.py` — Main module class for the Long Short Term Memory Model. Inlcudes basic model functions as create, compile, fit, summary and fit
- `Data` — Folder to store dataset files
    - `Xtrain.mat` — Original train dataset file
    - `xtrain.csv` — Train dataset Xtrain.mat but transformed to .csv file
- `Utils` — Utils needed for the implementation of the laser measurement prediction
    - `transform_data.py` — Prepare data for train and test, scale them and build sliding windows for the sequences
    - `plotting_helpers.py` — Plots for understanding original dataset, predictions, autocorreleation and training history
    - `forecasting.py` — Predict the next 200 datapoints once the model is trained
- `Images` — Stores images of the plots made with the plotting helpers and forecast

## Authors
- Andrei Medesan (4799526)
- Kumkum Singh ()
- Barath Kumar ()
- Diana Luna (9876847)

## --------------- Remaining tasks (TO DO): *** delete later

For the model:

- experiment with different values for the number of neurons in layers
- check if dropout is helping, otherwise use early stopping
- try clipping gradients to prevent exploding gradients (add clip_norm=1 at the optimizer)
- build a baseline model for comparison?
- for now the model seems really good, so the above tasks are optional and for error checking
