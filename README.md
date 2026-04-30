# deepLearning

Remaining tasks (TO DO):

For the model:

- experiment with different values for the number of neurons in layers
- check if dropout is helping, otherwise use early stopping
- try clipping gradients to prevent exploding gradients (add clip_norm=1 at the optimizer)
- set the seed for reproducibility
- build a baseline model for comparison?


Main program flow:

- make the main.py modular, split into functions
- plot the predicted vs actual values
- add training and validation learning curves (rethink if we need validation at all)
- now, the model is good, the RMSE on both sets show a good starting point, no overfitting
- plot the autocorrelation function (statsmodels.tsa.stattools.acf)

