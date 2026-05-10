# deepLearning

Remaining tasks (TO DO):

- fix the splitting phase by using time-series cross-validation so collapses appear in validation at least once. This can be done with `sklearn.model_selection.TimeSeriesSplit (n_splits=5)` on the training set, each fold uses an expanding window for train and the next chunk for validation. Report the mean ± std test metrics across folds.
- save the best-epoch model, not the last through callbacks. 
    ```python
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint("models/best_lstm.h5", monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(..., callbacks=callbacks)
    ```
- (optional) verify the GRU recursive forecast is not a limit cycle.
- test residual whiteness and bias with statistical tests:
    ```python
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    residuals = testY_real - testPred_real
    print("Mean residual:", residuals.mean())
    print("Bias t-test p-value:", stats.ttest_1samp(residuals, 0).pvalue)
    print(acorr_ljungbox(residuals, lags=[10, 20, 40], return_df=True))
    ```
- run multiple seeds for a fair LSTM-vs-GRU comparison
  ```python
  import pandas as pd

  def run_multi_seed(train_fn, X_train, y_train, X_test, y_test, scaler,
                     seeds=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), name=""):
      results = []
      for s in seeds:
          tf.keras.backend.clear_session()   # avoid memory bloat across runs
          _set_seeds(s)
          metrics = train_fn(X_train, y_train, X_test, y_test, scaler)
          metrics["seed"] = s
          results.append(metrics)
      df = pd.DataFrame(results)
      print(f"\n[{name}] over {len(seeds)} seeds:")
      print(df[["test_rmse", "test_mae"]].agg(["mean", "std"]))
      return df

  lstm_df = run_multi_seed(train_lstm, X_train, y_train, X_test, y_test, scaler, name="LSTM")
  gru_df  = run_multi_seed(train_gru,  X_train, y_train, X_test, y_test, scaler, name="GRU")

  # significance test on the difference
  from scipy.stats import ttest_ind
  t, p = ttest_ind(lstm_df["test_rmse"], gru_df["test_rmse"], equal_var=False)
  print(f"Welch t-test on test RMSE: t={t:.2f}, p={p:.4f}")
  ```