# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models, optimizers, callbacks
from keras.utils import timeseries_dataset_from_array
from tensorflow.data import Dataset
import mlflow
import mlflow.keras
import matplotlib
import tensorflow as tf
from copy import deepcopy
from tqdm import tqdm
import joblib

matplotlib.style.use("ggplot")

# %% tags=["parameters"]
upstream = None
product = None
x_cols = None
y_cols = None

# %%
train = pd.read_parquet(upstream["train_test_split"]["train_df"])
val = pd.read_parquet(upstream["train_test_split"]["val_df"])
test = pd.read_parquet(upstream["train_test_split"]["test_df"])

# %%
train_dataset = tf.data.Dataset.load(upstream["train_test_split"]["train_dataset"])
val_dataset = tf.data.Dataset.load(upstream["train_test_split"]["val_dataset"])
test_dataset = tf.data.Dataset.load(upstream["train_test_split"]["test_dataset"])

# %%
experiment_name = "well_forecasting"
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id


# %%
def calc_baseline(df):
    df_eval = df.copy()
    df_eval["shifted_oil"] = df_eval.groupby(level=0)["oil_rate"].shift(1)
    df_eval.dropna(subset=["shifted_oil"], inplace=True)

    mae = np.abs(df_eval["oil_rate"] - df_eval["shifted_oil"]).mean()
    return mae


with mlflow.start_run(experiment_id=experiment_id) as run:
    metrics = {
        "train_mae": calc_baseline(train),
        "val_mae": calc_baseline(val),
        "test_mae": calc_baseline(test),
    }
    mlflow.log_param("model_type", "baseline")
    mlflow.log_metrics(metrics)

# %% [markdown]
# # Train an LSTM model

# %%
n_out = 16

inputs = layers.Input(shape=(12, len(x_cols)))
x = layers.LSTM(n_out)(inputs)
outputs = layers.Dense(1)(x)
model = models.Model(inputs, outputs)

optimizer = optimizers.Adam()
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

model.summary()

# %%
train_pred = model.predict(train_dataset)
train_pred.shape

# %%
fit_callbacks = [
    callbacks.EarlyStopping(patience=20, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=10, monitor="val_loss", factor=0.5),
]
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.keras.autolog()
    mlflow.log_params({"model_type": "lstm", "n_out": n_out})
    model.fit(
        train_dataset, validation_data=val_dataset, epochs=3, callbacks=fit_callbacks
    )

# %%
