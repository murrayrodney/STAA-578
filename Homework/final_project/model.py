# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import layers, models, optimizers, callbacks
import mlflow
import mlflow.keras
import matplotlib
from keras.utils import to_categorical
import itertools
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

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
def calc_baseline(df, y_col):
    df_eval = df.copy()
    df_eval["shifted"] = df_eval.groupby(level=0)[y_col].shift(1)
    df_eval.dropna(subset=["shifted"], inplace=True)

    mae = np.abs(df_eval[y_col] - df_eval["shifted"]).mean()
    return mae


def get_numpy_arrays(dataset):
    dataset_inputs = []
    dataset_outputs = []
    for inputs, outputs in dataset.as_numpy_iterator():
        dataset_inputs.append(inputs)
        dataset_outputs.append(outputs)

    dataset_inputs = np.concatenate(dataset_inputs)
    dataset_outputs = np.concatenate(dataset_outputs)
    return dataset_inputs, dataset_outputs


train_inputs, train_outputs = get_numpy_arrays(train_dataset)
val_inputs, val_outputs = get_numpy_arrays(val_dataset)
test_inputs, test_outputs = get_numpy_arrays(test_dataset)

with mlflow.start_run(experiment_id=experiment_id) as run:
    dfs = [train, val, test]
    names = ["train", "val", "test"]
    metrics = {}
    for (i, df), col in itertools.product(enumerate(dfs), y_cols):
        metrics[f"{names[i]}_{col}_mae"] = calc_baseline(df, col)
    mlflow.log_param("model_type", "baseline")
    mlflow.log_metrics(metrics)

# %% [markdown]
# # Train an LSTM model

# %%
train_well_cat = to_categorical(pd.Categorical(train.index.get_level_values(0)).codes)
val_well_cat = to_categorical(pd.Categorical(val.index.get_level_values(0)).codes)

# %%
n_out = 128

inputs = layers.Input(shape=(12, len(x_cols + y_cols)))
norm = layers.Normalization()(inputs)
x = layers.LSTM(n_out, return_sequences=True)(norm)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(n_out, return_sequences=False)(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(len(y_cols), activation="relu")(x)
model = models.Model(inputs, outputs)

optimizer = optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

model.summary()

# %%
fit_callbacks = [
    callbacks.EarlyStopping(patience=20, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(patience=10, monitor="val_loss", factor=0.5),
]
with mlflow.start_run(experiment_id=experiment_id) as run:
    mlflow.keras.autolog()
    mlflow.log_params({"model_type": "lstm", "n_out": n_out})
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=500,
        # epochs=3,
        callbacks=fit_callbacks,
        batch_size=2048,
    )
    train_pred = model.predict(train_dataset, verbose=0)
    val_pred = model.predict(val_dataset, verbose=0)

    # Get metrics for the predictions
    dataset_ouputs = [train_outputs, val_outputs]
    names = ["train", "val"]
    preds = [train_pred, val_pred]
    metrics = {}
    for i, col in enumerate(y_cols):
        for name, pred, outputs in zip(names, preds, dataset_ouputs):
            metrics[f"{name}_{col}_mae"] = mean_absolute_error(
                outputs[:, i], pred[:, i]
            )

    # Log the metrics to mlflow
    mlflow.log_metrics(metrics)

# %%
test_pred = model.predict(test_dataset, verbose=0)

dataset_ouputs = [train_outputs, val_outputs, test_outputs]
names = ["train", "val", "test"]
preds = [train_pred, val_pred, test_pred]
metrics = {}
for i, col in enumerate(y_cols):
    for name, pred, outputs in zip(names, preds, dataset_ouputs):
        metrics[f"{name}_{col}_mae"] = mean_absolute_error(outputs[:, i], pred[:, i])
metrics

# %%
