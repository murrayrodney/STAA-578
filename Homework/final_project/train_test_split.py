# %%
import pandas as pd
from tqdm import tqdm
import joblib
from keras.utils import timeseries_dataset_from_array


# %% tags=["parameters"]
upstream = None
product = None
train_cutoff_date = None
val_cutoff_date = None
x_cols = None
y_cols = None


# %%
df = pd.read_parquet(upstream["process_data"]["input_data"])

wells = df["well_name_prod"].unique()
df = df[df["well_name_prod"].isin(wells[:100])]

# %%
train = df[df["reportdate"] < pd.Timestamp(train_cutoff_date)]
val = df[
    (df["reportdate"] >= pd.Timestamp(train_cutoff_date))
    & (df["reportdate"] < pd.Timestamp(val_cutoff_date))
]
test = df[df["reportdate"] >= pd.Timestamp(val_cutoff_date)]

train.set_index(["well_name_prod", "reportdate"], inplace=True)
val.set_index(["well_name_prod", "reportdate"], inplace=True)
test.set_index(["well_name_prod", "reportdate"], inplace=True)

assert len(df) == len(train) + len(val) + len(test)

# %%
means = train[x_cols + y_cols].mean().values.reshape(1, -1)
sds = train[x_cols + y_cols].std().values.reshape(1, -1)

# %%
train.to_parquet(product["train_df"])
val.to_parquet(product["val_df"])
test.to_parquet(product["test_df"])


# %%
def create_grouped_dataset(
    df, x_cols, y_cols, means, sds, sequence_length=12, sampling_rate=1
):
    delay = sequence_length * sampling_rate - 1
    grouped_dataset = None
    for idx, group in tqdm(df.groupby(level=0)):
        x_data = (group[x_cols + y_cols].values[:-delay] - means) / sds
        y_data = group[y_cols].values[delay:]

        dataset = timeseries_dataset_from_array(
            x_data,
            y_data,
            sequence_length=12,
            batch_size=2048 * 2,
        )
        if grouped_dataset is None:
            grouped_dataset = dataset
        else:
            grouped_dataset = grouped_dataset.concatenate(dataset)
    return grouped_dataset


# %%
train_dataset = create_grouped_dataset(train, x_cols, y_cols, means, sds)
train_dataset.save(product["train_dataset"])

# %%
val_dataset = create_grouped_dataset(val, x_cols, y_cols, means, sds)
val_dataset.save(product["val_dataset"])

# %%
test_dataset = create_grouped_dataset(test, x_cols, y_cols, means, sds)
test_dataset.save(product["test_dataset"])
