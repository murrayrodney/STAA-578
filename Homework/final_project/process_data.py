# %%
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.style.use("ggplot")

# %% tags=["parameters"]
upstream = None
product = None
resources_ = None

# %%
well_header = pd.read_csv(resources_["bh_locations"])
well_header.columns = well_header.columns.str.lower()

inj = pd.read_csv(resources_["well_injection"])
inj.columns = inj.columns.str.lower()

prod = pd.read_csv(resources_["well_production"])
prod.columns = prod.columns.str.lower().str.replace("prod", "")
prod["liquid"] = prod["oil"] + prod["water"]
prod["reportdate"] = pd.to_datetime(prod["reportdate"])
inj["reportdate"] = pd.to_datetime(inj["reportdate"])

prod = prod[prod["liquid"] > 0]

# Calculate producing rates from the volumes
streams = ["oil", "water", "liquid", "gas"]
for stream in streams:
    prod[f"{stream}_rate"] = prod[stream] / prod["days"]

# %%
inj_tots = inj.groupby(["well_name"], as_index=False)[["vol_liq", "vol_gas"]].sum()

prod_tots = prod.groupby(["well_name"])[["oil", "water", "liquid", "gas"]].sum()
prod_tots["start_date"] = prod.groupby(["well_name"])["reportdate"].min()
prod_tots["end_date"] = prod.groupby(["well_name"])["reportdate"].max()
prod_tots["years_prod"] = (
    prod_tots["end_date"] - prod_tots["start_date"]
) / pd.Timedelta(days=365.25)

# Filter to wells that have been on production for at least 2 years and have produced at least 1,000,000 bbls of oil
prod_tots = prod_tots[prod_tots["years_prod"] >= 2]
prod_tots = prod_tots[prod_tots["oil"] >= 1_000_000]

prod = prod[prod["well_name"].isin(prod_tots.index)]
prod = prod[~prod["well_name"].isin(inj["well_name"].unique())]
prod = prod[prod["liquid_rate"] < 30_000]
prod_tots = prod_tots[prod_tots.index.isin(prod["well_name"].unique())]

# %%
prod.describe()

# %%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
# sns.kdeplot(inj_tots['vol_liq'], ax=ax1, label='Total Liquid Injection', cut=0)
sns.kdeplot(prod_tots["oil"], ax=ax1, label="Total Oil Production", cut=0)
ax1.legend()

sns.kdeplot(prod_tots["years_prod"], ax=ax2, label="Total Years on Production", cut=0)
plt.show()

# %%
well_header["well_type"] = np.where(
    well_header["well_name"].isin(inj["well_name"]), "inj", "prod"
)
well_header["well_type"].value_counts()

inj_wells = set(inj["well_name"])
prod_wells = set(prod["well_name"])
all_wells = prod_wells.union(inj_wells)
well_header = well_header[well_header["well_name"].isin(all_wells)]

# %%
plt.figure(figsize=(15, 10))
sns.scatterplot(data=well_header, x="bh_long", y="bh_lat", hue="well_type", linewidth=0)
plt.show()


# %% [markdown]
# # Calculate distances betweein producers and injectors

# %%
well_header["field"] = "Prudhoe"
well_header = well_header[~well_header["bh_lat"].isna()]
geo_well_header = gpd.GeoDataFrame(
    well_header,
    geometry=gpd.points_from_xy(well_header["bh_long"], well_header["bh_lat"]),
    crs="epsg:4326",
)
# Get in NAD 83 Alaska Zone 4 in feet
geo_well_header = geo_well_header.to_crs("esri:102634")
prod_bh = geo_well_header[geo_well_header["well_name"].isin(prod["well_name"])]
inj_bh = geo_well_header[geo_well_header["well_name"].isin(inj["well_name"])]

# %%
prod_bh_buffer = prod_bh.copy()
prod_bh_buffer["geometry"] = prod_bh_buffer.buffer(5000)
comb_bh = gpd.sjoin(
    prod_bh_buffer, inj_bh, how="left", op="intersects", lsuffix="prod", rsuffix="inj"
)
comb_bh = comb_bh[["well_name_prod", "well_name_inj"]]

# %%
inj_counts = comb_bh.groupby("well_name_prod")["well_name_inj"].nunique()
inj_counts.describe()

# %%
# Make sure we have all the dates for production
start_date = prod.groupby("well_name")["reportdate"].min()
prod["start_date"] = prod["well_name"].map(start_date)

start = prod["reportdate"].min()
end = pd.Timestamp.now() + pd.Timedelta(31, "d")
dates = pd.date_range(start, end, freq="M") + pd.offsets.MonthBegin()
wells = set(prod["well_name"])
new_index = pd.MultiIndex.from_product([wells, dates])

prod.drop_duplicates(["well_name", "reportdate"], inplace=True)
full_prod = prod.set_index(["well_name", "reportdate"])
full_prod = full_prod.reindex(index=new_index).fillna(
    {
        "oil": 0,
        "water": 0,
        "gas": 0,
        "liquid": 0,
        "oil_rate": 0,
        "water_rate": 0,
        "gas_rate": 0,
        "liquid_rate": 0,
        "days": 0,
    }
)
full_prod.reset_index(names=["well_name", "reportdate"], inplace=True)
full_prod["start_date"] = full_prod.groupby("well_name")["start_date"].ffill()
full_prod["start_date"] = full_prod.groupby("well_name")["start_date"].bfill()
full_prod.isna().sum()

# %%
prod_inj = pd.merge(
    comb_bh, full_prod, how="left", left_on="well_name_prod", right_on="well_name"
).drop(columns=["well_name"])
prod_inj = pd.merge(
    prod_inj,
    inj,
    how="left",
    left_on=["well_name_inj", "reportdate"],
    right_on=["well_name", "reportdate"],
).drop(columns=["well_name"])
prod_inj.fillna({"vol_liq": 0, "vol_gas": 0}, inplace=True)
prod_inj.head()

# %%
aggs = {
    "oil": "mean",
    "water": "mean",
    "liquid": "mean",
    "gas": "mean",
    "oil_rate": "mean",
    "water_rate": "mean",
    "liquid_rate": "mean",
    "gas_rate": "mean",
    "vol_liq": "sum",
    "vol_gas": "sum",
    "well_name_inj": "count",
}
prod_summary = prod_inj.groupby(["well_name_prod", "reportdate"], as_index=False).agg(
    aggs
)
prod_summary["well_status"] = (prod_summary["liquid_rate"] > 0).astype(int)
prod_summary.head()

# %%
prod_summary.describe()

# %%
prod_summary.to_parquet(product["input_data"])
