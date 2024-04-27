# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.style.use("ggplot")

# %% tags=["parameters"]
upstream = None
product = None
resources_ = None

# %%
bh_loc = pd.read_csv(resources_["bh_locations"])
inj = pd.read_csv(resources_["well_injection"])
prod = pd.read_csv(resources_["well_production"])
