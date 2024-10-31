import pandas as pd
import numpy as np

import os
def make_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)

def print_count(data):
    print("{:,} individuals with {:,} rows".format(data.FINNGENID.nunique(), data.shape[0]))

import matplotlib.pyplot as plt
import seaborn as sns
def plot_data(data, fg_id):
    crnt_data = data.loc[data.FINNGENID == fg_id]
    fig, ax = plt.subplots()
    if "ABNORM_CUSTOM" in data.columns:
        sns.scatterplot(crnt_data, x="DATE", y="VALUE", hue="ABNORM_CUSTOM", ax=ax)
    elif "ABNORM" in data.columns:
        sns.scatterplot(crnt_data, x="DATE", y="VALUE", hue="ABNORM", ax=ax)
    sns.lineplot(crnt_data, x="DATE", y="VALUE", ax=ax)
    if "DIAG_DATE" in data.columns:
        ax.vlines(x=crnt_data.DIAG_DATE.max(), 
                  ymin=crnt_data.VALUE.min(),
                  ymax=crnt_data.VALUE.max(),
                  color="black")
    display(crnt_data)

try:
    from google.cloud import bigquery
except ImportError:
    pass
# From Mathias for FinnGen BigQuery
def query_to_df(query, **job_config_kwargs):
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(**job_config_kwargs)
    query_result = client.query(query, job_config=job_config)
    df = pd.DataFrame([dict(row) for row in query_result])
    df = parse_dates(df)
    return(df)
def parse_dates(df):
    for col in df.columns:
        if cols.endswith("_DATETIME"): df[col] = pd.to_datetime(df[col])
    return df

# https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python
import time
class Timer:
    def __init__(self):
        self.start_time = time.time()
    def restart(self):
        self.start_time = time.time()
    def get_elapsed(self):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        m, s = divmod(elapsed_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d.%02d" % (h, m, s)
        return time_str