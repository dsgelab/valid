# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Directories and Logging                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import os
def make_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)

import logging
def init_logging(out_dir, log_file_name, logger, args):
    log_dir = out_dir + "logs/"
    make_dir(log_dir)
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + get_datetime() + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Time                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from datetime import datetime
def get_date():
    return datetime.today().strftime("%Y-%m-%d")
def get_datetime():
    return datetime.today().strftime("%Y-%m-%d-%H%M")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Printing                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def print_count(data):
    print("{:,} individuals with {:,} rows".format(data.FINNGENID.nunique(), data.shape[0]))
    
def logging_print(text):
    print(text)
    logging.info(text)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Reading                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def read_file(full_file_path):
    if full_file_path.endswith(".csv"):
        return pl.read_csv(full_file_path, try_parse_dates=True)
    elif full_file_path.endswith(".parquet"):
        return pl.read_parquet(full_file_path)
    else:
        raise ValueError("File type not supported: " + full_file_path)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Plotting                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""Plots trajectory of lab values for a specific individual. (x=DATE, y=VALUE)"""
def plot_data(data, fg_id):
    crnt_data = data.loc[data.FINNGENID == fg_id]
    crnt_data = crnt_data.sort_values("DATE", ascending=True)
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
        if col.endswith("_DATETIME"): df[col] = pd.to_datetime(df[col])
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