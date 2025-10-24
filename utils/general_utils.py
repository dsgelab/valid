# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Directories and Logging                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import os
def make_dir(dir_path):
    if not os.path.exists(dir_path): os.makedirs(dir_path)

import logging
def init_logging(out_dir, 
                 lab_name, 
                 logger, 
                 args):
    log_dir = out_dir + "logs/" + lab_name + "/"
    make_dir(log_dir)
    logging.basicConfig(filename=log_dir+get_datetime()+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
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
    print("{:,} individuals with {:,} rows".format(data["FINNGENID"].unique().len(), data.shape[0]))
    
def logging_print(text):
    print(text)
    logging.info(text)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Reading                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def read_file(full_file_path,
              schema=None) -> pl.DataFrame:
    if full_file_path.endswith(".csv"):
        if not schema:
            return pl.read_csv(full_file_path, 
                               infer_schema_length=100000,
                               try_parse_dates=True,
                               null_values="NA")
        else:
            return pl.read_csv(full_file_path, 
                               infer_schema_length=100000,
                               schema_overrides=schema,
                               null_values="NA")
    elif full_file_path.endswith(".parquet"):
        return pl.read_parquet(full_file_path)
    else:
        raise ValueError("File type not supported: " + full_file_path)


### Going through all years function created with help of Claude AI
import os
from datetime import datetime, timedelta

def get_all_dates_in_year(year=2025):
    """Generate all dates in the given year as YYYY-MM-DD strings"""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return dates

def check_dated_path(file_path_start,
                     year=2025):
    dates = get_all_dates_in_year(year=year)
    for date in reversed(dates):
        if os.path.exists(file_path_start+date+"/preds_"+date+".parquet"):
            return True
    return False

def get_dated_path(file_path_start,
                     year=2025):
    dates = get_all_dates_in_year(year=year)
    for date in reversed(dates):
        if os.path.exists(file_path_start+date+"/preds_"+date+".parquet"):
            return file_path_start+date+"/preds_"+date+".parquet"
    return None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Plotting                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
"""Plots trajectory of lab values for a specific individual. (x=DATE, y=VALUE)"""
def plot_data(data, fg_id):
    data = data.to_pandas()
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
def query_to_df(query, **job_config_kwargs) -> pl.DataFrame:
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(**job_config_kwargs)
    query_result = client.query(query, job_config=job_config)
    df = pl.DataFrame([dict(row) for row in query_result], infer_schema_length=10000)
    df = parse_dates(df)
    return(df)

def parse_dates(df: pl.DataFrame) -> pl.DataFrame:
    for col in df.columns:
        if col.endswith("_DATETIME"):
            df = df.with_column(pl.col(col).str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S"))
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