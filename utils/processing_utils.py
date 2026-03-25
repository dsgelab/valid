# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 File reading                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from datetime import datetime
import calendar
import numpy as np
def generate_random_date(year):
    # Randomly select a month (1 to 12)
    month = np.random.randint(1, 13)
        
    # Get the number of days in the selected month (handling leap years for February)
    days_in_month = calendar.monthrange(year, month)[1]
        
    # Randomly select a day within that month
    day = np.random.randint(1, days_in_month + 1)
        
    # Return the sampled random date
    return datetime(year, month, day).date()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Transforms                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def egfr_ckdepi2021_transform(data,
                              value_col_name="VALUE"):
    fem_k = 0.7
    male_k = 0.9
    fem_alpha = -0.241
    male_alpha = -0.302
    data = data.with_columns((pl.col(value_col_name)/88.4).alias(value_col_name))
    data=data.with_columns(
        pl.when(pl.col.SEX=="male")
        .then(142*(pl.min_horizontal(pl.col(value_col_name)/male_k, 1)**male_alpha)*(pl.max_horizontal(pl.col(value_col_name)/male_k, 1)**(-1.2))*(0.9938**pl.col.EVENT_AGE))
        .otherwise(142*(pl.min_horizontal(pl.col(value_col_name)/fem_k, 1)**fem_alpha)*(pl.max_horizontal(pl.col(value_col_name)/male_k, 1)**(-1.2))*(0.9938**pl.col.EVENT_AGE)*1.012)
        .alias(value_col_name)
    )
    data = data.with_columns(pl.Series("UNIT", ["ml/min/1.73m2"]*data.height))

    return(data)
    
import polars as pl
def cystc_ckdepi2012_transform(data,
                              value_col_name="VALUE"):
    data=data.with_columns(
        pl.when(pl.col.SEX=="male")
        .then(133*(pl.min_horizontal(pl.col(value_col_name)/0.8, 1)**(-0.499))*(pl.max_horizontal(pl.col(value_col_name)/0.8, 1)**(-1.328))*(0.996**pl.col.EVENT_AGE))
        .otherwise(133*(pl.min_horizontal(pl.col(value_col_name)/0.8, 1)**(-0.499))*(pl.max_horizontal(pl.col(value_col_name)/0.8, 1)**(-1.328))*(0.996**pl.col.EVENT_AGE)*0.932)
        .alias(value_col_name)
    )
    
    return(data)
    