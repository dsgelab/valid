import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import polars as pl


# Function to generate a random date for each year
from datetime import datetime
import calendar
def generate_random_date(year):
    # Randomly select a month (1 to 12)
    month = np.random.randint(1, 13)
        
    # Get the number of days in the selected month (handling leap years for February)
    days_in_month = calendar.monthrange(year, month)[1]
        
    # Randomly select a day within that month
    day = np.random.randint(1, days_in_month + 1)
        
    # Return the sampled random date
    return datetime(year, month, day).date()
    
def egfr_transform(data):
    data = data.to_pandas()
    data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-0.329))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "VALUE"]/61.9 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 62, data["SEX"] == "female"), "EVENT_AGE"])*144
    data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-0.411))*(0.993**data.loc[np.logical_and(data["VALUE"] <= 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE_TRANSFORM"]=((data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "VALUE"]/79.6 )**(-1.209))*(0.993**data.loc[np.logical_and(data["VALUE"] > 80, data["SEX"] == "male"), "EVENT_AGE"])*141
    data = data.assign(VALUE = data["VALUE_TRANSFORM"])
    data = data.drop("VALUE_TRANSFORM", axis=1)
    
    return(pl.DataFrame(data))
    
    
def get_abnorm_func_based_on_name(lab_name,
                                  extra_choice=""):
    if lab_name == "tsh": return(tsh_abnorm)
    if lab_name == "hba1c": return(hba1c_abnorm)
    if lab_name == "ldl": return(ldl_abnorm)
    if lab_name == "egfr" or lab_name == "krea": 
        if extra_choice == "age":
            return(egfr_transform)
        if extra_choice == "KDIGO-strict":
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=True))
        if extra_choice == "KDIGO-soft":
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=False))
        else:
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=True))
    if lab_name == "cyst" or lab_name =="cystc": return(cystc_abnorm)
    if lab_name == "gluc" or lab_name=="fgluc": return(gluc_abnorm)
    if lab_name == "alat": return(alat_abnorm)
    if lab_name == "asat": return(asat_abnorm)

    else: raise ValueError("Sorry, no function for this lab name.")
        
"""Based on transformed kreatinine to eGFR."""
def egfr_kdigo_abnorm(data, 
                      value_col_name="VALUE_TRANSFORM",
                      strict=False):
    if strict:
        data = data.with_columns(
            pl.when(pl.col(value_col_name) <60).then(1)
            .otherwise(0).alias("ABNORM_CUSTOM")
        )    
    else:
        data = data.with_columns(
                pl.when(pl.col(value_col_name) <60).then(1)
                .when((pl.col(value_col_name) <=65)&(pl.col(value_col_name)>=60)).then(0.5)
                .otherwise(0).alias("ABNORM_CUSTOM")
            )   
        print(data["ABNORM_CUSTOM"].value_counts())
    return(data)

"""Individual ABNORMity with grey area 2.5-4"""
def tsh_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[data[value_col_name] < 4,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 2.5,"ABNORM_CUSTOM"] = 0.5
    data.loc[data[value_col_name] >= 4,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] < 0.4,"ABNORM_CUSTOM"] = -1
    return(pl.DataFrame(data))

def gluc_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] >= 7, "ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] < 4, "ABNORM_CUSTOM"] = -1
    return(pl.DataFrame(data))

"""Based on what andrea sent."""
def alat_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 45, "ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))

"""Based on what andrea sent."""
def asat_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 35, "ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))

"""Abnormality for cystatin c"""
def cystc_abnorm(data, value_col_name):
    data = data.to_pandas()

    data.loc[data[value_col_name] <= 1.2,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 1.2,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] > 1, data.EVENT_AGE<=50),"ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))
    
"""Individual ABNORMity >42mmol/l also theoretically should be at least 20 but this does not have a 
   diagnostic value aparently"""
def hba1c_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[data[value_col_name] > 42,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] > 47,"ABNORM_CUSTOM"] = 2
    data.loc[data[value_col_name] <= 42,"ABNORM_CUSTOM"] = 0
    return(pl.DataFrame(data))

"""Based on FinnGen ABNORMity column have only normal and ABNORM."""
def simple_abnorm(data):
    data = data.to_pandas()

    data.loc[np.logical_and(data.ABNORM != "N", data.ABNORM.notnull()), "ABNORM"] = 1
    data.loc[data.ABNORM == "N","ABNORM"] = 0
    return(pl.DataFrame(data))
    
"""Based on FinnGen ABNORMity column have normal (0), high (-1), and low (1).""" 
def three_level_abnorm(data):
    data = data.to_pandas()

    data.loc[data.ABNORM == "N","ABNORM"] = 0
    data.loc[data.ABNORM == "L","ABNORM"] = -1
    data.loc[data.ABNORM == "H","ABNORM"] = 1
    return(pl.DataFrame(data))

def ldl_abnorm(data):
    data = data.to_pandas()

    data.loc[data.VALUE <= 5.3,"ABNORM_CUSTOM"] = 0
    data.loc[data.VALUE > 5.3,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data.EVENT_AGE <= 49, data.VALUE > 4.7),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data.EVENT_AGE <= 29, data.VALUE > 4.3),"ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))

"""Based on transformed kreatinine to eGFR."""
def egfr_abnorm(data, value_col_name="VALUE_TRANSFORM"):
    data = data.to_pandas()

    # 0-39 -> >88
    data.loc[np.logical_and(data[value_col_name] < 89, round(data.EVENT_AGE) <=39),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] >= 89, round(data.EVENT_AGE) <=39),"ABNORM_CUSTOM"] = 0
    # 40-49 -> >82
    data.loc[np.logical_and(data[value_col_name] < 83, np.logical_and(round(data.EVENT_AGE) >= 40, round(data.EVENT_AGE) <=49)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] >= 83, np.logical_and(round(data.EVENT_AGE) >= 40, round(data.EVENT_AGE) <=49)),"ABNORM_CUSTOM"] = 0
    # 50-59 -> >76
    data.loc[np.logical_and(data[value_col_name] < 77, np.logical_and(round(data.EVENT_AGE) >= 50, round(data.EVENT_AGE) <=59)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] >= 77, np.logical_and(round(data.EVENT_AGE) >= 50, round(data.EVENT_AGE) <=59)),"ABNORM_CUSTOM"] = 0
    # 60-69 -> >68
    data.loc[np.logical_and(data[value_col_name] < 69, np.logical_and(round(data.EVENT_AGE) >= 60, round(data.EVENT_AGE) <=69)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] >= 69, np.logical_and(round(data.EVENT_AGE) >= 60, round(data.EVENT_AGE) <=69)),"ABNORM_CUSTOM"] = 0
    # 70+ -> >58
    data.loc[np.logical_and(data[value_col_name] < 59, round(data.EVENT_AGE) >= 70),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] >= 59, round(data.EVENT_AGE) >= 70),"ABNORM_CUSTOM"] = 0
    return(pl.DataFrame(data))

def add_measure_counts(data):
    data = data.to_pandas()

    data.DATE = data.DATE.astype("datetime64[ns]")
    n_measures = data.groupby("FINNGENID").agg({"DATE": [("N_YEAR", lambda x: len(set(x.dt.year)))], "VALUE": len}).reset_index()
    n_measures.columns = ["FINNGENID", "N_YEAR", "N_MEASURE"]
    data = pd.merge(data, n_measures, on="FINNGENID", how="left")
    return(pl.DataFrame(data))

def add_set(unique_data, 
            test_pct=0.1, 
            valid_pct=0.1):
    """Adds SET column to data based on random split of individuals.
       Data passed must be unique data with only one row per individual."""
    if test_pct > 0:
        data_train, data_rest = train_test_split(unique_data, 
                                                 shuffle=True, 
                                                 random_state=3291, 
                                                 test_size=round((valid_pct+test_pct),2), 
                                                 train_size=round(1-(valid_pct+test_pct),2), 
                                                 stratify=unique_data["y_DIAG"])
        new_test_size = round(test_pct/(test_pct+valid_pct),2)
        new_train_size = round(valid_pct/(test_pct+valid_pct),2)
    else:
        data_rest = unique_data
        new_test_size = valid_pct
        new_train_size = 1-valid_pct
    data_valid, data_test = train_test_split(data_rest, 
                                             shuffle=True, 
                                             random_state=391, 
                                             test_size=new_test_size, 
                                             train_size=new_train_size, 
                                             stratify=data_rest["y_DIAG"])
    if test_pct == 0:
        unique_data = unique_data.with_columns(
                            SET=pl.when(pl.col("FINNGENID").is_in(data_valid["FINNGENID"])).then(0)
                                   .when(pl.col("FINNGENID").is_in(data_test["FINNGENID"])).then(1)
                                   .otherwise(None)
    )
        print(f"N rows train {len(data_valid)}   N indvs train {len(set(data_valid["FINNGENID"]))}  N cases train {sum(data_valid["y_DIAG"])} pct cases {round(sum(data_valid["y_DIAG"])/len(data_valid), 2)}")

    else:
        unique_data = unique_data.with_columns(
            SET=pl.when(pl.col("FINNGENID").is_in(data_train["FINNGENID"])).then(0)
               .when(pl.col("FINNGENID").is_in(data_valid["FINNGENID"])).then(1)
               .when(pl.col("FINNGENID").is_in(data_test["FINNGENID"])).then(2)
               .otherwise(None)
        )
        print(f"N rows train {len(data_train)}   N indvs train {len(set(data_train["FINNGENID"]))}  N cases train {sum(data_train["y_DIAG"])} pct cases {round(sum(data_train["y_DIAG"])/len(data_train), 2)}")

    print(unique_data.select(pl.col("SET")).to_series().value_counts())
    return(pl.DataFrame(unique_data))
