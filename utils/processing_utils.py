import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_abnorm_func_based_on_name(lab_name):
    if lab_name == "tsh": return(tsh_abnorm)
    if lab_name == "hba1c": return(hba1c_abnorm)
    if lab_name == "ldl": return(ldl_abnorm)
    if lab_name == "egfr" or lab_name == "krea": return(egfr_abnorm)
    if lab_name == "cyst": return(cystc_abnorm)
    if lab_name == "gluc" or lab_name=="fgluc": return(gluc_abnorm)
    else: raise("Sorry, no function for this lab name.")

"""Individual ABNORMity with grey area 2.5-4"""
def tsh_abnorm(data, value_col_name="VALUE"):
    data.loc[data[value_col_name] < 4,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 2.5,"ABNORM_CUSTOM"] = 0.5
    data.loc[data[value_col_name] >= 4,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] < 0.4,"ABNORM_CUSTOM"] = -1
    return(data)

def gluc_abnorm(data, value_col_name="VALUE"):
    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 6, "ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] < 4, "ABNORM_CUSTOM"] = -1
    return(data)

"""Abnormality for cystatin c"""
def cystc_abnorm(data, value_col_name):
    data.loc[data[value_col_name] <= 1.2,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 1.2,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] > 1, data.EVENT_AGE<=50),"ABNORM_CUSTOM"] = 1
    return(data)   
    
"""Individual ABNORMity >42mmol/l also theoretically should be at least 20 but this does not have a 
   diagnostic value aparently"""
def hba1c_abnorm(data, value_col_name="VALUE"):
    data.loc[data[value_col_name] > 42,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] > 47,"ABNORM_CUSTOM"] = 2
    data.loc[data[value_col_name] <= 42,"ABNORM_CUSTOM"] = 0
    return(data)

"""Based on FinnGen ABNORMity column have only normal and ABNORM."""
def simple_abnorm(data):
    data.loc[np.logical_and(data.ABNORM != "N", data.ABNORM.notnull()), "ABNORM"] = 1
    data.loc[data.ABNORM == "N","ABNORM"] = 0
    return(data)
    
"""Based on FinnGen ABNORMity column have normal (0), high (-1), and low (1).""" 
def three_level_abnorm(data):
    data.loc[data.ABNORM == "N","ABNORM"] = 0
    data.loc[data.ABNORM == "L","ABNORM"] = -1
    data.loc[data.ABNORM == "H","ABNORM"] = 1
    return(data)

def ldl_abnorm(data):
    data.loc[data.VALUE <= 5.3,"ABNORM_CUSTOM"] = 0
    data.loc[data.VALUE > 5.3,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data.EVENT_AGE <= 49, data.VALUE > 4.7),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data.EVENT_AGE <= 29, data.VALUE > 4.3),"ABNORM_CUSTOM"] = 1

    return(data)   

    return(data)

"""Based on transformed kreatinine to eGFR."""
def egfr_abnorm(data, value_col_name="VALUE_TRANSFORM"):
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
    return(data)

def add_measure_counts(data):
    data.DATE = data.DATE.astype("datetime64[ns]")
    n_measures = data.groupby("FINNGENID").agg({"DATE": [("N_YEAR", lambda x: len(set(x.dt.year)))], "VALUE": len}).reset_index()
    n_measures.columns = ["FINNGENID", "N_YEAR", "N_MEASURE"]
    data = pd.merge(data, n_measures, on="FINNGENID", how="left")
    return(data)

def add_set(unique_data, test_pct=0.1, valid_pct=0.1):
    """Adds SET column to data based on random split of individuals.
       Data passed must be unique data with only one row per individual."""
    data_train, data_rest = train_test_split(unique_data, shuffle=True, random_state=3291, test_size=(valid_pct+test_pct), train_size=1-(valid_pct+test_pct), stratify=unique_data.y_DIAG)
    print(f"N rows {len(data_train)}   N indvs {len(set(data_train.FINNGENID))}  N cases {sum(data_train.y_DIAG)} pct cases {round(sum(data_train.y_DIAG)/len(data_train), 2)}")
    print(f"N rows {len(data_rest)}   N indvs {len(set(data_rest.FINNGENID))}  N cases {sum(data_rest.y_DIAG)} pct cases {round(sum(data_rest.y_DIAG)/len(data_rest), 2)}")

    data_valid, data_test = train_test_split(data_rest, shuffle=True, random_state=391, test_size=test_pct/(test_pct+valid_pct), train_size=valid_pct/(test_pct+valid_pct), stratify=data_rest.y_DIAG)
    print(f"N rows {len(data_valid)}   N indvs {len(set(data_valid.FINNGENID))}  N cases {sum(data_valid.y_DIAG)} pct cases {round(sum(data_valid.y_DIAG)/len(data_valid), 2)}")
    print(f"N rows {len(data_test)}   N indvs {len(set(data_test.FINNGENID))}  N cases {sum(data_test.y_DIAG)} pct cases {round(sum(data_test.y_DIAG)/len(data_test), 2)}")

    unique_data.loc[unique_data.FINNGENID.isin(data_train.FINNGENID),"SET"] = 0
    unique_data.loc[unique_data.FINNGENID.isin(data_valid.FINNGENID),"SET"] = 1
    unique_data.loc[unique_data.FINNGENID.isin(data_test.FINNGENID),"SET"] = 2
    print(unique_data.SET.value_counts(dropna=False))
    return(unique_data)
