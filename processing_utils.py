import numpy as np

def get_abnorm_func_based_on_name(lab_name):
    if lab_name == "tsh": return(tsh_abnorm)
    if lab_name == "hba1c": return(hba1c_abnorm)
    if lab_name == "ldl": return(ldl_abnorm)
    if lab_name == "egfr" or lab_name == "krea": return(egfr_abnorm)
    if lab_name == "cyst": return(cystc_abnorm)
    if lab_name == "gluc" or lab_name=="fgluc": return(gluc_abnorm)
    else: raise("Sorry, no function for this lab name.")

def get_abnorm_func_based_on_name(lab_name):
    if lab_name == "tsh": return(tsh_abnorm)
    if lab_name == "hba1c": return(hba1c_abnorm)
    if lab_name == "ldl": return(ldl_abnorm)
    if lab_name == "egfr" or lab_name == "krea": return(egfr_abnorm)
    if lab_name == "cyst": return(cystc_abnorm)
    if lab_name == "gluc" or lab_name=="fgluc": return(gluc_abnorm)
    else: raise("Sorry, no function for this lab name.")


"""Individual ABNORMity with grey area 2.5-4"""
def tsh_abnorm(data):
    data.loc[data.VALUE < 4,"ABNORM_CUSTOM"] = 0
    data.loc[data.VALUE > 2.5,"ABNORM_CUSTOM"] = 0.5
    data.loc[data.VALUE >= 4,"ABNORM_CUSTOM"] = 1
    data.loc[data.VALUE < 0.4,"ABNORM_CUSTOM"] = -1
    return(data)

def gluc_abnorm(data):
    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data.VALUE > 6, "ABNORM_CUSTOM"] = 1
    data.loc[data.VALUE < 4, "ABNORM_CUSTOM"] = -1
    return(data)

"""Abnormality for cystatin c"""
def cystc_abnorm(data):
    data.loc[data.VALUE <= 1.2,"ABNORM_CUSTOM"] = 0
    data.loc[data.VALUE > 1.2,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data.VALUE > 1, data.EVENT_AGE<=50),"ABNORM_CUSTOM"] = 1
    return(data)   
    
"""Individual ABNORMity >42mmol/l also theoretically should be at least 20 but this does not have a 
   diagnostic value aparently"""
def hba1c_abnorm(data):
    data.loc[data.VALUE > 42,"ABNORM_CUSTOM"] = 1
    data.loc[data.VALUE <= 42,"ABNORM_CUSTOM"] = 0
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
def egfr_abnorm(data, valu_col_name="VALUE_TRANSFORM"):
    # 0-39 -> >88
    data.loc[np.logical_and(data[valu_col_name] < 89, round(data.EVENT_AGE) <=39),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[valu_col_name] >= 89, round(data.EVENT_AGE) <=39),"ABNORM_CUSTOM"] = 0
    # 40-49 -> >82
    data.loc[np.logical_and(data[valu_col_name] < 83, np.logical_and(round(data.EVENT_AGE) >= 40, round(data.EVENT_AGE) <=49)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[valu_col_name] >= 83, np.logical_and(round(data.EVENT_AGE) >= 40, round(data.EVENT_AGE) <=49)),"ABNORM_CUSTOM"] = 0
    # 50-59 -> >76
    data.loc[np.logical_and(data[valu_col_name] < 77, np.logical_and(round(data.EVENT_AGE) >= 50, round(data.EVENT_AGE) <=59)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[valu_col_name] >= 77, np.logical_and(round(data.EVENT_AGE) >= 50, round(data.EVENT_AGE) <=59)),"ABNORM_CUSTOM"] = 0
    # 60-69 -> >68
    data.loc[np.logical_and(data[valu_col_name] < 69, np.logical_and(round(data.EVENT_AGE) >= 60, round(data.EVENT_AGE) <=69)),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[valu_col_name] >= 69, np.logical_and(round(data.EVENT_AGE) >= 60, round(data.EVENT_AGE) <=69)),"ABNORM_CUSTOM"] = 0
    # 59-70 -> >58
    data.loc[np.logical_and(data[valu_col_name] < 59, round(data.EVENT_AGE) >= 70),"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[valu_col_name] >= 59, round(data.EVENT_AGE) >= 70),"ABNORM_CUSTOM"] = 0
    return(data)
