def get_abnorm_func_based_on_name(lab_name,
                                  extra_choice=""):
    if lab_name == "tsh": 
        if extra_choice == "simple": return(tsh_abnorm_simple)
        if extra_choice == "multi": return(tsh_abnorm_complex)
    if lab_name == "hba1c": 
        if extra_choice == "soft": return(hba1c_mild_abnorm)
        if extra_choice == "strong": return(hba1c_strong_abnorm)
    if lab_name == "ldl": return(ldl_abnorm)
    if lab_name == "ana": return(ana_abnorm)
    if lab_name == "uacr": return(uacr_abnorm)
    if lab_name == "leuk": return(leuk_abnorm)
    if lab_name == "egfr" or lab_name == "krea": 
        if extra_choice == "herold-full":
            return(lambda x, y: egfr_herold(x,y, strict=True))
        if extra_choice == "herold-part":
            return(lambda x, y: egfr_herold(x,y, strict=False))
        if extra_choice == "KDIGO-strict":
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=True))
        if extra_choice == "KDIGO-soft":
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=False))
        if extra_choice == "KDIGO-soft2":
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=False, soft_type=2))
        else:
            return(lambda x, y: egfr_kdigo_abnorm(x, y, strict=True))
    if lab_name == "cyst" or lab_name =="cystc": return(cystc_abnorm)
    if lab_name == "gluc" or lab_name=="fgluc": return(gluc_abnorm)
    if lab_name == "alat": return(alat_abnorm)
    if lab_name == "asat": return(asat_abnorm)
    if lab_name == "t4": return(t4_abnorm)
    if lab_name == "t3": return(t3_abnorm)
    if lab_name == "ftri" or lab_name == "tri": return(tri_abnorm)

    else: raise ValueError("Sorry, no function for this lab name.")
     
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 eGFR                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from importlib.resources import simple
import polars as pl
def egfr_kdigo_abnorm(data, 
                      value_col_name="VALUE_TRANSFORM",
                      strict=False,
                      soft_type=1):
    if strict:
        # simple binary cut-off at 60
        data = data.with_columns(
                            pl.when(pl.col(value_col_name)<60)
                              .then(1)
                              .otherwise(0).alias("ABNORM_CUSTOM")
        )    
    else:
        if soft_type == 1:
            # Add grey zone between 60 and 65
            data = data.with_columns(
                    pl.when(pl.col(value_col_name) <60).then(1)
                    .when((pl.col(value_col_name) <=65)&(pl.col(value_col_name)>=60)).then(0.5)
                    .otherwise(0).alias("ABNORM_CUSTOM")
                )   
        elif soft_type == 2:
            # Add grey zone between 60 and 70
            print(data)
            data = data.with_columns(
                    pl.when(pl.col(value_col_name) <60).then(1)
                    .when((pl.col(value_col_name) <=70)&(pl.col(value_col_name)>=60)).then(0.5)
                    .otherwise(0).alias("ABNORM_CUSTOM")
                )   
            print(data)
        print(data["ABNORM_CUSTOM"].value_counts())
    return(data)


"""Based on transformed kreatinine to eGFR. Herold et al. 2024 age-specific cut-offs"""
import polars as pl
def egfr_herold(data, value_col_name="VALUE_TRANSFORM", strict=False):
    if strict:
        return(data.with_columns(
            pl.when((pl.col.EVENT_AGE<40)&(pl.col(value_col_name)<60)).then(1)
            .when((pl.col.EVENT_AGE>=30)&(pl.col.EVENT_AGE<40)&(pl.col(value_col_name)<75)).then(1)
            .when((pl.col.EVENT_AGE>=40)&(pl.col.EVENT_AGE<50)&(pl.col(value_col_name)<70)).then(1)
            .when((pl.col.EVENT_AGE>=50)&(pl.col.EVENT_AGE<60)&(pl.col(value_col_name)<60)).then(1)
            .when((pl.col.EVENT_AGE>=60)&(pl.col.EVENT_AGE<70)&(pl.col(value_col_name)<50)).then(1)
            .when((pl.col.EVENT_AGE>=70)&(pl.col.EVENT_AGE<80)&(pl.col(value_col_name)<40)).then(1)
            .when((pl.col.EVENT_AGE>=80)&(pl.col(value_col_name)<35)).then(1)
            .otherwise(0)
            .alias("ABNORM_CUSTOM")
        ))
    else:
        return(data.with_columns(
            pl.when((pl.col.EVENT_AGE.floor()<60)&(pl.col(value_col_name)<60)).then(1)
            .when((pl.col.EVENT_AGE.floor()>=60)&(pl.col(value_col_name)<50)).then(1)
            .otherwise(0)
            .alias("ABNORM_CUSTOM")
        ))
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 eGFR                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
import polars as pl
"""Abnormality for cystatin c"""
def cystc_abnorm(data, value_col_name):
    data = data.to_pandas()

    data.loc[data[value_col_name] <= 1.2,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 1.2,"ABNORM_CUSTOM"] = 1
    data.loc[np.logical_and(data[value_col_name] > 1, data.EVENT_AGE<=50),"ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Leukocytes                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def leuk_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)<3.4)
                .then(-1)
                .when(pl.col(value_col_name)>8.2)
                .then(1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    print(data)
    print(data["ABNORM_CUSTOM"].value_counts())
    return(data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 TSH                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def tsh_abnorm_complex(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)>=4)
                .then(1)
                .when(pl.col(value_col_name)<0.4)
                .then(-1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    print(data["ABNORM_CUSTOM"].value_counts())
    return(data)
import polars as pl
def tsh_abnorm_simple(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)>4)
                .then(1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    return(data)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Triglycerides?                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def tri_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)>2)
                .then(1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    return(data)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 UACR                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def uacr_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)<3)
                .then(pl.lit(0))
                .when((pl.col(value_col_name)>3)&(pl.col(value_col_name)<=30))
                .then(pl.lit(1))
                .when((pl.col(value_col_name)>30)&(pl.col(value_col_name)<=220))
                .then(pl.lit(2))
                .otherwise(pl.lit(3))
        .alias("ABNORM_CUSTOM")
    )
    print(data["ABNORM_CUSTOM"].value_counts())
    return(data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 HbA1c                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
"""Individual ABNORMity >42mmol/l also theoretically should be at least 20 but this does not have a 
   diagnostic value aparently"""
def hba1c_mild_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[data[value_col_name] > 42,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] <= 42,"ABNORM_CUSTOM"] = 0
    return(pl.DataFrame(data))

import polars as pl
def hba1c_strong_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[data[value_col_name] > 47,"ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] <= 47,"ABNORM_CUSTOM"] = 0
    return(pl.DataFrame(data))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Glucose                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def gluc_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] >= 7, "ABNORM_CUSTOM"] = 1
    data.loc[data[value_col_name] < 4, "ABNORM_CUSTOM"] = -1
    return(pl.DataFrame(data))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 LDL                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def ldl_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(pl.when(pl.col(value_col_name)<4.0).then(0).otherwise(1).alias("ABNORM_CUSTOM"))
    return(data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Thyroid                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def t4_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)>23)
                .then(1)
                .when(pl.col(value_col_name)<=9)
                .then(-1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    print(data["ABNORM_CUSTOM"].value_counts())
    return(data)
    
import polars as pl
def t3_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(
                pl.when(pl.col(value_col_name)>6.3)
                .then(1)
                .otherwise(0)
        .alias("ABNORM_CUSTOM")
    )
    print(data["ABNORM_CUSTOM"].value_counts())
    return(data)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 ANA                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def ana_abnorm(data, value_col_name="VALUE"):
    data = data.with_columns(pl.when(pl.col(value_col_name)>400).then(1).otherwise(0).alias("ABNORM_CUSTOM"))
    return(pl.DataFrame(data))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Liver                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
"""Based on what andrea sent."""
def alat_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 45, "ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))

import polars as pl
"""Based on what andrea sent."""
def asat_abnorm(data, value_col_name="VALUE"):
    data = data.to_pandas()

    data.loc[:,"ABNORM_CUSTOM"] = 0
    data.loc[data[value_col_name] > 35, "ABNORM_CUSTOM"] = 1
    return(pl.DataFrame(data))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Standard                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
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
