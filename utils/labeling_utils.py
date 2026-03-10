from sklearn.model_selection import train_test_split
import polars as pl


def add_set(unique_data, 
            valid_pct=0.1,
            finetune_valid_pct=0.1):
    """Adds SET column to data based on random split of individuals.
       Data passed must be unique data with only one row per individual."""
    total_valid_pct = valid_pct + finetune_valid_pct

    data_train, data_total_valid = train_test_split(unique_data,
                                                    shuffle=True, 
                                                    random_state=0, 
                                                    test_size=total_valid_pct, 
                                                    train_size=1-total_valid_pct, 
                                                    stratify=unique_data["y_MEAN_ABNORM", "y_NEXT_ABNORM", "y_MIN_ABNORM"])
    
    if finetune_valid_pct > 0:
        data_valid, data_finetune_valid = train_test_split(data_total_valid,
                                                           shuffle=True, 
                                                           random_state=0, 
                                                           test_size=round(finetune_valid_pct/total_valid_pct,2), 
                                                           train_size=round(valid_pct/total_valid_pct,2), 
                                                           stratify=data_total_valid["y_MEAN_ABNORM", "y_NEXT_ABNORM", "y_MIN_ABNORM"])
    else:
        data_valid = data_total_valid
        data_finetune_valid = pl.DataFrame({"FINNGENID": []})  # Empty DataFrame

    unique_data = unique_data.with_columns(
                                pl.when(pl.col("FINNGENID").is_in(data_train["FINNGENID"])).then(0)
                                  .when(pl.col("FINNGENID").is_in(data_valid["FINNGENID"])).then(1)
                                  .when(pl.col("FINNGENID").is_in(data_finetune_valid["FINNGENID"])).then(0.5)
                                  .otherwise(None)
                                  .cast(pl.Float64)
                                  .alias("SET")
        )
    print(unique_data.select(pl.col("SET")).to_series().value_counts())
    print(unique_data.select(pl.col("SET")).to_series().value_counts(normalize=True))

    return(unique_data)

import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import logging_print
import polars as pl
def log_print_n(labels: pl.DataFrame,
                name: str):
    logging_print(name)
    logging_print(f"MEAN N indvs {labels.height}  N cases { labels.get_column('y_MEAN_ABNORM').sum()} pct cases {round( labels.get_column('y_MEAN_ABNORM').sum()/labels.height*100,2)}%")
    if "y_MIN" in labels.columns: logging_print(f"MIN N indvs {labels.height}  N cases { labels.get_column('y_MIN_ABNORM').sum()} pct cases {round( labels.get_column('y_MIN_ABNORM').sum()/labels.height*100,2)}%")
    if "y_NEXT" in labels.columns: logging_print(f"NEXT N indvs {labels.height}  N cases { labels.get_column('y_NEXT_ABNORM').sum()} pct cases {round( labels.get_column('y_NEXT_ABNORM').sum()/labels.height*100,2)}%")

import polars as pl
from datetime import datetime
def add_ages(data: pl.DataFrame,
             date: pl.Date,
             fg_ver) -> pl.DataFrame:
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_1.0.txt.gz"

    ages = pl.read_csv(minimum_file_name,
                       separator="\t",
                       columns=["FINNGENID", "APPROX_BIRTH_DATE"])
    data = (data
              .join(ages, on="FINNGENID", how="left")
              .with_columns(pl.col.APPROX_BIRTH_DATE.cast(pl.Utf8).str.to_date())
    )
    data = data.with_columns(
            ((date-pl.col.APPROX_BIRTH_DATE).dt.total_days()/365.25).floor()
            .alias("EVENT_AGE")
    )
    return(data.drop("APPROX_BIRTH_DATE"))

import polars as pl
def get_bbs_indvs(fg_ver="R13",
                   bbs=["HELSINKI BIOBANK"]) -> pl.Series:
    """Selecting individuals in FinnGen that are in Helsinki Biobank, not dead before end date."""

    # Read in the minimum data file
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_extended_1.0.txt.gz"
    min_data = pl.read_csv(minimum_file_name, 
                           separator="\t",
                           columns=["FINNGENID",  "COHORT"])
    # Filtering
    select_fids = (min_data
                   .filter(
                       # In Helsinki Biobank
                       (pl.col.COHORT.is_in(bbs))
                   )
                   .get_column("FINNGENID")
    )
    return(select_fids)

import polars as pl
from datetime import datetime
def get_extra_file_descr(start_pred_date: pl.Date,
                         end_pred_date: pl.Date,
                         months_buffer: int,
                         test_version: str)-> str:
    extra = "test" + test_version
    if start_pred_date == datetime(2021, 1, 1) and end_pred_date == datetime(2023, 12, 31):
        extra += "_2021t2023"
    elif start_pred_date == datetime(2022, 6, 1) and end_pred_date == datetime(2022, 12, 31):
        extra += "_end2022"
    elif start_pred_date == datetime(2022, 1, 1) and end_pred_date == datetime(2022, 12, 31):
        extra += "_2022"
    elif start_pred_date == datetime(2022, 1, 1) and end_pred_date == datetime(2024, 12, 31):
        extra += "_2022t2024"
    else:
        raise("Please description of this prediction time period to the function 'get_extra_file_descr'.")
    if months_buffer != 0:
        extra += "_w" + str(months_buffer)
    return(extra)

import polars as pl
import pandas as pd
from datetime import datetime
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import logging_print
from abnorm_utils import get_abnorm_func_based_on_name
def label_cases_and_controls(data: pl.DataFrame,
                             start_pred_date: datetime,
                             end_pred_date: datetime,
                             min_per_year: bool,
                             lab_name: str,
                             abnorm_type: str,
                             removed_ids_path: str,
                             out_file_name: str) -> pl.DataFrame:

    if min_per_year:
        labels = (data.filter((pl.col.DATE>=start_pred_date)&(pl.col.DATE<=end_pred_date))
                    .group_by(pl.col.DATE.dt.year(), pl.col.FINNGENID, pl.col.SEX)
                              .agg(pl.col.VALUE.mean().alias("y_MEAN_YEAR"), 
                                   pl.col.VALUE.min().alias("y_MIN_YEAR"), 
                                   pl.col.VALUE.get(pl.col.DATE.arg_min()).alias("y_NEXT_YEAR"))
                    .with_columns(
                        pl.when(lab_name=="egfr")
                                .then(pl.col.y_MEAN_YEAR.min().over("FINNGENID"))
                                .otherwise(pl.col.y_MEAN_YEAR.max().over("FINNGENID"))
                                .alias("y_MEAN"),
                        pl.when(lab_name=="egfr")
                                .then(pl.col.y_MIN_YEAR.min().over("FINNGENID"))
                                .otherwise(pl.col.y_MIN_YEAR.max().over("FINNGENID"))
                                .alias("y_MIN"),
                        pl.col.y_NEXT_YEAR.get(pl.col.DATE.arg_min()).over("FINNGENID").alias("y_NEXT"),    
                   )
        )
    else:
        labels = (data.filter((pl.col.DATE>=start_pred_date)&(pl.col.DATE<=end_pred_date))
                    .with_columns(
                        pl.col.VALUE.mean().over("FINNGENID").alias("y_MEAN"),
                        pl.when((lab_name=="egfr")
                        .then(pl.col.VALUE.min().over("FINNGENID"))
                        .otherwise(pl.col.VALUE.max().over("FINNGENID"))
                        .alias("y_MIN")),
                        pl.col.VALUE.get(pl.col.DATE.arg_min()).over("FINNGENID").alias("y_NEXT"),
                    )
        )
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_MEAN").rename({"ABNORM_CUSTOM": "y_MEAN_ABNORM"})
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_MIN").rename({"ABNORM_CUSTOM": "y_MIN_ABNORM"})
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_NEXT").rename({"ABNORM_CUSTOM": "y_NEXT_ABNORM"})
    labels = labels.select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM").unique()
        
    no_labels_data_fids = data.filter(~pl.col.FINNGENID.is_in(labels["FINNGENID"]))["FINNGENID"].unique()
    logging_print("Removed " + str(len(set(no_labels_data_fids))) + " individuals because of no measurement in the time.")
    pd.DataFrame({"FINNGENID":list(set(no_labels_data_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_no-measure_fids.csv", sep=",")
    
    log_print_n(labels, "Start")
    return(labels)

import polars as pl
import pandas as pd
from datetime import datetime
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import logging_print
def remove_age_outliers(labels: pl.DataFrame,
                        base_date: datetime,
                        fg_ver: str,
                        min_age: float,
                        max_age: float,
                        removed_ids_path: str,
                        out_file_name: str) -> pl.DataFrame:
    labels = add_ages(labels, base_date, fg_ver=fg_ver)
    age_remove_fids = set(labels
                          .filter((pl.col("EVENT_AGE")<min_age)|(pl.col("EVENT_AGE")>max_age))
                          .get_column("FINNGENID")
    )
    logging_print("Removed " + str(len(set(age_remove_fids))) + " individuals because of age.")
    pd.DataFrame({"FINNGENID":list(set(age_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_age_fids.csv", sep=",")
    
    labels = labels.filter(~pl.col("FINNGENID").is_in(age_remove_fids))
    log_print_n(labels, "Age")
    return(labels)