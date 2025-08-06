import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))

import polars as pl
import pandas as pd
from processing_utils import get_abnorm_func_based_on_name
from general_utils import read_file, logging_print
from datetime import datetime

def log_print_n(labels: pl.DataFrame,
                name: str):
    logging_print(name)
    logging_print(f"MEAN N indvs {labels.height}  N cases { labels.get_column("y_MEAN_ABNORM").sum()} pct cases {round( labels.get_column("y_MEAN_ABNORM").sum()/labels.height*100,2)}%")
    logging_print(f"MIN N indvs {labels.height}  N cases { labels.get_column("y_MIN_ABNORM").sum()} pct cases {round( labels.get_column("y_MIN_ABNORM").sum()/labels.height*100,2)}%")
    logging_print(f"NEXT N indvs {labels.height}  N cases { labels.get_column("y_NEXT_ABNORM").sum()} pct cases {round( labels.get_column("y_NEXT_ABNORM").sum()/labels.height*100,2)}%")
    logging_print(f"DIAG N indvs {labels.height}  N cases { labels.get_column("y_DIAG").sum()} pct cases {round( labels.get_column("y_DIAG").sum()/labels.height*100,2)}%")
    
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


def get_low_bmi_indvs(fg_ver="R13") -> pl.Series:
    """BMI not recorded or BMI >= 18.5 (low might make eGFR unreliable)."""

    # Read in the minimum data file
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_extended_1.0.txt.gz"
    min_data = pl.read_csv(minimum_file_name, 
                           separator="\t",
                           columns=["FINNGENID", "BMI"])
    min_data = min_data.with_columns([
                pl.col.BMI.cast(pl.Float64, strict=False).alias("BMI")
    ])
    # Filtering
    select_fids = (min_data
                   .filter(# BMI not recorded or BMI >= 18.5 (low might make eGFR unreliable)
                       ((pl.col.BMI.is_null()) | (pl.col.BMI>=18.5)) 
                   )
                   .get_column("FINNGENID")
    )
    return(select_fids)

def remove_prior_diags(data: pl.DataFrame,
                       start_pred_date: datetime,
                       data_diag_excl: bool,
                       removed_ids_path: str,
                       out_file_name: str) -> pl.DataFrame:
    """Remove individuals with a diagnosis before the start date of the prediction."""
    # Remove individuals with a diagnosis before the start date of the prediction
    data_prior_diag_removes = remove_future_diags(data, 
                                                  start_pred_date, 
                                                  strict=True,
                                                  data_diag_excl=data_diag_excl)
    # individuals with a diagnosis before the start date of the prediction
    prior_diags_fids = data_prior_diag_removes.filter(~pl.col.FIRST_DIAG_DATE.is_null())["FINNGENID"].unique()
    data = data.filter(~pl.col.FINNGENID.is_in(prior_diags_fids))

    logging_print("Removed " + str(len(set(prior_diags_fids))) + " individuals because of prior diagnosis.")
    pd.DataFrame({"FINNGENID":list(set(prior_diags_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_prior-diag_fids.csv", sep=",")

    return(data)

def remove_only_icd_or_atc_diags(data: pl.DataFrame,
                                 removed_ids_path: str,
                                 out_file_name: str) -> pl.DataFrame:
    # Data-diagnosis is missing but not first diagnosis
    other_diags_fids = data.filter((pl.col.DATA_DIAG_DATE.is_null()&(~pl.col.FIRST_DIAG_DATE.is_null())))["FINNGENID"].unique()
    data = data.filter(~pl.col.FINNGENID.is_in(other_diags_fids))

    logging_print("Removed " + str(len(set(other_diags_fids))) + " individuals because of other diagosis but not data.")
    pd.DataFrame({"FINNGENID":list(set(other_diags_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_only-other-diags_fids.csv", sep=",")

    return(data)

def remove_buffer_abnorm(data: pl.DataFrame,
                         start_noabnorm_date: datetime,
                         base_date: datetime,
                         lab_name: str,
                         abnorm_type: str,
                         removed_ids_path: str,
                         out_file_name: str) -> pl.DataFrame:
    
    # Mean during the buffer period before baseline
    abnorm_data = (data
                       .filter((pl.col.DATE >= start_noabnorm_date)&(pl.col.DATE < base_date))
                       .with_columns(pl.col.VALUE.mean().over("FINNGENID").alias("MEAN_PRIOR"))
                       .select("FINNGENID", "MEAN_PRIOR")
    )
    # transform mean into abnormal
    abnorm_data = get_abnorm_func_based_on_name(lab_name, abnorm_type)(abnorm_data, "MEAN_PRIOR")
    # Remove individuals with abnormal mean
    abnorm_fids = abnorm_data.filter(pl.col.ABNORM_CUSTOM>0).get_column("FINNGENID").unique()
    data = data.filter(~pl.col.FINNGENID.is_in(abnorm_fids))
    # Logging
    logging_print("Removed " + str(len(set(abnorm_fids))) + " individuals because of prior mean abnormal.")
    pd.DataFrame({"FINNGENID":list(set(abnorm_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_prior-abnorm_fids.csv", sep=",")

    return(data)    

def remove_last_value_abnormal(data: pl.DataFrame,
                               base_date: datetime,
                               removed_ids_path: str,
                               out_file_name: str) -> pl.DataFrame:
    # Diagnosis is inside prediction period
    # But first abnormal is not before baseline date
    last_val_abnormal_fids = (data.filter(pl.col.DATE<base_date)
                                  .filter(pl.col.ABNORM_CUSTOM!=0.5)
                                  .filter(pl.col.DATE==pl.col.DATE.max().over("FINNGENID"))
                                  .filter(pl.col.ABNORM_CUSTOM==1)
                                  .get_column("FINNGENID")
                                  .unique()
    )
    logging_print("Removed " + str(len(set(last_val_abnormal_fids))) + " individuals because of last value abnormal.")
    pd.DataFrame({"FINNGENID":list(set(last_val_abnormal_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_last-abnormal_fids.csv", sep=",")
    data = data.filter(~pl.col.FINNGENID.is_in(last_val_abnormal_fids))
    return(data)    

def label_cases_and_controls(data: pl.DataFrame,
                             start_pred_date: datetime,
                             end_pred_date: datetime,
                             lab_name: str,
                             abnorm_type: str,
                             removed_ids_path: str,
                             out_file_name: str) -> pl.DataFrame:

    labels = (data.filter((pl.col.DATE>=start_pred_date)&(pl.col.DATE<=end_pred_date))
                .with_columns(
                    pl.col.VALUE.mean().over("FINNGENID").alias("y_MEAN"),
                    pl.col.VALUE.min().over("FINNGENID").alias("y_MIN"),
                    pl.col.VALUE.get(pl.col.DATE.arg_min()).over("FINNGENID").alias("y_NEXT"),
                    # Cases should have the diagnosis date inside the prediction period
                    # Will not have first abnormal here, if heappened earlier but no diag yet.
                    # Or also the first abnormal is enough here
                    pl.when(((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>=start_pred_date)&(pl.col.DATA_FIRST_DIAG_ABNORM_DATE<=end_pred_date)) | \
                            ((pl.col.DATA_DIAG_DATE>=start_pred_date)&(pl.col.DATA_DIAG_DATE<=end_pred_date)))              
                      .then(1).otherwise(0).alias("y_DIAG")
                )
    )
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_MEAN").rename({"ABNORM_CUSTOM": "y_MEAN_ABNORM"})
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_MIN").rename({"ABNORM_CUSTOM": "y_MIN_ABNORM"})
    labels = get_abnorm_func_based_on_name(lab_name, abnorm_type)(labels, "y_NEXT").rename({"ABNORM_CUSTOM": "y_NEXT_ABNORM"})
    labels = labels.select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM", "y_DIAG").unique()
    
    no_labels_data_fids = data.filter(~pl.col.FINNGENID.is_in(labels["FINNGENID"]))["FINNGENID"].unique()
    logging_print("Removed " + str(len(set(no_labels_data_fids))) + " individuals because of no measurement in the time.")
    pd.DataFrame({"FINNGENID":list(set(no_labels_data_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_no-measure_fids.csv", sep=",")
    
    log_print_n(labels, "Start")
    return(labels)

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

def remove_other_exclusion(labels: pl.DataFrame,
                           diags_path: str,
                           exclude_file_name: str,
                           base_date: datetime,
                           removed_ids_path: str,
                           out_file_name: str) -> pl.DataFrame:
    exclusion_data = read_file(diags_path+exclude_file_name)
    # We do not want any of the exclusion individuals in the test set
    excl_remove_fids = (exclusion_data
                            .filter((pl.col("EXCL_DATE") < base_date)&(pl.col.FINNGENID.is_in(labels["FINNGENID"])))
                            .get_column("FINNGENID"))
    labels = labels.filter(~pl.col("FINNGENID").is_in(excl_remove_fids))

    logging_print("Removed " + str(len(set(excl_remove_fids))) + " individuals because of diagnosis exclusions.")
    pd.DataFrame({"FINNGENID":list(set(excl_remove_fids))}).to_csv(removed_ids_path + out_file_name + "_reason_diag-exclusion_fids.csv", sep=",")

    log_print_n(labels, "Exclusion")
    return(labels)

def remove_future_diags(data, 
                        end_pred_date,
                        strict=False,
                        data_diag_excl=True,
                        diff_days=0):
    if data_diag_excl == True:
        # Future diagnoses
        if strict:
            # Removing all information that leads to a diagnosis after the end_pred_date
            # That means we also remove the first abnormal date, considering it simply
            # a "solo" abnormal date
            data = data.with_columns([
                # First diagnosis after end_pred_date -> first abnormal date set to None
                pl.when(pl.col.FIRST_DIAG_DATE>end_pred_date)
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                    .alias("DATA_FIRST_DIAG_ABNORM_DATE"),
                # First overall diagnosis after end_pred_date -> set to None
                pl.when(pl.col.FIRST_DIAG_DATE>end_pred_date)
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.FIRST_DIAG_DATE)
                    .alias("FIRST_DIAG_DATE"),
                # First data-based diagnosis after end_pred_date -> set to None
                pl.when(pl.col.DATA_DIAG_DATE>end_pred_date)
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.DATA_DIAG_DATE)
                    .alias("DATA_DIAG_DATE")
                ])
        else:
            # Removing only those individuals who also have a first measurement that is after the end_pred_date
            # We keep individuals whos second is abnormal date is after the end_pred_date
            # we remove first diagnosis after the end_pred_date dates without an abnormal date
            data = data.with_columns([
                # First abnormal leading to diagnosis OR first diagnosis without abnormal date -> set first diagnosis to None
                pl.when((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date)|
                        ((pl.col.FIRST_DIAG_DATE>end_pred_date)&(pl.col.DATA_FIRST_DIAG_ABNORM_DATE.is_null())))
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.FIRST_DIAG_DATE)
                    .alias("FIRST_DIAG_DATE"),
                # First data-based diagnosis after end_pred_date -> set both abnorma and diagnosis date to None
                pl.when((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date))
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.DATA_DIAG_DATE)
                    .alias("DATA_DIAG_DATE"),
                pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date)
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                    .alias("DATA_FIRST_DIAG_ABNORM_DATE")
                ])
    else:
        data = data.with_columns(
                # There is a diagnosis but it is not data-based
                pl.when(((pl.col.FIRST_ICD_DIAG_DATE.is_null())|(pl.col.FIRST_MED_DIAG_DATE.is_null()))\
                        &(~pl.col.DATA_FIRST_DIAG_ABNORM_DATE.is_null()))
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.FIRST_DIAG_DATE)
                    .alias("FIRST_DIAG_DATE"),
        )
        data = data.with_columns(
                # but still the diagnosis is after the end_pred_date
                pl.when((pl.col.FIRST_DIAG_DATE>end_pred_date))
                    .then(pl.lit(None).cast(pl.Date))
                    .otherwise(pl.col.FIRST_DIAG_DATE)
                    .alias("FIRST_DIAG_DATE")
        )

    return(data)

def get_lab_data(data_path: str, 
                 diags_path: str) -> pl.DataFrame:
    """Get lab data."""
    data = read_file(data_path, schema={"DATE": pl.Date})
    if data.schema["DATE"] == pl.Datetime: data=data.with_columns(pl.col.DATE.dt.date().alias("DATE"))
    metadata = read_file(diags_path,
                         schema={"FIRST_DIAG_DATE": pl.Date, 
                                 "DATA_FIRST_DIAG_ABNORM_DATE": pl.Date, 
                                 "DATA_DIAG_DATE": pl.Date,
                                 "FIRST_ICD_DIAG_DATE": pl.Date,
                                 "FIRST_MED_DIAG_DATE": pl.Date})
    data = data.join(metadata, on="FINNGENID", how="left")
    data = data.with_columns(
                    pl.col("DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATE"),
                    pl.col("DATA_FIRST_DIAG_ABNORM_DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_FIRST_DIAG_ABNORM_DATE"),
                    pl.col("FIRST_DIAG_DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("FIRST_DIAG_DATE"),
                    pl.col("DATA_DIAG_DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_DIAG_DATE")

    )

    return(data)