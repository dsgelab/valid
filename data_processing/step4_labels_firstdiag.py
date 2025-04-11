# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, read_file, logging_print
from processing_utils import add_measure_counts, add_set
# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

def correct_ages(data: pl.DataFrame,
                 fg_ver) -> pl.DataFrame:
    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_1.0.txt.gz"

    ages = pl.read_csv(minimum_file_name,
                       separator="\t",
                        columns=["FINNGENID", "APPROX_BIRTH_DATE"])
    data = (data.join(ages, on="FINNGENID", how="left")
                .with_columns(pl.col.APPROX_BIRTH_DATE.cast(pl.Utf8).str.to_date(),
                              pl.col.START_DATE.cast(pl.Utf8).str.to_date()))
    data = data.with_columns(
            ((pl.col.START_DATE-pl.col.APPROX_BIRTH_DATE).dt.total_days()/365.25)
            .alias("END_AGE")
    )
    return(data.drop("APPROX_BIRTH_DATE"))
    
def get_controls(data: pl.DataFrame,  
                 no_abnormal_ctrls=1,
                 months_buffer=0):
    """Get controls based on the data.
         Controls are defined as individuals without a diagnosis and data-based diagnosis."""
    # Removing all individuals with a dia   
    controls = data.filter(pl.col("FIRST_DIAG_DATE").is_null())
    # Figuring out their last measurements (mostly for plotting in the end - predicting control status)
    controls_end_data = (controls.sort(["FINNGENID", "DATE"], descending=True)
                                 .group_by("FINNGENID")
                                 .head(1)
                                 .select(["FINNGENID", "DATE", "EVENT_AGE", "VALUE", "ABNORM_CUSTOM"])
                                 .rename({"DATE":"START_DATE", "EVENT_AGE": "END_AGE", "VALUE": "y_MEAN", "ABNORM_CUSTOM": "LAST_ABNORM"}))
    controls_end_data = controls_end_data.with_columns(pl.col("START_DATE").dt.offset_by(f"-{months_buffer}mo").alias("START_DATE"))
    controls = controls.join(controls_end_data, on="FINNGENID", how="left")
    controls = controls.with_columns(y_DIAG=0)
    # Removing cases with prior abnormal data - if wanted
    if(no_abnormal_ctrls == 1):
        ctrl_prior_data = controls.filter(pl.col("DATE") < pl.col("START_DATE"))
        remove_fids = (ctrl_prior_data
                       .group_by("FINNGENID")
                       .agg(pl.col("ABNORM_CUSTOM").sum().alias("N_PRIOR_ABNORM"))
                       .filter(pl.col("N_PRIOR_ABNORM")>=1)
                       .get_column("FINNGENID"))
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))
    else:
        # Always removing last abnormal without diagnosis because of fear of censoring
        remove_fids = controls_end_data.filter(pl.col("LAST_ABNORM")>=1).select("FINNGENID")
        remove_fids = set(remove_fids["FINNGENID"])
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))

    return(controls, remove_fids)

def get_cases(data: pl.DataFrame, 
              no_abnormal_cases=1,
              months_buffer=0,
              normal_before_diag=0) -> pl.DataFrame:
    """Get cases based on the data.
       Cases are defined as individuals with a diagnosis and data-based diagnosis. 
       The start date is the first abnormality that lead to the diagnosis or the first diagnosis date."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Cases have diagnosis # # # # # # # # # # # # # # # # # #
    cases = data.filter(pl.col("DATA_FIRST_DIAG_ABNORM_DATE").is_not_null())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Date of prediction # # # # # # # # # # # # # # # # # # #
    # Start date is either the first diag date or the first abnormal that lead to the diagnosis 
    # minus a buffer of `months_buffer` months
    first_measurement = (cases
                         .sort(["FINNGENID", "DATE"], descending=False)
                         .group_by("FINNGENID")
                         .head(1)
                         .select(["FINNGENID", "DATE", "ABNORM_CUSTOM"])
                         .rename({"DATE": "FIRST_MEASUREMENT_DATE", "ABNORM_CUSTOM": "FIRST_MEASUREMENT_ABNORM"}))
    cases = cases.join(first_measurement, on="FINNGENID", how="left")
    cases = cases.with_columns(y_DIAG=1, 
                               START_DATE=pl.min_horizontal(["FIRST_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"]))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # No prior abnormal (if wanted) # # # # # # # # # # # # # #
    # removing individuals whose measurements only start after or with diagnosis
    # either data-based all abnormal or very old ICD-code diagnosis
    remove_fids = set(cases
                      .filter(pl.col("FIRST_MEASUREMENT_DATE") >= pl.col("FIRST_DIAG_DATE"))
                      .get_column("FINNGENID")
                      .unique())
    if normal_before_diag == 1:
        # removing individuals whose measurements only start after diagnosis
        # this means they need at least one normal before diagnosis
        remove_fids |= set(cases
                           .filter(pl.col("FIRST_MEASUREMENT_ABNORM") >= 1)
                           .get_column("FINNGENID")
                           .unique())
    
    if no_abnormal_cases == 1:
        case_prior_data = cases.filter(pl.col("DATE") < pl.col("START_DATE"))
        remove_fids |= set(case_prior_data
                            .group_by("FINNGENID")
                            .agg(pl.col("ABNORM_CUSTOM").sum().alias("N_PRIOR_ABNORM"))
                            .filter(pl.col("N_PRIOR_ABNORM")>=1)
                            .get_column("FINNGENID"))
    cases = cases.filter(~pl.col("FINNGENID").is_in(remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Age at prediction time # # # # # # # # # # # # # # # # #
    # Age and value at first abnormality that lead to the data-based diagnosis
    # it might be that diag earlier TODO deal with later
    case_ages = (cases.filter(pl.col("DATE")==pl.col("DATA_FIRST_DIAG_ABNORM_DATE"))
                      .group_by("FINNGENID").head(1) # bug there might be multiple measurements TODO fix later in earlier steps 
                      .rename({"EVENT_AGE": "END_AGE", "VALUE": "y_MEAN", "ABNORM_CUSTOM": "LAST_ABNORM"})
                      .select(["FINNGENID", "END_AGE", "y_MEAN", "LAST_ABNORM"])
                      .unique())
    cases = cases.join(case_ages, on="FINNGENID", how="left")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Removing buffer time from start # # # # # # # # # # # # #
    cases = cases.with_columns(pl.col("START_DATE").dt.offset_by(f"-{months_buffer}mo").alias("START_DATE"))
    cases = cases.drop(["FIRST_MEASUREMENT_DATE", "FIRST_MEASUREMENT_ABNORM"])
    return(cases, remove_fids)
import calendar
from datetime import datetime
def sample_controls(cases: pl.DataFrame,
                    controls: pl.DataFrame,
                    months_buffer: int,
                    fg_ver: str) -> pl.DataFrame:
    """Sample controls based on the cases."""

    # Example: Step 1: Get the number of cases per month
    case_bins = (cases.with_columns([
                            pl.col("START_DATE").dt.year().alias("CASE_BIN")
                        ])
                      .group_by("CASE_BIN")
                      .agg(pl.len().alias("N_CASES"))
    )
    # Step 2: Expand case bin counts into a list of index dates
    case_index_pool = (case_bins
                        .select([
                            pl.col("CASE_BIN").repeat_by("N_CASES")
                        ])
                        .explode("CASE_BIN")
                        .rename({"CASE_BIN": "SAMPLED_INDEX_DATE"})
    )
    # Step 3: Get one row per control
    control_meta = (controls.select("FINNGENID").unique())

    # Step 4: Sample index dates for controls from case_index_pool
    n_controls = control_meta.height
    sampled_dates = case_index_pool.sample(n_controls, with_replacement=True).to_series()

    # Step 5: Attach sampled index dates to controls
    months = np.random.randint(1, 13)
    
    # Function to generate a random date for each year
    def generate_random_date(year):
        # Randomly select a month (1 to 12)
        month = np.random.randint(1, 13)
        
        # Get the number of days in the selected month (handling leap years for February)
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Randomly select a day within that month
        day = np.random.randint(1, days_in_month + 1)
        
        # Return the sampled random date
        return datetime(year, month, day).date()
    
    # Vectorized approach to generate random dates for the 'year' column
    control_meta = control_meta.with_columns([
        pl.Series("START_DATE", sampled_dates).map_elements(generate_random_date, return_dtype=pl.Date).alias("START_DATE")
    ])
    n_prior = control_meta.get_column("FINNGENID").len()

    controls = controls.drop(["START_DATE"])

    # Step 7: Join back to df_controls and truncate to the window
    controls = controls.join(control_meta, on="FINNGENID", how="inner")
    controls = correct_ages(controls, fg_ver)

    return(controls)


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--data_path", type=str, help="Path to data.", default="/home/ivm/valid/data/processed_data/step3_meta/")
    parser.add_argument("--file_name", type=str, help="Path to data.")
    parser.add_argument("--exclusion_path", type=str, help="Path to exclusions, atm excluding all FGIDs in there.")
    parser.add_argument("--exclude_diags", type=int, default=0, help="Whether to exclude individuals with prior diags or not. 0 = no, 1 = yes")
    parser.add_argument("--fg_ver", type=str, default="R12", help="Which FinnGen release [options: R12 and R13]")

    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)

    # Settings for data-processing
    parser.add_argument("--test_pct", type=float, help="Percentage of test data.", default=0.2)
    parser.add_argument("--valid_pct", type=float, help="Percentage of validation data.", default=0.2)
    parser.add_argument("--no_abnorm", type=float, help="Whether to create super controls where all prior data is normal.", default=1)
    parser.add_argument("--normal_before_diag", type=float, help="Whether cases need at least one recorded normal before diagnosis.", default=0)
    parser.add_argument("--sample_ctrls", type=float, help="Whether cases need at least one recorded normal before diagnosis.", default=0)
    parser.add_argument("--start_year", type=float, help="Minimum number of years with at least one measurement to be included", default=2013)

    parser.add_argument("--min_n_years", type=float, help="Minimum number of years with at least one measurement to be included", default=2)
    parser.add_argument("--months_buffer", type=int, help="Minimum number months before prediction to be removed.", default=2)
    parser.add_argument("--min_age", type=float, help="Minimum age at prediction time", default=30)
    parser.add_argument("--max_age", type=float, help="Maximum age at prediction time", default=70)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()

    # Setting up logging
    extra = ""
    if args.no_abnorm == 1:
        extra = "-noabnorm"
    if args.normal_before_diag:
        extra += "-normbefore"     
    if args.sample_ctrls == 1:
        extra += "-ctrlsample"
    if args.start_year > 2013:
        extra += "-start" + str(int(args.start_year))
    out_file_name = args.file_name + "_data-diag" + extra + "_" + get_date() 
        
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.file_name + "_" + get_datetime(), logger, args)
    make_dir(args.res_dir + "logs/removed_fids/")
    logging.info("Saving files to: "+out_file_name)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(args.data_path + args.file_name + ".parquet")
    metadata = read_file(args.data_path + args.file_name + "_meta.parquet",
                         schema={"FIRST_DIAG_DATE": pl.Date, "DATA_FIRST_DIAG_ABNORM_DATE": pl.Date, "DATA_DIAG_DATE": pl.Date})
    data = data.join(metadata, on="FINNGENID", how="left")
    data = data.with_columns(
                    pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("DATE"),
                    pl.col("DATA_FIRST_DIAG_ABNORM_DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("DATA_FIRST_DIAG_ABNORM_DATE"),
                    pl.col("FIRST_DIAG_DATE").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("FIRST_DIAG_DATE")
    )
    ### Processing
    if args.min_n_years > 1:
        data = add_measure_counts(data)
        fids = set(data.filter(pl.col("N_YEAR")==args.min_n_years)["FINNGENID"])
        logging.info("Removed " + str(len(fids)) + " individuals with less than two years data")
        data = data.filter(~pl.col("FINNGENID").is_in(fids))
        pd.DataFrame({"FINNGENID":list(fids)}).to_csv(args.res_dir + "logs/removed_fids/" + args.file_name + out_file_name + "_reason_nyears.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cases and controls                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    cases, case_remove_fids = get_cases(data, args.no_abnorm, args.months_buffer, args.normal_before_diag)
    controls, ctrl_remove_fids = get_controls(data, args.no_abnorm, args.months_buffer)
    new_data = pl.concat([cases, controls.select(cases.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Year                                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    year_remove_fids = set(new_data.filter(pl.col("START_DATE").dt.year()<(args.start_year-1))["FINNGENID"])
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(year_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Age                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    age_remove_fids = set(new_data.filter((pl.col("END_AGE")<args.min_age)|(pl.col("END_AGE")>args.max_age))["FINNGENID"])
    new_data = new_data.filter(~pl.col("FINNGENID").is_in(age_remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Diag exclusions                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    if args.exclude_diags == 1:
        exclusion_data = read_file(args.exclusion_path)
        diag_remove_fids = (exclusion_data
                            .join(new_data.select(["FINNGENID", "START_DATE"]), on="FINNGENID", how="left")
                            .filter(pl.col("EXCL_DATE") > pl.col("START_DATE"))
                            .get_column("FINNGENID"))
        new_data = new_data.filter(~pl.col("FINNGENID").is_in(diag_remove_fids))
        logging.info("Removed " + str(len(set(diag_remove_fids))) + " individuals because of diagnosis exclusions.")
        pd.DataFrame({"FINNGENID":list(set(diag_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_diag-exclusion_fids.csv", sep=",")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Control sampling                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.sample_ctrls == 1:
        controls = sample_controls(new_data.filter(pl.col("y_DIAG")==1), 
                                   new_data.filter(pl.col("y_DIAG")==0), 
                                   args.months_buffer,
                                   args.fg_ver)
        new_data = pl.concat([new_data.filter(pl.col("y_DIAG")==1), 
                              controls.select(new_data.columns)]) 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Labels                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    labels = new_data.select(["FINNGENID", "y_DIAG", "END_AGE", "SEX","y_MEAN", "START_DATE"]).rename({"END_AGE": "EVENT_AGE"}).unique(subset=["FINNGENID", "y_DIAG"])
    labels = add_set(labels, args.test_pct, args.valid_pct)
    logging_print(f"N rows data {len(new_data)} N rows labels {len(labels)}  N indvs {len(set(labels["FINNGENID"]))}  N cases {sum(labels["y_DIAG"])} pct cases {round(sum(labels["y_DIAG"])/len(labels)*100,2)}%")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data before start                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    new_data = new_data.filter(pl.col("DATE") < pl.col("START_DATE"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info("Removed " + str(len(age_remove_fids)) + " individuals with age < " + str(args.min_age) + " or > " + str(args.max_age))
    pd.DataFrame({"FINNGENID":list(set(age_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_age.csv", sep=",")
    logging.info("Removed " + str(len(year_remove_fids)) + " individuals with time of prediction < " + str(args.start_year))
    pd.DataFrame({"FINNGENID":list(set(year_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_year.csv", sep=",")
    logging.info("Removed " + str(len(set(case_remove_fids))) + " individuals because of case with prior abnormal ")
    pd.DataFrame({"FINNGENID":list(set(case_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_case-abnorm_fids.csv", sep=",")
    logging.info("Removed " + str(len(set(ctrl_remove_fids))) + " individuals because of controls with prior abnormal or last abnormal if no ctrl abnormal filtering")
    pd.DataFrame({"FINNGENID":list(set(ctrl_remove_fids))}).to_csv(args.res_dir + "logs/removed_fids/" + out_file_name + "_reason_ctrl-abnorm_fids.csv", sep=",")


    new_data.select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"]).write_parquet(args.res_dir + out_file_name + ".parquet")
    labels.write_parquet(args.res_dir + out_file_name + "_labels.parquet")
    #Final logging
    logger.info("Time total: "+timer.get_elapsed())
