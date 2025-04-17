
import polars as pl
from processing_utils import generate_random_date
from general_utils import read_file

def add_ages(data: pl.DataFrame,
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
                              pl.col.PRED_DATE.cast(pl.Utf8).str.to_date()))
    data = data.with_columns(
            ((pl.col.START_DATE-pl.col.APPROX_BIRTH_DATE).dt.total_days()/365.25).floor()
            .alias("END_AGE")
    )
    return(data.drop("APPROX_BIRTH_DATE"))
   
def add_pred_helpers(data: pl.DataFrame,
                     fg_ver: str) -> pl.DataFrame:
    after_pred_data = data.filter(pl.col.DATE>=pl.col.PRED_DATE)
    after_pred_data = (after_pred_data
                       .filter((pl.col.DATE==pl.col.DATE.min()).over("FINNGENID"))
                       .select("FINNGENID", "VALUE")
                       .rename({"VALUE": "y_MEAN"})
                      )
    data = data.join(after_pred_data, on="FINNGENID", how="left")
    data = add_ages(data, fg_ver)
    return(data)


def remove_future_data(data, 
                       end_pred_date,
                       strict=False):
    data = data.filter(pl.col.DATE<end_pred_date)
    # Future diagnoses
    if strict:
        data = data.with_columns([
            pl.when(pl.col.FIRST_DIAG_DATE>end_pred_date)
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                .alias("DATA_FIRST_DIAG_ABNORM_DATE"),
            pl.when(pl.col.FIRST_DIAG_DATE>end_pred_date)
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.FIRST_DIAG_DATE)
                .alias("FIRST_DIAG_DATE"),
            pl.when(pl.col.DATA_DIAG_DATE>end_pred_date)
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.DATA_DIAG_DATE)
                .alias("DATA_DIAG_DATE")
            ])
    else:
        data = data.with_columns([
            pl.when((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date)|
                    ((pl.col.FIRST_DIAG_DATE>end_pred_date)&(pl.col.DATA_FIRST_DIAG_ABNORM_DATE.is_null())))
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.FIRST_DIAG_DATE)
                .alias("FIRST_DIAG_DATE"),
            pl.when((pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date)|
                    ((pl.col.DATA_DIAG_DATE>end_pred_date)&(pl.col.DATA_FIRST_DIAG_ABNORM_DATE.is_null())))
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.DATA_DIAG_DATE)
                .alias("DATA_DIAG_DATE"),
            pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE>end_pred_date)
                .then(pl.lit(None).cast(pl.Date))
                .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                .alias("DATA_FIRST_DIAG_ABNORM_DATE")
            ])

    return(data)
    
def get_controls(data: pl.DataFrame,  
                 no_abnorm=1,
                 months_buffer=0):
    """Get controls based on the data.
         Controls are defined as individuals without a diagnosis and data-based diagnosis."""
    # Removing all individuals with a dia   
    
    controls = data.filter(pl.col.FIRST_DIAG_DATE.is_null()&pl.col.DATA_FIRST_DIAG_ABNORM_DATE.is_null())
    # Figuring out their last measurements (mostly for plotting in the end - predicting control status)
    controls_end_data = (controls.filter(pl.col.DATE==pl.col.DATE.max().over("FINNGENID"))
                                 .select(["FINNGENID", "DATE", "ABNORM_CUSTOM"])
                                 .rename({"DATE": "PRED_DATE", "ABNORM_CUSTOM": "LAST_ABNORM"}))
    # Adding buffer time where we remove data
    controls_end_data = controls_end_data.with_columns(
                                pl.col("PRED_DATE").dt.offset_by(f"-{months_buffer}mo").dt.date().alias("START_DATE"),
    )
    controls = controls.join(controls_end_data, on="FINNGENID", how="left")
    controls = controls.with_columns(y_DIAG=0)
    # Removing cases with prior abnormal data - if wanted
    if(no_abnorm == 1):
        ctrl_prior_data = controls.filter(pl.col("DATE") < pl.col("START_DATE"))
        remove_fids = (ctrl_prior_data
                       .agg(pl.col("ABNORM_CUSTOM").sum().over("FINNGENI").alias("N_PRIOR_ABNORM"))
                       .filter(pl.col("N_PRIOR_ABNORM")>=1)
                       .get_column("FINNGENID"))
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))
    else:
        # Always removing last abnormal without diagnosis because of fear of censoring
        remove_fids = controls_end_data.filter(pl.col("LAST_ABNORM")>=1).select("FINNGENID")
        remove_fids = set(remove_fids["FINNGENID"])
        controls = controls.filter(~pl.col("FINNGENID").is_in(remove_fids))
        
    controls = controls.with_columns([
                    pl.col.PRED_DATE.dt.date().alias("PRED_DATE"),
                    pl.col.DATE.dt.date().alias("DATE")
    ])
    return(controls, remove_fids)

def get_cases(data: pl.DataFrame, 
              no_abnorm=1,
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
                         .filter(pl.col.DATE==pl.col.DATE.min().over("FINNGENID")).unique()
                         .select(["FINNGENID", "DATE", "ABNORM_CUSTOM"])
                         .rename({"DATE": "FIRST_MEASUREMENT_DATE", 
                                  "ABNORM_CUSTOM": "FIRST_MEASUREMENT_ABNORM"}))
    cases = cases.join(first_measurement, on="FINNGENID", how="left", coalesce=True)
    cases = cases.with_columns(y_DIAG=1, 
                               PRED_DATE=pl.min_horizontal(["FIRST_DIAG_DATE", 
                                                            "DATA_FIRST_DIAG_ABNORM_DATE"]))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # No prior abnormal (if wanted) # # # # # # # # # # # # # #
    # removing individuals whose measurements only start after or with diagnosis
    # either data-based all abnormal or very old ICD-code diagnosis
    remove_fids = set(cases
                      .filter(pl.col("FIRST_MEASUREMENT_DATE") > pl.col("FIRST_DIAG_DATE"))
                      .get_column("FINNGENID")
                      .unique())
    if normal_before_diag == 1:
        # removing individuals whose measurements only start after diagnosis
        # this means they need at least one normal before diagnosis
        remove_fids |= set(cases
                           .filter(pl.col("FIRST_MEASUREMENT_ABNORM") >= 1)
                           .get_column("FINNGENID")
                           .unique())
    
    if no_abnorm == 1:
        case_prior_data = cases.filter(pl.col("DATE") < pl.col("PRED_DATE"))
        remove_fids |= set(case_prior_data
                            .agg(pl.col("ABNORM_CUSTOM").sum().over("FINNGENID").alias("N_PRIOR_ABNORM"))
                            .filter(pl.col("N_PRIOR_ABNORM")>=1)
                            .get_column("FINNGENID"))
    cases = cases.filter(~pl.col("FINNGENID").is_in(remove_fids))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # # # # # # # # # Removing buffer time from start # # # # # # # # # # # # #
    cases = cases.with_columns(
                    pl.col("PRED_DATE").dt.offset_by(f"-{months_buffer}mo").dt.date().alias("START_DATE")
                )
    cases = cases.drop(["FIRST_MEASUREMENT_DATE", "FIRST_MEASUREMENT_ABNORM"])

    cases = cases.with_columns([
                    pl.col.PRED_DATE.dt.date().alias("PRED_DATE"),
                    pl.col.DATE.dt.date().alias("DATE")
    ])

    return(cases, remove_fids)

def sample_controls(cases: pl.DataFrame,
                    controls: pl.DataFrame,
                    months_buffer: int=6) -> pl.DataFrame:
    """Sample controls based on the cases."""

    # Example: Step 1: Get the number of cases per year
    case_bins = (cases.with_columns([pl.col("PRED_DATE").dt.year().alias("CASE_BIN")])
                      .group_by("CASE_BIN")
                      .agg(pl.len().alias("N_CASES"))
    )
    # Step 2: Expand case bin counts into a list of index dates
    case_index_pool = (case_bins
                        .select([pl.col("CASE_BIN").repeat_by("N_CASES")])
                        .explode("CASE_BIN")
                        .rename({"CASE_BIN": "SAMPLED_INDEX_DATE"})
    )
    # Step 3: Get one row per control
    control_meta = (controls.select("FINNGENID").unique())

    # Step 4: Sample index dates for controls from case_index_pool
    sampled_dates = case_index_pool.sample(control_meta.height, with_replacement=True).to_series()

    # Vectorized approach to generate random dates for the 'year' column
    control_meta = control_meta.with_columns([
        pl.Series("PRED_DATE", sampled_dates)
          .map_elements(generate_random_date, return_dtype=pl.Date)
          .alias("PRED_DATE")
    ])
    control_meta = control_meta.with_columns(
                    pl.col("PRED_DATE").dt.offset_by(f"-{months_buffer}mo").dt.date().alias("START_DATE")
    )
    # Step 7: Join back to df_controls and truncate to the window
    controls = controls.drop(["PRED_DATE", "START_DATE"])
    controls = controls.join(control_meta, on="FINNGENID", how="inner")

    return(controls)

def get_lab_data(data_path: str, 
                 diags_path: str) -> pl.DataFrame:
    """Get lab data."""
    data = read_file(data_path)
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
