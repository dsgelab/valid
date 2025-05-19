import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))

import polars as pl
from processing_utils import generate_random_date
from general_utils import read_file, logging_print

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


def remove_future_diags(data, 
                        end_pred_date,
                        strict=False):
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