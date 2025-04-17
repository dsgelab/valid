
# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file

# Standard stuff
import polars as pl
import argparse
# Logging
import logging
logger = logging.getLogger(__name__)


def get_abnorm_start_dates(data: pl.DataFrame) -> pl.DataFrame:
    """
    Get the start dates of abnormality sequences.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    data = data.with_columns(pl.when(pl.col.ABNORM_CUSTOM==0).then(0).otherwise(1).alias("ABNORM_BIN"))
    data = (data
            .filter((pl.len()>1).over("FINNGENID"))
            .filter(pl.col.ABNORM_BIN.is_not_null())
            .sort(["FINNGENID", "DATE"], descending=False)
            .select("FINNGENID", "DATE", "ABNORM_BIN")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Shifting data                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
            .with_columns(pl.col.ABNORM_BIN.shift(1).over("FINNGENID").alias("PREV_ABNORM"))
    )
    data = (data
            .with_columns(pl.when(pl.col.PREV_ABNORM.is_null())
                          .then(-1)
                          .otherwise(pl.col.PREV_ABNORM)
                          .alias("PREV_ABNORM"),

            )
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Find Start                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # Marks start of sequence where the prior abnormality is different from the current one
    # -1 is used to mark the start of the first sequence
    # Otherwise it will bee some shift from 0 to 1 or 1 to 0 etc.
    data = (data
            .with_columns(pl.when(pl.col.ABNORM_BIN!=pl.col.PREV_ABNORM)
                            .then(pl.lit("START"))
                            .otherwise(pl.lit("CONTINUE"))
                            .alias("START")
            )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Add start dates                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    data = (data
            .with_columns(pl.when(pl.col.START=="START").then(pl.col.DATE).otherwise(None).alias("START_DATE"))
    )
    # Recursively fill the start dates forward until all are filled
    # This is done to ensure that all start dates are filled in the same sequence
    data = data.with_columns(
                pl.col("START_DATE").fill_null(strategy="forward").over("FINNGENID")
    )
    # Add time difference between the start of the sequence and the current date
    data = data.with_columns((pl.col.DATE-pl.col.START_DATE).alias("DIFF"))

    return(data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path", type=str, help="Path to the data file", required=True)
    parser.add_argument("--file_name", type=str, help="Name of the data file", required=True)
    parser.add_argument("--diags_path_start", type=str, help="Path to the diagnosis data file", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    # Settings
    parser.add_argument("--diff_days", type=int, help="Minimum number of days between measurements needed for diagnosis.", required=True)
    args = parser.parse_args()
    return(args)


#do the main file functions and runs 
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    data = read_file(args.file_path+args.file_name+".parquet")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Abnormality block lengths                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # Lets just remove if present
    data = data.filter(pl.col.ABNORM_CUSTOM != 0.5)
    data = get_abnorm_start_dates(data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data-based diagnosis                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    all_data_diags = (data
                      # Candiates of abnormality sequences, x days apart
                      .filter((pl.col("ABNORM_BIN") >= 1)&(pl.col("DIFF") >= args.diff_days))
                      .sort(["START_DATE", "DIFF"], descending=False)
                      # Longest difference from the start of this abnormality
                      .filter((pl.col.DIFF==pl.col.DIFF.first()).over(["FINNGENID", "START_DATE"]))
                      # First abnormality sequence that satisfies the criteria
                      .filter((pl.col.START_DATE==pl.col.START_DATE.first()).over("FINNGENID"))
                      .with_columns(pl.col("DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_DIAG_DATE"),
                                    pl.col("START_DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_FIRST_DIAG_ABNORM_DATE"))
    )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 ICD-based diagnoses                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    icd_data = read_file(args.diags_path_start+"_diags.parquet")
    icd_data = (icd_data
                .sort(["FINNGENID", "DIAG_DATE"], descending=False)
                .filter((pl.col.DIAG_DATE==pl.col.DIAG_DATE.first()).over("FINNGENID"))
                .rename({"DIAG_DATE": "FIRST_ICD_DIAG_DATE"})
                .select(["FINNGENID", "FIRST_ICD_DIAG_DATE"])
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 ATC-based diagnoses                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    med_data = read_file(args.diags_path_start+"_meds.parquet")
    med_data = (med_data
                .sort(["FINNGENID", "MED_DATE"], descending=False)
                .filter((pl.col.MED_DATE==pl.col.MED_DATE.first()).over("FINNGENID"))
                .rename({"MED_DATE": "FIRST_MED_DIAG_DATE"})
                .select(["FINNGENID", "FIRST_MED_DIAG_DATE"])
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Join data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
    all_diags = all_data_diags.join(icd_data, on="FINNGENID", how="outer")
    all_diags = all_diags.join(med_data, on="FINNGENID", how="outer")
    # minimum date of diagnosis
    all_diags = (all_diags
                 .with_columns(pl.min_horizontal(
                                  pl.col("DATA_DIAG_DATE"), 
                                  pl.col("FIRST_ICD_DIAG_DATE"), 
                                  pl.col("FIRST_MED_DIAG_DATE")
                               ).alias("FIRST_DIAG_DATE")
                  )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
    logging.info("Data-based diagnosis: " + str(all_data_diags.shape[0]) + " rows")
    logging.info("ICD-based diagnosis: " + str(icd_data.shape[0]) + " rows")
    logging.info("ATC-based diagnosis: " + str(med_data.shape[0]) + " rows")
    logging.info("All diagnoses: " + str(all_diags.shape[0]) + " rows")
    logging.info("Took " + str(timer.get_time()) + " seconds to process all diagnoses.")

    all_diags.write_parquet(args.res_dir + args.file_name + "_diff" + args.diff_days + "_alldiags_" + get_date + ".parquet")