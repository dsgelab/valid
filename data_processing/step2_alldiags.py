
# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file
from diag_utils import get_abnorm_start_dates, get_data_diags

# Standard stuff
import polars as pl
import argparse
# Logging
import logging
logger = logging.getLogger(__name__)



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
    all_data_diags = get_data_diags(data, args.diff_days)
    print(all_data_diags)

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
    all_diags = all_data_diags.join(icd_data, on="FINNGENID", how="full", coalesce=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 ATC-based diagnoses                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    try:
        med_data = read_file(args.diags_path_start+"_meds.parquet")
        med_data = (med_data
                    .sort(["FINNGENID", "MED_DATE"], descending=False)
                    .filter((pl.col.MED_DATE==pl.col.MED_DATE.first()).over("FINNGENID"))
                    .rename({"MED_DATE": "FIRST_MED_DIAG_DATE"})
                    .select(["FINNGENID", "FIRST_MED_DIAG_DATE"])
        )
        all_diags = all_diags.join(med_data, on="FINNGENID", how="full", coalesce=True)
    except FileNotFoundError:
        print("No medication-based diagnosis file found")
        all_diags = all_diags.with_columns(FIRST_MED_DIAG_DATE=None)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Join data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #       
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
    logging.info("Data-based diagnosis: " + str(all_data_diags.height) + " rows")
    logging.info("ICD-based diagnosis: " + str(icd_data.height) + " rows")
    try:
        logging.info("ATC-based diagnosis: " + str(med_data.height) + " rows")
    except:
        pass
    logging.info("All diagnoses: " + str(all_diags.height) + " rows")
    logging.info("Took " + str(timer.get_elapsed()) + " seconds to process all diagnoses.")

    (all_diags
         .select("FINNGENID", "FIRST_DIAG_DATE", "DATA_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE", "FIRST_ICD_DIAG_DATE", "FIRST_MED_DIAG_DATE")
         .unique()
         .write_parquet(args.res_dir + args.file_name + "_diff" + str(args.diff_days) + "_alldiags_" + get_date() + ".parquet")
    )