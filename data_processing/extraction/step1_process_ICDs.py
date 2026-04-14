
# Custom utils
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, init_logging, Timer, read_file, logging_print
from input_utils import get_min_file_path

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
    parser.add_argument("--file_path_icds", type=str, help="Path to the data file", required=True)
    parser.add_argument("--min_pct", type=float, help="Path to the diagnosis data file", default="")
    # Settings
    parser.add_argument("--fg_ver", type=str, default=0, help="")

    parser.add_argument("--min_age", type=int, default=18, help="")
    parser.add_argument("--max_age", type=int, default=70, help="")

    parser.add_argument("--memory_save_mode", type=int, default=0, help="Need for ML4Health data which does not run otherwise.")

    args = parser.parse_args()
    return(args)



#do the main file functions and runs 
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    init_logging(args.res_dir, "ICD", logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    out_file_path = args.res_dir+"icds_"+args.fg_ver+"_min"+str(int(args.min_pct*100)).replace(".","p")+"pct_"+str(args.min_age)+"t"+str(args.max_age)+"_"+get_date()+".parquet"

    minimum_file_path = get_min_file_path(args.fg_ver)

    if args.fg_ver != "ml4h":
        col_name = "FINNGENID"
    else:
        col_name = "FID"
    ages = pl.read_csv(minimum_file_path,
                        separator="\t" if args.fg_ver != "ml4h" else ",",
                        columns=[col_name])
    N_total = ages.height 
    icd_data = read_file(args.file_path_icds)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Process data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if "ICD_THREE" not in icd_data.columns:
        icd_data = icd_data.with_columns(pl.col("ICD_CODE").str.slice(0,3).alias("ICD_THREE"))
    if args.memory_save_mode:
        total_timer = Timer()
        all_stats = pl.DataFrame()
        all_icd_data = pl.DataFrame()

        icd_codes = sorted(set(icd_data["ICD_THREE"]))
        logging_print(f"Processing {len(icd_codes)} ICD codes one by one to save memory. Total individuals: {N_total}. Time for setup: {timer.get_elapsed()}")
        for crnt_icd in icd_codes:
            crnt_icd_data = (icd_data
                             .filter(pl.col.ICD_THREE == crnt_icd)
                             .filter(pl.col.EVENT_AGE >= args.min_age, pl.col.EVENT_AGE <= args.max_age) 
                             .drop("ICD_CODE")
                             .unique()
                             )
            crnt_stats = (crnt_icd_data
                          .with_columns(pl.col.FINNGENID.unique().len().alias("N_INDV"))
                          .with_columns((pl.col.N_INDV/N_total).alias("PCT_INDV"))
            )
            if crnt_icd_data.height > 0:
                if crnt_stats["PCT_INDV"][0] >= args.min_pct:
                    all_stats = pl.concat([all_stats, crnt_stats]) if all_stats.height > 0 else crnt_stats
                    all_icd_data = pl.concat([all_icd_data, crnt_icd_data]) if all_icd_data.height > 0 else crnt_icd_data
        logging_print(f"Finished processing all ICDs. Total time: {total_timer.get_elapsed()}")
    else:
        all_stats = (icd_data
                .filter(pl.col.EVENT_AGE >= args.min_age, pl.col.EVENT_AGE <= args.max_age) 
                .group_by("ICD_THREE")
                .agg(pl.len().alias("N_ENTRY"), 
                    pl.col.FINNGENID.unique().len().alias("N_INDV"))
                .with_columns((pl.col.N_INDV/N_total).alias("N_PERCENT"))
                .sort("N_INDV", descending=True)
        )      
        all_icd_data = icd_data.filter(pl.col.ICD_THREE.is_in(all_stats.filter(pl.col.N_PERCENT>=args.min_pct)["ICD_THREE"]))

    print(all_icd_data)
    print(out_file_path)
    all_icd_data.write_parquet(out_file_path)
    all_stats.write_parquet(out_file_path.replace(".parquet", "_counts.parquet"))