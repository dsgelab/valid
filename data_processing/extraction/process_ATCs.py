
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

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--file_path_atcs", type=str, help="Path to the data file", required=True)
    parser.add_argument("--min_pct", type=float, help="Path to the diagnosis data file", default="")
    parser.add_argument("--expansion", type=int, default=0, help="")
    # Settings
    parser.add_argument("--count_occ", type=int, default=0, help="")
    parser.add_argument("--fg_ver", type=str, default=0, help="")

    args = parser.parse_args()
    return(args)



#do the main file functions and runs 
if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    init_logging(args.res_dir, "ATC_CODE", logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Get data                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    if args.count_occ == 1: count_name = "sum"
    else: count_name = "bin"
    if args.expansion == 1: expanse = "ontexpand"
    else: expanse = "onttop"

    out_file_path = args.res_dir+"atcs_"+args.fg_ver+"_"+get_date()+"_min"+str(float(args.min_pct*100)).replace(".","p")+"pct_"+count_name+"_"+expanse+"_"+get_date()+".parquet"

    # Data in at least 1% of individuals
    if args.fg_ver=="r12":
        minimum_file_name = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_1.0.txt.gz"
    elif args.fg_ver == "r13":        
        minimum_file_name = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_1.0.txt.gz"

    ages = pl.read_csv(minimum_file_name,
                       separator="\t",
                       columns=["FINNGENID", "APPROX_BIRTH_DATE"])
    
    if args.fg_ver == "r12": N_total = ages.height 
    if args.fg_ver == "r13": N_total = ages.height
    atc_data = read_file(args.file_path_atcs)

    atc_data = atc_data.with_columns(pl.col.ATC_CODE.str.slice(0,5).alias("ATC_FIVE"))
    stats = (atc_data
             .group_by("ATC_FIVE")
             .agg(pl.len().alias("N_ENTRY"), 
                  pl.col.FINNGENID.unique().len().alias("N_INDV"))
             .with_columns((pl.col.N_INDV/N_total).alias("N_PERCENT"))
             .sort("N_INDV", descending=True)
    )
             
    atc_data = atc_data.filter(pl.col.ATC_FIVE.is_in(stats.filter(pl.col.N_PERCENT>=args.min_pct)["ATC_FIVE"]))
    print(atc_data)
    print(out_file_path)
    atc_data.write_parquet(out_file_path)
    stats.write_parquet(args.res_dir+"atcs_"+args.fg_ver+"_"+get_date()+"_counts.parquet")

# python3 /home/ivm/valid/data/extra_data/scripts/process/process_ATCs.py --res_dir /home/ivm/valid/data/extra_data/processed_data/step1_clean/ --file_path_atcs /home/ivm/valid/data/extra_data/processed_data/step0_extract/ATC_long_r13_2025-06-11.parquet --min_pct 0.01 --expansion 0 --count_occ 0 --fg_ver r13

# python3 /home/ivm/valid/data/extra_data/scripts/process/process_ATCs.py --res_dir /home/ivm/valid/data/extra_data/processed_data/step1_clean/ --file_path_atcs /home/ivm/valid/data/extra_data/processed_data/step0_extract/ATC_long_r12_2025-02-04.csv --min_pct 0.01 --expansion 0 --count_occ 0 --fg_ver r12