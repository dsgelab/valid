# Utils
import re
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, logging_print, make_dir, gz_to_parquet
# Standard stuff
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime
# For file handling
import gzip
import csv
import re
from operator import itemgetter

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_ver", help="FinnGen release version", default="r12")
    parser.add_argument("--res_dir", type=str, help="Where data should be save", default="/home/ivm/valid/data/extra_data/processed_data/step0_extract/")
    parser.add_argument("--data_path", type=str, help="Path to data if not bigquery. Currently only supports gzip compressed tabular files.", default="")
    parser.add_argument("--in_col_names", nargs="+", type=str, default=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY", "CODE1"], help="Column names for the output file if not bigquery.'")
    parser.add_argument("--out_col_names", nargs="+", type=str, default=["FINNGENID", "EVENT_AGE", "DATE", "OPER_CODE"], help="Column names for the output file if not bigquery.")
    parser.add_argument("--file_delim", type=str, help="Delimiter for data file if not bigquery. Default is comma.", default=",")
    parser.add_argument("--regex", type=str, required=True, help="Regex for filtering opercodes.")
    parser.add_argument("--name", type=str, required=True, help="Name for the output file.")

    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))
     
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    
    log_dir = args.res_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.name + args.fg_ver + "_" + date
    log_file_name = args.name + args.fg_ver + "_" +date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    
    count = 0
    columns = []
    with gzip.open(args.data_path, "rt") as fin:
        with gzip.open(args.res_dir + file_name + ".tsv.gz", "wt") as fout:
            for line in csv.reader(fin, delimiter=args.file_delim.encode().decode('unicode_escape')):
                if count == 0: 
                    columns = line
                    fout.write("\t".join(args.out_col_names)+"\n")
                else:
                    good_source = re.match(r"(^OPER_IN)|(^OPER_OUT)", line[columns.index("SOURCE")])
                    good_code1 = re.match(args.regex, line[columns.index("CODE1")])
                    if good_source is not None and good_code1 is not None:
                        row_list = list(itemgetter(*[columns.index(name) for name in args.in_col_names])(line))
                        fout.write("\t".join(row_list)+"\n")
                count += 1
    logging_print("Time for processing file: "+timer.get_elapsed())
    gz_to_parquet(args.res_dir + file_name + ".tsv.gz", 
                  args.res_dir + file_name + ".parquet")