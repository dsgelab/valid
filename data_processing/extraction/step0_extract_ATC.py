# Utils
import re
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, logging_print, make_dir, query_to_df, gz_to_parquet
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


def get_med_data(file_path,
                  output_file_path,
                  med_regex="",
                  chunk_size=10_000_000,
                  total_rows=0):
    count = 0
    writer = None
    timer = Timer()

    for chunk in pd.read_csv(file_path,
                            compression="gzip",
                            chunksize=chunk_size,
                            dtype={"CODE1": str, "SOURCE": str, "EVENT_AGE": float, "FID": str},
                            usecols=["FID", "SOURCE", "CODE1",  "EVENT_AGE", "EVENT_DAY"]):
        
        mask = ((chunk["SOURCE"] == "PURCH") & chunk["CODE1"].str.contains(med_regex, regex=True, na=False))
        chunk = chunk[mask].copy()
        chunk = chunk[["FID", "EVENT_AGE", "EVENT_DAY", "SOURCE", "CODE1"]]
        chunk = chunk.rename(columns={"FID": "FINNGENID", "CODE1": "CODE"})
        chunk["EVENT_DAY"] = pd.to_datetime(chunk["EVENT_DAY"]).dt.date
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(output_file_path, table.schema)
        try:
            writer.write_table(table)
        except Exception as e:
            print(chunk)
            print(table)
            print(e)

        count += chunk_size
        if count % 100_000_000 == 0: print(f"Processed {count} rows. Time elapsed: {timer.get_elapsed()}")
        if total_rows > 0 and count >= total_rows: 
            logging_print(f"Finished writing N={total_rows}. Total time: {timer.get_elapsed()}")
            writer.close()
            return

    if writer:
        logging_print(f"Finished writing. Total time: {timer.get_elapsed()} and total rows: {count}")
        writer.close()


def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--fg_ver", help="FinnGen release version", default="r12")
    parser.add_argument("--res_dir", type=str, help="Where data should be save", default="/home/ivm/valid/data/extra_data/processed_data/step0_extract/")
    parser.add_argument("--data_path", type=str, help="Path to data if not bigquery. Currently only supports gzip compressed tabular files.", default="")
    parser.add_argument("--in_col_names", nargs="+", type=str, default=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY", "ICD_CODE", "CATEGORY", "ICDVER", "SOURCE"], help="Column names for the output file if not bigquery.'")
    parser.add_argument("--out_col_names", nargs="+", type=str, default=["FINNGENID", "EVENT_AGE", "DATE", "ICD_CODE", "CATEGORY",  "ICD_VERSION", "SOURCE"], help="Column names for the output file if not bigquery.")
    parser.add_argument("--file_delim", type=str, help="Delimiter for data file if not bigquery. Default is comma.", default=",")

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
    file_name = "ATC_long_" + args.fg_ver + "_" + date
    log_file_name = "ATC_long_" + args.fg_ver + "_" +date_time
    make_dir(args.res_dir)
    make_dir(log_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    
    if args.data_path=="":
        table = f"`finngen-production-library.sandbox_tools_{args.fg_ver}.finngen_{args.fg_ver}_service_sector_detailed_longitudinal_v1`"
        query_pat = f"""SELECT FINNGENID, EVENT_AGE, APPROX_EVENT_DAY, CODE1
                        FROM {table}
                        WHERE SOURCE = 'PURCH' AND CODE1 != 'NA'
                    """
        all_atcs = query_to_df(query_pat)
        logger.info("Time import: "+timer.get_elapsed())
        print(all_atcs)
        all_atcs=all_atcs.rename({"CODE1": "ATC_CODE", "APPROX_EVENT_DAY": "DATE"}).unique()
        print(all_atcs)
        all_atcs.write_parquet(args.res_dir + file_name + ".parquet")
        init_logging(f"Number of rows {all_atcs.height} with {all_atcs["FINNGENID"].unique().len()} individuals.")
    else:
        count = 0
        columns = []
        with gzip.open(args.data_path, "rt") as fin:
            with gzip.open(args.res_dir + file_name + ".tsv.gz", "wt") as fout:
                for line in csv.reader(fin, delimiter=args.file_delim.encode().decode('unicode_escape')):
                    if count == 0: 
                        columns = line
                        fout.write("\t".join(args.out_col_names)+"\n")
                    else:
                        good_source = re.match(r"^PURCH", line[columns.index("SOURCE")])
                        good_code1 = re.match(r"^[A-Z][0-9]+", line[columns.index("CODE1")])
                        if good_source is not None and good_code1 is not None:
                            row_list = list(itemgetter(*[columns.index(name) for name in args.in_col_names])(line))
                            if args.out_col_names[-1] == "ATC_FIVE":
                                row_list.append(line[columns.index("CODE1")][0:5])
                            fout.write("\t".join(row_list)+"\n")
                    count += 1
        logging_print("Time for processing file: "+timer.get_elapsed())
        gz_to_parquet(args.res_dir + file_name + ".tsv.gz", 
                      args.res_dir + file_name + ".parquet")