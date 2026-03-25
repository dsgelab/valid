import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, make_dir, init_logging
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

"""Get data for a specific OMOP concept ID from the kanta lab data. Returns a panda with data."""
def get_orig_omop_id_data_parquet(omop_concept_id = "3020564",
                                  table = "/finngen/library-red/finngen_R13/kanta_lab_1.0/data/finngen_R13_kanta_lab_1.0.parquet",
                                  columns = ["FINNGENID", "SEX", "EVENT_AGE", "APPROX_EVENT_DATETIME", "MEASUREMENT_VALUE_HARMONIZED", "MEASUREMENT_VALUE_MERGED", "MEASUREMENT_UNIT_HARMONIZED",  "TEST_OUTCOME_IMPUTED"]):
    data = pl.DataFrame(pd.read_parquet(table, 
                                        filters=[("OMOP_CONCEPT_ID", "==", omop_concept_id)],
                                        columns=columns))
    return(data)

"""Setting up the parser arguments."""
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--omop", help="OMOP ID to extract", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory.", required=True)
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    parser.add_argument("--table_path", help="Path to data.", default="/finngen/library-red/finngen_R13/kanta_lab_1.0/data/finngen_R13_kanta_lab_1.0.parquet")

    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Setup
    timer = Timer()
    args = get_parser_arguments()
    ## Paths
    file_name = args.lab_name + "_" + get_date()
    make_dir(args.res_dir)
    ## Logging
    init_logging(args.res_dir, args.lab_name, logger, args)
    
    #### Data processing
    ## Raw data
    data = get_orig_omop_id_data_parquet(omop_concept_id=args.omop, 
                                         table=args.table_path,
                                         columns=["FINNGENID", "SEX", "EVENT_AGE",  "TEST_OUTCOME_IMPUTED", "MEASUREMENT_VALUE_MERGED", "MEASUREMENT_UNIT_HARMONIZED", "MEASUREMENT_VALUE_HARMONIZED", "APPROX_EVENT_DATETIME"],
                                        )
    ## Sorting
    data = data.sort(["FINNGENID", "APPROX_EVENT_DATETIME"], descending=True)
    data = data.rename({"APPROX_EVENT_DATETIME": "DATE",  "MEASUREMENT_VALUE_MERGED": "VALUE",  "MEASUREMENT_UNIT_HARMONIZED": "UNIT", "TEST_OUTCOME_IMPUTED":"ABNORM"})
    data = data.select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "UNIT", "ABNORM"])

    ## Saving
    data.write_parquet(args.res_dir + file_name + ".parquet")
    logger.info("Time total: "+timer.get_elapsed())