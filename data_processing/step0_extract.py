import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer, get_date, get_datetime, make_dir, init_logging, query_to_df
# Standard stuff
import pandas as pd
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

"""Get data for a specific OMOP concept ID from the kanta lab data. Returns a panda with data."""
def get_omop_id_data_bq(omop_concept_ids = ["3020564"],
                     table = f"`finngen-production-library.sandbox_tools_r12.kanta_r12_v1`",
                     columns = ["FINNGENID", "SEX", "EVENT_AGE", "TEST_OUTCOME", "TEST_OUTCOME_IMPUTED", "MEASUREMENT_VALUE_HARMONIZED",  "MEASUREMENT_UNIT_HARMONIZED", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT", "APPROX_EVENT_DATETIME"]):
    omop_concept_str = ",".join(f"'{id}'" for id in omop_concept_ids)
    col_str = ", ".join(columns)
    query = f""" SELECT {col_str}
                 FROM {table}
                 WHERE CAST({table}.OMOP_CONCEPT_ID as STRING) IN ({omop_concept_str}) AND (MEASUREMENT_VALUE is not NULL OR TEST_OUTCOME is not NULL)
                 ORDER BY FINNGENID, APPROX_EVENT_DATETIME
         	 """
    timer = Timer()
    data = query_to_df(query)
    logger.info("Time import: "+timer.get_elapsed())
    return(data)

"""Get data for a specific OMOP concept ID from the kanta lab data. Returns a panda with data."""
def get_orig_omop_id_data_parquet(omop_concept_id = "3020564",
                                  table = "/finngen/library-red/finngen_R13/kanta_lab_1.0/finngen_R12_kanta_lab_2.0-rc.1.parquet",
                                  columns = ["FINNGENID", "SEX", "EVENT_AGE", "APPROX_EVENT_DATETIME", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT",  "TEST_OUTCOME"]):
    data = pl.DataFrame(pd.read_parquet(table, 
                                        filers=[("OMOP_CONCEPT_ID", "==", omop_concept_id)],
                                        columns=columns))
    return(data)

"""Get data for a specific OMOP concept ID from the kanta lab data. Returns a panda with data."""
def get_extract_omop_id_data_parquet(omop_concept_id = "3020564",
                                     table = "/finngen/pipeline/DATA_IMPORT/fg-3/kanta/finngen_R12_kanta_lab_2.0-rc.1.parquet",
                                     columns = ["FINNGENID", "SEX", "EVENT_AGE", "APPROX_EVENT_DATETIME", "TEST_OUTCOME", "MEASUREMENT_VALUE_EXTRACTED", "IS_VALUE_EXTRACTED"]):
    data = pl.DataFrame(pd.read_parquet(table, 
                                        filers=[("OMOP_CONCEPT_ID", "==", omop_concept_id)],
                                        columns=columns))
    data = (data.filter(pl.col("IS_VALUE_EXTRACTED") == "1")
                .with_columns(MEASUREMENT_VALUE=pl.col("MEASUREMENT_VALUE_EXTRACTED"),
                              MEASUREMENT_UNIT=None,
                              MEASUREMENT_VALUE_HARMONIZED=None,
                              MEASUREMENT_UNIT_HARMONIZED=None,
                              TEST_OUTCOME_IMPUTED=None)
                .drop(["MEASUREMENT_VALUE_EXTRACTED", "IS_VALUE_EXTRACTED"]))
    return(data)

"""Setting up the parser arguments."""
def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--omop", help="OMOP ID to extract", required=True)
    parser.add_argument("--res_dir", help="Path to the results directory.", required=True)
    parser.add_argument("--lab_name", help="Readable name of the measurement value.", required=True)
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    #### Setup
    timer = Timer()
    args = get_parser_arguments()
    ## Paths
    file_name = args.lab_name + "_" + get_date()
    log_file_name = args.lab_name + "_" + get_datetime()
    make_dir(args.res_dir)
    ## Logging
    init_logging(args.res_dir, log_file_name, logger, args)
    
    #### Data processing
    ## Raw data
    data = get_orig_omop_id_data_parquet(omop_concept_ids=args.omop, 
                                         columns=["FINNGENID", "SEX", "EVENT_AGE", "TEST_OUTCOME", "TEST_OUTCOME_IMPUTED", "MEASUREMENT_VALUE_HARMONIZED", "MEASUREMENT_UNIT_HARMONIZED", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT", "APPROX_EVENT_DATETIME"])
    extract_data = get_extract_omop_id_data_parquet(omop_concept_ids=args.omop)
    extract_data = extract_data.select(data.columns)
    all_data = pl.concat([data, extract_data])
    ## Sorting
    data = data.sort(["FINNGENID", "APPROX_EVENT_DATETIME"], descending=True)
    data = data.rename({"APPROX_EVENT_DATETIME": "DATE", "MEASUREMENT_VALUE": "VALUE", "TEST_OUTCOME": "ABNORM", "MEASUREMENT_UNIT": "UNIT", "MEASUREMENT_VALUE_HARMONIZED": "VALUE_FG", "MEASUREMENT_UNIT_HARMONIZED": "UNIT_FG", "TEST_OUTCOME_IMPUTED":"ABNORM_FG"})
    data = data.select(["FINNGENID", "SEX", "EVENT_AGE", "DATE", "VALUE", "UNIT", "ABNORM", "VALUE_FG", "UNIT_FG", "ABNORM_FG"])

    ## Saving
    data.write_parquet(args.res_dir + file_name + ".parquet")
    logger.info("Time total: "+timer.get_elapsed())