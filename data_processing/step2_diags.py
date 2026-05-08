 # Utils
import sys
sys.path.append(("../utils/"))
from general_utils import Timer, query_to_df, get_date, get_common_schema, align_schema, make_dir, init_logging, logging_print, read_file
# Standard stuff
import polars as pl
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse


def get_diag_med_data(diag_regex="",
                      med_regex="",
                      fg_ver = "R13"):
    """Get medication and diagnosis information based on the diagnosis and medication regex given."""

    if fg_ver == "R12":
        table = f"`finngen-production-library.sandbox_tools_r12.finngen_r12_service_sector_detailed_longitudinal_v1`"
    elif fg_ver == "R13":
        table = f"`finngen-production-library.sandbox_tools_r13.finngen_r13_service_sector_detailed_longitudinal_v1`"
    else:
        raise ValueError("Finngen version must be R12 or R13.")
    timer = Timer()

    if diag_regex != "":
        if med_regex != "":
            query = f"""SELECT FINNGENID, SOURCE, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2
                             FROM {table}
                             WHERE (((SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT", "REIMB") AND REGEXP_CONTAINS(CODE2, "{diag_regex}")) OR 
                                   (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT") AND REGEXP_CONTAINS(CODE1, "{diag_regex}"))) AND
                                   (NOT (REGEXP_CONTAINS(CATEGORY, "^ICP") OR REGEXP_CONTAINS(CATEGORY, "^OP") OR REGEXP_CONTAINS(CATEGORY, "^MOP") OR CATEGORY = "None"))) OR
                                   (SOURCE = "PURCH" AND REGEXP_CONTAINS(CODE1, "{med_regex}"))
                                   """
        else:
            query = f"""SELECT FINNGENID, SOURCE, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2
                             FROM {table}
                             WHERE (((SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT", "REIMB") AND REGEXP_CONTAINS(CODE2, "{diag_regex}")) OR 
                                   (SOURCE IN ("INPAT", "OUTPAT", "PRIM_OUT") AND REGEXP_CONTAINS(CODE1, "{diag_regex}"))) AND
                                   (NOT (REGEXP_CONTAINS(CATEGORY, "^ICP") OR REGEXP_CONTAINS(CATEGORY, "^OP") OR REGEXP_CONTAINS(CATEGORY, "^MOP") OR CATEGORY = "None")))
                                   """
    elif med_regex != "":
        print(med_regex)
        query = f"""SELECT FINNGENID, SOURCE, EVENT_AGE, APPROX_EVENT_DAY, CODE1, CODE2
                             FROM {table}
                             WHERE (SOURCE = "PURCH" AND REGEXP_CONTAINS(CODE1, "{med_regex}")) 
                                   """
    else:
        raise Exception("Have to either provide diag_regex or med_regex at minimum.")
    diags = query_to_df(query)

    # CODE1 is diagnosis and CODE2 symptom
    diags = diags.unpivot(index=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY", "SOURCE"])
    diags = diags.drop(["variable"]).unique()
    diags = diags.rename({"value":"CODE"})
    print(diags)
    # Other codes are numbers
    diags = diags.filter(pl.col("CODE").is_not_null())
    diags = diags.filter(pl.col("CODE").str.contains("^[A-Z][0-9]"))
    logging_print("Time import: "+timer.get_elapsed())
    return(diags)
    
def get_codes_first(data: pl.DataFrame,
                    crnt_regex: str) -> pl.DataFrame:
    """Get the first occurance of a code for each individual and code."""
    data = data.filter(pl.col("CODE").str.contains(crnt_regex))
    data = data.unique()
    data = data.filter((pl.col.APPROX_EVENT_DAY==pl.col.APPROX_EVENT_DAYS.min()).over(pl.col(["FINNGENID", "CODE"])))

    return(data)

def get_kidney_register_data(fg_ver = "R12"):
    """Get kidney register data."""
    if fg_ver == "R12":
        kd_data_path = "/finngen/library-red/finngen_R12/kidney_disease_register_1.0/data/finngen_R12_kidney_combined_1.0.txt"
    elif fg_ver == "R13":
        kd_data_path = "/finngen/library-red/finngen_R13/kidney_disease_register_1.0/data/finngen_R13_kidney_combined_1.0.txt"
    elif fg_ver == "R14":
        print("R14 kidney not yet processed (2026-04-14), using R13 instead.")
        kd_data_path = "/finngen/library-red/finngen_R13/kidney_disease_register_1.0/data/finngen_R13_kidney_combined_1.0.txt"
    else:
        raise ValueError("Finngen version must be R12, R13, or R14.")

    kd_data = (pl.read_csv(kd_data_path, separator="\t", infer_schema_length=100_000)
                 .select(["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY", "CURRENT_FORM_OF_TREATMENT"])
                 # All dialysis or transplant
                 .filter(pl.col.CURRENT_FORM_OF_TREATMENT.str.to_integer(strict=False)<52)
                 .filter(pl.col("EVENT_AGE").is_not_null())
                 # pivot longer to combine diagnosis columns
                 .unpivot(index=["FINNGENID", "EVENT_AGE", "APPROX_EVENT_DAY"], value_name="EXCL_CODE")
                 .drop(["variable"])
                 .rename({"APPROX_EVENT_DAY":"EXCL_DATE"})
                  # string to datetime
                 .with_columns(pl.col("EXCL_DATE").str.to_date("%Y-%m-%d", strict=False),
                               pl.col("EVENT_AGE").cast(pl.Float64))
                )

    return(kd_data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Info for reading and naming
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--fg_ver", type=str, help="Finngen version (R12 or R13)", required=True)

    parser.add_argument("--icd_file_path", type=str, help="Path to the diagnosis data file. If not given using bigquery.", default="")
    parser.add_argument("--atc_file_path", type=str, help="Path to the medication data file. If not given using bigquery.", default="")

    # Regex for selecting the data
    parser.add_argument("--diag_regex", type=str, help="Regex for selecting Code1 and 2 from the service sector detailed longitudinal data.", default="")
    parser.add_argument("--med_regex", type=str, help="Regex for selecting medication purchases.", required=False, default="")


    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    timer = Timer()
    args = get_parser_arguments()
    out_file_path = args.res_dir + args.lab_name + "_" + args.fg_ver + "_" + get_date()
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data from bigquery                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.icd_file_path == "" and args.atc_file_path == "":
        all_diags = get_diag_med_data(diag_regex=args.diag_regex, 
                                      med_regex=args.med_regex, 
                                      fg_ver=args.fg_ver)
        if args.med_regex != "":  
            med_diags = (get_codes_first(all_diags, args.med_regex)
                         .rename({"APPROX_EVENT_DAY": "MED_DATE", "CODE":"MED"}))
        if args.icd_regex != "":
            icd_diags = (get_codes_first(all_diags, args.icd_regex)
                         .rename({"APPROX_EVENT_DAY": "DIAG_DATE", "CODE":"DIAG"}))
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Data from long files                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    else:
        if args.atc_file_path != "" and args.med_regex !="":
            med_diags = read_file(args.atc_file_path)
            med_diags = med_diags.filter(pl.col("ATC_CODE").str.contains(args.med_regex))
            med_diags = med_diags.rename({"DATE": "MED_DATE", "ATC_CODE":"MED"})
        if args.icd_file_path != "" and args.diag_regex != "":
            icd_diags = read_file(args.icd_file_path).select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_CODE", "SOURCE"])
            icd_diags = icd_diags.filter(pl.col("ICD_CODE").str.contains(args.diag_regex))
            icd_diags = icd_diags.rename({"DATE": "DIAG_DATE", "ICD_CODE":"DIAG"})
            
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Kidney disease data                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.lab_name == "egfr" and args.fg_ver != "ml4h":
        kd_data = get_kidney_register_data(fg_ver=args.fg_ver).rename({"EXCL_DATE": "DIAG_DATE", "EXCL_CODE": "DIAG"})
        kd_data = kd_data.with_columns(pl.lit("KD_REGISTER").alias("SOURCE"))
        common_schema = get_common_schema([icd_diags, kd_data])
        icd_diags = pl.concat([align_schema(icd_diags, common_schema), align_schema(kd_data.select(icd_diags.columns), common_schema)])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Write output files                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.med_regex != "": 
        med_diags.write_parquet(out_file_path + "_meds.parquet")
    if args.diag_regex != "":
        icd_diags.write_parquet(out_file_path + "_diags.parquet")



    #Final logging
    logger.info("Time total: "+timer.get_elapsed())