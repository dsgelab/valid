# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import logging_print, init_logging, make_dir, get_date, read_file, Timer
# Standard stuff
import numpy as np
import pandas as pd
import polars as pl
# Pickle
import pickle
# Time stuff
from datetime import datetime
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse

from collections import defaultdict
from tqdm import tqdm

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Data paths
    parser.add_argument("--res_dir", type=str, help="Path to the directory where results will be saved.", default="/home/ivm/valid/data/processed_data/step5_data/pytorch_ehr/")
    parser.add_argument("--data_path_dir", type=str, help="Path to the directory where data is stored.", required=True)
    parser.add_argument("--file_name_start", type=str, help="File name of data without the '.parquet' part", required=True)
    parser.add_argument("--icd_data_path", type=str, help="Path to data and labels without the '.parquet' part", required= False)
    parser.add_argument("--atc_data_path", type=str, help="Path to data and labels without the '.parquet' part", required= False)

    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--preds", type=str, help="List of predictors. Special options: ICD, ATC, LAB. Taking all columns from the respective data file.", 
                        default=["LAB"], nargs="+")
    parser.add_argument("--end_obs_date", type=str, help="Date of the last observation. If not provided, the date of the last observation in the data will be used.", default=None)
    parser.add_argument("--rep_codes", type=int, default=0, help="Whether to skip months with exact same codes as last.")
    parser.add_argument("--quant_step_size", type=float, default=None, help="Size for even quantile steps.")
    parser.add_argument("--quant_steps", type=int, nargs="+", default=None, help="Steps for quantiles")

    args = parser.parse_args()

    return(args)

def get_data(file_path_start: str,
             goal: str) -> pl.DataFrame:
    lab_data = read_file(file_path_start+".parquet")
    labels = read_file(file_path_start+"_labels.parquet")

    if "START_DATE" in labels.columns:
        labels = labels.with_columns(pl.col("START_DATE").cast(pl.Utf8).str.to_date())
    lab_data = lab_data.with_columns(pl.col("DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", exact=False))

    lab_data = lab_data.join(labels.select(["FINNGENID", goal, "SET"]), on="FINNGENID", how="full", coalesce=True)
    return(lab_data, labels)

pl.enable_string_cache()
def quantile_data(lab_name: str,
                  data: pl.DataFrame,
                  step_size: int=None,
                  quants: list[int]=None) -> pl.DataFrame:
    if quants is None:
        quants = np.append(np.quantile(data.filter((pl.col.SET==0)&(~pl.col.VALUE.is_null()))["VALUE"], np.arange(0, 1, step_size), method="higher"), 
                       data.filter(pl.col.SET==0)["VALUE"].max())
        logging_print("Quantiles: "+"  ".join([str(quant.round(2)) for quant in quants]))

    # Adding column with cut data
    data = data.with_columns(pl.when(pl.col("VALUE").is_not_null())
                                .then(pl.col.VALUE.cut(quants))
                                .alias("QUANT")
    )
    logging_print("Manual quantiles: "+"  ".join([str(quant) for quant in quants]))

    
    # Mapping quantiles to tokens
    quant_df = pl.DataFrame({"IDX": range(2, len(data["QUANT"].cat.get_categories())+2),
                             "QUANT": data["QUANT"].cat.get_categories()})
    quant_df = quant_df.with_columns([
                            pl.col("QUANT")
                               .map_elements(lambda interval_str: float(interval_str.replace("(", "").split(",")[0]),
                                            return_dtype=pl.Float64)
                               .alias("LOWER_EDGE"),
                            pl.col("QUANT")
                               .map_elements(lambda interval_str: float(interval_str.replace("]", "").split(",")[1]),
                                            return_dtype=pl.Float64)
                               .alias("UPPER_EDGE"),
    ])
    quant_df = quant_df.filter(~pl.col.LOWER_EDGE.is_infinite())
    quant_df = quant_df.filter(~pl.col.UPPER_EDGE.is_infinite())
    quant_df = quant_df.with_columns([(pl.col.LOWER_EDGE+((pl.col.UPPER_EDGE-pl.col.LOWER_EDGE)/2)).round().cast(pl.Int64).alias("MID")])
    print(data["VALUE"].min())
    print(data["VALUE"].max())
    quant_df = (quant_df
                .with_columns(pl.col.MID.map_elements(lambda x: "_".join([lab_name, str(x)]), 
                                                      return_dtype=pl.Utf8)
                                .alias("TEXT"))
               )
    data = (data
                 .join(quant_df.select(["QUANT", "TEXT"])
                 .with_columns(pl.col.QUANT.cast((pl.Categorical))), on="QUANT", how="left")
    )
    print(data["TEXT"].value_counts().sort("TEXT"))

    return data, quant_df

def get_icd_data(icd_data_path: str,
                 labels: pl.DataFrame) -> pl.DataFrame:
    icd_data = read_file(icd_data_path)
    icd_data = icd_data.cast({"DATE": pl.Date, "EVENT_AGE": pl.Float64})
    icd_data = icd_data.filter(pl.col("FINNGENID").is_in(labels["FINNGENID"].unique()))
    icd_data = icd_data.filter(pl.col("DATE").dt.year() >= 2012)
    icd_data = icd_data.sort(pl.col("DATE")).unique(subset=["DATE", "ICD_THREE"]).select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"])
    return(icd_data)

def get_atc_data(atc_data_path: str) -> pl.DataFrame:
    atc_data = read_file(atc_data_path)
    atc_data = atc_data.cast({"DATE": pl.Date, "EVENT_AGE": pl.Float64})
    atc_data = atc_data.filter(pl.col("FINNGENID").is_in(data["FINNGENID"].unique()))
    atc_data = atc_data.filter(pl.col("DATE").dt.year() >= 2012)
    atc_data = atc_data.sort(pl.col("DATE")).unique(subset=["DATE", "ATC_FIVE"]).select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"])
    return(atc_data)

def create_code_map(preds: list,
                    data: pl.DataFrame,
                    res_path_start: str,
                    icd_data=None,
                    atc_data=None) -> dict:
    if "SEX" in preds:
        list_codes = ["female", "male"]
    for code in list(data.filter(~pl.col.TEXT.is_null())["TEXT"].unique()):
        list_codes.append(code) 
    if "ICD" in preds:
        for code in list(icd_data.get_column("ICD_THREE").unique()): 
            list_codes.append(code)
    if "ATC" in preds:
        for code in list(atc_data.get_column("ATC_FIVE").unique()): 
            list_codes.append(code)
    code_map = dict(zip(list_codes, range(2, len(list_codes)+2)))
    print(code_map)
    pickle.dump(code_map, open(res_path_start + "_codemap.pkl", "wb"))
    return code_map

def merge_data(preds: list,
                lab_data: pl.DataFrame,
                icd_data: pl.DataFrame,
                atc_data: pl.DataFrame,
                end_obs_date=None,
                labels=None) -> pl.DataFrame:
    if "ICD" in preds and "ATC" in preds and "LAB" in preds:
        all_data = pl.concat([lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]), 
                            icd_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"}),
                            atc_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"]).rename({"ATC_FIVE":"TEXT"})])
    elif "ICD" in preds and "LAB" in preds:
        all_data = pl.concat([lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]), 
                            icd_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"})])
    elif "ATC" in preds and "LAB" in preds:
        all_data = pl.concat([lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]), 
                            atc_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"]).rename({"ATC_FIVE":"TEXT"})])
    elif "ICD" in preds:
        all_data = icd_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"})
    elif "ATC" in preds:
        all_data = atc_data.select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"]).rename({"ATC_FIVE":"TEXT"})
    elif "LAB" in preds:
        all_data = lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"])
    else:
        raise ValueError("No valid predictors selected.")
    
    if end_obs_date is None:
        all_data = all_data.join(pl.DataFrame(labels[["FINNGENID", "START_DATE"]]), how="left", on="FINNGENID")
        all_data = all_data.filter(pl.col("DATE") < pl.col("START_DATE").cast(pl.Date), pl.col("TEXT").is_not_null())
    else:
        all_data = all_data.filter(pl.col("DATE") < end_obs_date, pl.col("TEXT").is_not_null())

    all_data = (all_data
                .with_columns(pl.col("DATE")
                              .map_elements(lambda date: datetime.strptime(datetime.strftime(date, "%Y-%m"), "%Y-%m")))
               )
    return(all_data)
    
def get_list_data(fids: list, 
                  all_data: pl.DataFrame, 
                  labels: pl.DataFrame, 
                  code_map: dict,
                  preds: list,
                  end_obs_date=None,
                  skip_rep_codes=False) -> pl.DataFrame:
    if "SEX" in preds:
        sex = (labels.select(["FINNGENID", "SEX"])
                    .with_columns(pl.col.SEX.replace_strict(code_map).alias("SEX_TOKEN")))
        sex = dict(zip(sex.get_column("FINNGENID"), sex.get_column("SEX_TOKEN")))
    
    labels_dict = dict(zip(labels["FINNGENID"], labels["y_DIAG"]))
    start_age = dict(zip(labels["FINNGENID"], labels["EVENT_AGE"]))
    if end_obs_date is None:
        start_date = dict(zip(labels["FINNGENID"], labels["START_DATE"]))

    first_age = (all_data.select(["FINNGENID", "EVENT_AGE", "DATE"])
                         .sort(["FINNGENID", "EVENT_AGE", "DATE"], descending=False)
                         .group_by("FINNGENID")
                         .head(1)
                         .rename({"EVENT_AGE": "MIN_AGE", "DATE": "MIN_DATE"}))    
    out_data = defaultdict(list)    
    time_diffs = defaultdict(list)
    
    for fid in fids:
        out_data[fid] = [fid, labels_dict[fid], start_age[fid], sex[fid], []]
        if fid not in all_data["FINNGENID"]:
            out_data[fid][4].append([[0], [1]])
   
    all_data = all_data.filter(pl.col("FINNGENID").is_in(fids)).sort("DATE", descending=False).with_columns(pl.col("TEXT").replace_strict(code_map))
    for fid, crnt_data in tqdm(all_data.group_by("FINNGENID"), total=len(fids)):
        fid = fid[0]
        last_date = None
        time_diff = None
        last_codes = []
        start = True
        for date, date_data in crnt_data.sort("DATE", descending=False).group_by("DATE"):
            date = date[0]
            if not start: time_diff = np.round((date-last_date).days/30)
            else: 
                start = False
                time_diff = 0
            try:
                codes = list(set(date_data.get_column("TEXT").to_list()))
            except:
                codes = date_data.get_column("TEXT").to_list()
            if (not skip_rep_codes or last_codes != codes):
                last_date = date
                out_data[fid][4].append([[time_diff], codes])
                last_codes = codes                
        if end_obs_date is None:
            out_data[fid][4].append([[np.round((start_date[fid]-last_date).days/30)], [1]])
        else:
            out_data[fid][4].append([[np.round((end_obs_date-last_date).days/30)], [1]])
    return(out_data)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()

    # Setting up logging
    res_path_start = args.res_dir + args.file_name_start + "_long_" + "_".join(args.preds) 
    if args.quant_steps is not None: 
        res_path_start += "_manualquants"
    else:
        res_path_start += "_quantstep"+str(args.quant_step_size*10)
    if args.rep_codes == 1:
        res_path_start += "_norep" 
    res_path_start += "_" + get_date() 
        
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    lab_data, labels = get_data(file_path_start=args.data_path_dir+args.file_name_start, 
                                goal=args.goal)    
    lab_data, quant_df = quantile_data(args.lab_name, lab_data, args.quant_step_size, args.quant_steps)
    if "ICD" in args.preds: 
        icd_data = get_icd_data(args.icd_data_path, labels)
    else:
        icd_data = pl.DataFrame()
    if "ATC" in args.preds: 
        atc_data = get_atc_data(args.atc_data_path)
    else:
        atc_data = pl.DataFrame()
    code_map = create_code_map(args.preds, lab_data, res_path_start, icd_data, atc_data)
    
    if args.end_obs_date != "":
        args.end_obs_date = datetime.strptime(args.end_obs_date, "%Y-%m-%d")
        all_data = merge_data(args.preds, lab_data, icd_data, atc_data, end_obs_date=args.end_obs_date)
    else:
        all_data = merge_data(args.preds, lab_data, icd_data, atc_data, labels=labels)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Creating longitudinal                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    train_data = get_list_data(labels.filter(pl.col.SET==0).get_column("FINNGENID").unique(), 
                               all_data, 
                               labels, 
                               code_map,
                               args.preds,
                               args.end_obs_date,
                               args.rep_codes)
    pickle.dump(list(train_data.values()), open(res_path_start + "_train.pkl", "wb"), -1)
    test_data = get_list_data(labels.filter(pl.col.SET==2).get_column("FINNGENID").unique(), 
                            all_data, 
                            labels, 
                            code_map,
                               args.preds,
                               args.end_obs_date,
                               args.rep_codes)
    print(test_data)
    val_data = get_list_data(labels.filter(pl.col.SET==1).get_column("FINNGENID").unique(), 
                            all_data, 
                            labels, 
                            code_map,
                               args.preds,
                               args.end_obs_date,
                               args.rep_codes)
    pickle.dump(list(val_data.values()), open(res_path_start + "_valid.pkl", "wb"), -1)
    pickle.dump(list(test_data.values()), open(res_path_start + "_test.pkl", "wb"), -1)