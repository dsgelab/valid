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
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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
    parser.add_argument("--extra_lab_data_path", type=str, help="Path to full kanta lab data.", default="")
    parser.add_argument("--extra_omop_ids", type=str, help="List of OMOP IDs to add from kanta lab file..", 
                        required=False, nargs="+")
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--preds", type=str, help="List of predictors. Special options: ICD, ATC, LAB. Taking all columns from the respective data file.", 
                        default=["LAB"], nargs="+")
    parser.add_argument("--preds_name", type=str, help="Name for list of predictors.") 
    parser.add_argument("--end_obs_date", type=str, help="Date of the last observation. If not provided, the date of the last observation in the data will be used.", default=None)
    parser.add_argument("--skip_rep_codes", type=int, default=0, help="Whether to skip months with exact same codes as last.")
    parser.add_argument("--quant_step_size", type=float, default=None, help="Size for even quantile steps.")
    parser.add_argument("--quant_steps", type=float, nargs="+", default=None, help="Steps for quantiles")
    parser.add_argument("--time_detail", type=str, default="month")

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
    data = data.with_columns(pl.col.VALUE.cast(pl.Float64).alias("VALUE"))
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

def quantile_all_lab_data(data: pl.DataFrame,
                          lab_name: str,
                           omop_ids):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Removing duplicate predictors                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    if args.lab_name == "krea" or args.lab_name == "egfr":
        # krea, egfr
        data = data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["40764999", "3020564"]))
    if args.lab_name == "hba1c":
        data = data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3004410"]))
    if args.lab_name == "alatasat":
        # ALAT, ASATs
        data = data.filter(~pl.col.OMOP_CONCEPT_ID.is_in(["3006923", "3013721"]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Quantiling separately                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    for omop_id in data["OMOP_CONCEPT_ID"].unique():
        if omop_id in omop_ids:
            crnt_lab_data = data.filter(pl.col.OMOP_CONCEPT_ID == omop_id)
            try:
                crnt_lab_data, crnt_quant_df = quantile_data(omop_id, crnt_lab_data, step_size=0.25)
            except:
                try:
                    crnt_lab_data, crnt_quant_df = quantile_data(omop_id, crnt_lab_data, step_size=0.9)
                except:
                    crnt_lab_data, crnt_quant_df = quantile_data(omop_id, crnt_lab_data, step_size=0.99)
            if "all_lab_data" in locals():
                all_lab_data = pl.concat([all_lab_data, crnt_lab_data])
                all_quant_df = pl.concat([all_quant_df, crnt_quant_df])
            else:
                all_lab_data = crnt_lab_data
                all_quant_df = crnt_quant_df
    
    return all_lab_data, all_quant_df

def get_icd_data(icd_data_path: str,
                 labels: pl.DataFrame) -> pl.DataFrame:
    icd_data = read_file(icd_data_path)
    print(icd_data)
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
                labels=None,
                time_detail: str="months") -> pl.DataFrame:
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

    if time_detail=="months":
        all_data = (all_data.with_columns(pl.col("DATE").dt.strftime("%Y-%m").str.to_date("%Y-%m").alias("DATE")))
    return(all_data)
    
def get_list_data(fids: list, 
                  all_data: pl.DataFrame, 
                  labels: pl.DataFrame, 
                  code_map: dict,
                  age_map: dict,
                  preds: list,
                  end_obs_date=None,
                  skip_skip_rep_codes=False,
                  goal: str="y_DIAG",
                 time_detail: str="months") -> pl.DataFrame:
    if "SEX" in preds:
        sex = (labels.select(["FINNGENID", "SEX"])
                    .with_columns(pl.col.SEX.replace_strict(code_map).alias("SEX_TOKEN")))
        sex = dict(zip(sex.get_column("FINNGENID"), sex.get_column("SEX_TOKEN")))
    
    labels_dict = dict(zip(labels["FINNGENID"], labels[goal]))
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
        out_data[fid] = [fid, labels_dict[fid], age_map[fid], sex[fid], []]
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
            if not start: 
                time_diff = (date-last_date).days
                if time_detail=="months": time_diff = np.round(time_diff/30)
            else: 
                start = False
                time_diff = 0
            try:
                codes = list(set(date_data.get_column("TEXT").to_list()))
            except:
                codes = date_data.get_column("TEXT").to_list()
            if (not skip_skip_rep_codes or last_codes != codes):
                last_date = date
                out_data[fid][4].append([[time_diff], codes])
                last_codes = codes                
        if end_obs_date is None:
            crnt_time_diff = (start_date[fid]-last_date).days
            if time_detail=="months": crnt_time_diff = np.round(crnt_time_diff/30)
            out_data[fid][4].append([[crnt_time_diff], [1]])
        else:
            crnt_time_diff = (end_obs_date-last_date).days
            if time_detail=="months": crnt_time_diff = np.round(crnt_time_diff/30)
            out_data[fid][4].append([[crnt_time_diff], [1]])
    return(out_data)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    timer = Timer()
    args = get_parser_arguments()

    # Setting up logging
    res_path_start = args.res_dir + args.file_name_start + "_long_" + args.preds_name 
    if args.quant_steps is not None: 
        res_path_start += "_manualquants"
    else:
        res_path_start += "_quantstep"+str(args.quant_step_size*10)
    if args.skip_rep_codes == 1:
        res_path_start += "_norep" 
    res_path_start += "_" + args.time_detail + "_" + get_date() 
        
    make_dir(args.res_dir)
    init_logging(args.res_dir, args.lab_name, logger, args)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    lab_data, labels = get_data(file_path_start=args.data_path_dir+args.file_name_start, 
                                goal=args.goal) 
    lab_data, quant_df = quantile_data(args.lab_name, lab_data, args.quant_step_size, args.quant_steps)
    if args.extra_lab_data_path != "":
        extra_lab_data = read_file(args.extra_lab_data_path)
        extra_lab_data = extra_lab_data.rename({"MEASUREMENT_VALUE_HARMONIZED": "VALUE"})
        extra_lab_data = extra_lab_data.with_columns(pl.col("APPROX_EVENT_DATETIME").cast(pl.Utf8).str.to_date("%Y-%m-%d", exact=False).alias("DATE"),
                                                    pl.col.EVENT_AGE.cast(pl.Float64).alias("EVENT_AGE"))

        extra_lab_data = extra_lab_data.join(labels.select(["FINNGENID", args.goal, "SET"]), on="FINNGENID", how="left", coalesce=True)
        extra_lab_data, all_quant_df = quantile_all_lab_data(extra_lab_data, args.lab_name, args.extra_omop_ids)
        lab_data = pl.concat([lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]), extra_lab_data.select(["FINNGENID", "EVENT_AGE", "DATE", "TEXT"])])   
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
        args.end_obs_date = datetime.strptime(args.end_obs_date, "%Y-%m-%d").date()
        all_data = merge_data(args.preds, lab_data, icd_data, atc_data, end_obs_date=args.end_obs_date, time_detail=args.time_detail)
    else:
        all_data = merge_data(args.preds, lab_data, icd_data, atc_data, labels=labels, time_detail=args.time_detail)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Creating longitudinal                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    age_scaler = RobustScaler()
    #age_scaler.set_output(transform="polars")
    print(labels.filter(pl.col.SET==0).get_column("EVENT_AGE").to_numpy())
    ages_scaled = age_scaler.fit_transform(labels.filter(pl.col.SET==0).get_column("EVENT_AGE").to_numpy().reshape(-1,1)).flatten()
    print(ages_scaled)
    age_map = dict(zip(labels.filter(pl.col.SET==0).get_column("FINNGENID"), ages_scaled))
    train_data = get_list_data(labels.filter(pl.col.SET==0).get_column("FINNGENID").unique(), 
                               all_data, 
                               labels, 
                               code_map,
                               age_map,
                               args.preds,
                               args.end_obs_date,
                               args.skip_rep_codes,
                               args.goal,
                              args.time_detail)
    pickle.dump(list(train_data.values()), open(res_path_start + "_train.pkl", "wb"), -1)

    ages_scaled = age_scaler.transform(labels.filter(pl.col.SET==2).get_column("EVENT_AGE").to_numpy().reshape(-1,1)).flatten()
    age_map = dict(zip(labels.filter(pl.col.SET==2).get_column("FINNGENID"), ages_scaled))
    test_data = get_list_data(labels.filter(pl.col.SET==2).get_column("FINNGENID").unique(), 
                            all_data, 
                            labels, 
                            code_map,
                              age_map,
                               args.preds,
                               args.end_obs_date,
                               args.skip_rep_codes,
                             args.goal,
                              args.time_detail)
    ages_scaled = age_scaler.transform(labels.filter(pl.col.SET==1).get_column("EVENT_AGE").to_numpy().reshape(-1,1)).flatten()
    age_map = dict(zip(labels.filter(pl.col.SET==1).get_column("FINNGENID"), ages_scaled))
    val_data = get_list_data(labels.filter(pl.col.SET==1).get_column("FINNGENID").unique(), 
                            all_data, 
                            labels, 
                            code_map,
                             age_map,
                               args.preds,
                               args.end_obs_date,
                               args.skip_rep_codes,
                            args.goal,
                              args.time_detail)
    pickle.dump(list(val_data.values()), open(res_path_start + "_valid.pkl", "wb"), -1)
    pickle.dump(list(test_data.values()), open(res_path_start + "_test.pkl", "wb"), -1)