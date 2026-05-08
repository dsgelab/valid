import polars as pl
import sys
sys.path.append("../../utils/")
from general_utils import read_file, print_count
from datetime import datetime
import gc
def get_data_and_pred_list(file_path_labels: str, 
                           file_path_icds: str,
                           file_path_atcs: str,
                           file_path_sumstats: str,
                           file_path_second_sumstats: str,
                           file_path_labs: str,
                           lab_name: str,
                           clean: int,
                           file_path_pgs1: str,
                           file_path_pgs2: str,
                           file_path_transformer: str,
                           preds: list,
                           start_date: str,
                           needed_X_cols: list=[],
                           fill_missing: int=0,
                           fids_path: str="",
                           future_val: str="",
                           fg_ver: str=None) -> tuple[pl.DataFrame, list]:
    """Creates the data and predictor list. 

   Returns the data and the predictors.
   Args:
        file_path_labels (str): Path to the labels file.
        file_path_icds (str): Path to the ICD codes file.
        file_path_atcs (str): Path to the ATC codes file.
        file_path_sumstats (str): Path to the sumstats file.
        file_path_second_sumstats (str): Path to the second sumstats file.
        file_path_labs (str): Path to the labs file.
        preds (list): List of predictors.
        start_date (str): Start date for the data.
        
   Returns:
        tuple: A tuple containing the data (polars DataFrame) and the list of predictors
            (data, preds)"""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting labels                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(file_path_labels)

    # # # # # # # # # Filtering # # # # # # # # # # # # # # # # # # #
    if future_val == "final train":
        # Using valid&test as train for final fit, so filtering out train set and setting valid&test to 0
        data = data.filter(pl.col("SET").is_in([1,2])).with_columns(pl.Series("SET", [0]*data.height))
    elif future_val == "train" : data = data.filter(pl.col.SET==0)
    elif future_val == "val": data = data.filter(pl.col.SET!=0)
    # For Final val actually being passed a different df so no fitering needed
    
    if fids_path != "": 
        fids = read_file(fids_path)
        print_count(data)
        data = data.filter(pl.col.FINNGENID.is_in(fids["FINNGENID"]))
    print_count(data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting predictors                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data, trans_cols = get_transformer_data(data, file_path_transformer); gc.collect()
    data = get_pgs_data(data, file_path_pgs1, file_path_pgs2); gc.collect()
    data, icd_cols = get_icd_atc_data(data, file_path_icds); gc.collect()
    data, atc_cols = get_icd_atc_data(data, file_path_atcs); gc.collect()
    data, sumstats_cols = get_sumstats_data(data, start_date, file_path_sumstats, is_second=False)
    data, second_sumstats_cols = get_sumstats_data(data, start_date, file_path_second_sumstats, is_second=True)
    data, labs_cols = get_labs_data(data, lab_name, clean, fg_ver, file_path_labs)
    data = get_other_preds(data, preds, start_date, fill_missing, fg_ver)

    X_cols = get_col_list(preds, 
                          icd_cols, 
                          atc_cols,
                          sumstats_cols, 
                          second_sumstats_cols, 
                          labs_cols, 
                          trans_cols)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Memory saving and other fixes                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    data = fix_sex_col(data)

    # # # # # # # # # Smaller dtypes # # # # # # # # # # # # # # # # # # #
    binary_cols = ([col_name for col_name in X_cols
                       if(set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {0, 1} or
                          set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {"0", "1"})  
    ])
    data = data.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col, dtype in data.schema.items() if (isinstance(dtype, pl.Int64) or isinstance(dtype, pl.Int32)) and col not in binary_cols
    ])
    data = data.with_columns([
        pl.col(col).fill_nan(None).cast(pl.Int8, strict=False) for col in data.columns if col in binary_cols
    ])
    
    # # # # # # # # # Adding needed empty columns missing # # # # # # # # # # # # # # # # # # #
    for crnt_col in needed_X_cols:
        if crnt_col not in data.columns:
            data = data.with_columns(pl.Series(crnt_col, [0]*data.height))

    return(data.unique(), X_cols)

def get_min_file_path(fg_ver: str) -> str:
    """Returns the file path to the minimum data file based on the FinnGen version.
    Args:
        fg_ver (str): The FinnGen version, used to get the correct minimum data file path. If R12, R13, or R14, uses the respective R12, R13, or R14 minimum data file. Otherwise, uses the ML4H minimum data file.
    Returns:
        str: The file path to the minimum data file.
    """

    if fg_ver == "R12" or fg_ver == "r12":
        minimum_file_path = "/finngen/library-red/finngen_R12/phenotype_1.0/data/finngen_R12_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R13" or fg_ver == "r13":        
        minimum_file_path = "/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_extended_1.0.txt.gz"
    elif fg_ver == "R14" or fg_ver == "r14":
        minimum_file_path = "/finngen/library-red/finngen_R14/phenotype_1.0/data/finngen_R14_minimum_extended_1.0.txt.gz"
    elif fg_ver == "ML4H" or fg_ver == "ml4h" or fg_ver=="ML4Health" or fg_ver=="ml4health":
        minimum_file_path = "/finngen/red/ml4health/processed/main_modalities/DVV_processed.csv"

    return(minimum_file_path)

import polars as pl
def get_all_indvs(fg_ver: str) -> pl.DataFrame:
    """Returns a DataFrame containing all individuals in the minimum data file based on the FinnGen version, with a column "FINNGENID" containing the individual IDs.
    Args:
        fg_ver (str): The FinnGen version, used to get the correct minimum data file path. If R12, R13, or R14, uses the respective R12, R13, or R14 minimum data file. Otherwise, uses the ML4H minimum data file. 
    Returns:
        pl.DataFrame: A DataFrame containing all individuals in the minimum data file, with a column "FINNGENID" containing the individual IDs.
    """

    min_file_path = get_min_file_path(fg_ver)
    if fg_ver != "ml4h":
        col_name = "FINNGENID"
    else:
        col_name = "FID"
    all_indvs = pl.read_csv(min_file_path,  
                            separator="\t" if fg_ver != "ml4h" else ",",
                            columns=[col_name])
    if "FID" in all_indvs.columns:
        all_indvs = all_indvs.rename({"FID": "FINNGENID"})

    return(all_indvs)

def duplicate_lab_predictors(lab_name: str, 
                             clean: int,
                             fg_ver: str="R14") -> list:
    """Returns a list of OMOP IDs that are similar to the lab we are using as predictor, based on the lab name, the clean version, and the FinnGen version. Used to filter out these OMOP IDs from the labs data to avoid data leakage.
    Args:
        lab_name (str): The name of the lab, used to determine which OMOP IDs to filter out. Can be "krea", "egfr", "hba1c", "tsh", or "ldl".
        clean (int): The clean version, used to determine which OMOP IDs to filter out. If 1, filters out more OMOP IDs to be more strict. If 0, filters out less OMOP IDs to be more lenient.
        fg_ver (str): The FinnGen version, used to determine which OMOP IDs to filter out for certain labs. If ML4H, filters out less OMOP IDs for hba1c and tsh to be more lenient, as the ML4H data is already very clean. For other versions, filters out more OMOP IDs for hba1c and tsh to be more strict, as the data is less clean.
    Returns:
        list: A list of OMOP IDs that are similar to the lab we are using as predictor, based on the lab name, the clean version, and the FinnGen version. Used to filter out these OMOP IDs from the labs data to avoid data leakage.
    """

    if lab_name == "krea" or lab_name == "egfr":
        if clean == 1:
            # krea, egfr some other formula, egfr, cystatin c, UACR
            not_select_omops = ["40764999", "46236952", "3020564", "3030366", "3020682"]
        else:
            not_select_omops = ["40764999", "46236952", "3020564"]
    if lab_name == "hba1c":
        if fg_ver != "ml4h":
            if clean == 1:
                not_select_omops = ["3004410", "3018251", "3013826", "3013826"]# hba1c and fasting glucose, glucose, glucose 2 hours post dose
            else:
                not_select_omops = ["3004410", "3018251"] # hba1c and fasting glucose
        else:
            not_select_omops = ["3004410"]
    if lab_name == "tsh":
        if fg_ver != "ml4h":
            if clean == 1:
                not_select_omops = ["3009201", "3008486", "3026989"] # tsh, t4, t3
            else:
                not_select_omops = ["3009201", "3008486"]
        else:
            not_select_omops = ["3009201"]
    if lab_name == "ldl":
        if fg_ver != "ml4h":
            if clean == 1: 
                # LDL, LDL/HDL, HDl standard, HDL total, cholesterol total, weird cholesterol non hdl rare, tri fast, tri
                not_select_omops = ["3001308", "3019900", "3023602", "42868674","3019900", "3048773", "3025839"]
            else:
                # LDL, tryg free, tryg
                not_select_omops = ["3001308", "3048773", "3025839"]
        else:
            not_select_omops = ["3001308"]

    return not_select_omops

import polars as pl
def filter_out_no_pcs(data: pl.DataFrame) -> pl.DataFrame:
    """Filters out individuals without PCs from the data, and prints the number of individuals left.
    Args:
        data (pl.DataFrame): The data to filter.
    Returns:        
        pl.DataFrame: The data with individuals without PCs filtered out.
    """

    pcs = pl.read_csv("/finngen/library-red/finngen_R13/analysis_covariates/data/R13_COV_PHENO_V0.FID.txt.gz", separator="\t", columns=["IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"])
    data = data.join(pcs.select("IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"), left_on="FINNGENID", right_on="IID", how="left")
    print("Number of individuals without PCs")
    print_count(data.filter(pl.col.PC1.is_null()))
    print("Number of individuals left")
    print_count(data.filter(~pl.col.PC1.is_null()))
    data = data.filter(~pl.col.PC1.is_null())

    return(data)

import polars as pl
def get_transformer_data(data: pl.DataFrame, 
                         file_path_transformer: str) -> pl.DataFrame:
    """Returns the data with the transformer columns added if the file path is provided, and the list of columns.
    Args:
        data (pl.DataFrame): The data to which the transformer columns will be added.
        file_path_transformer (str): The file path to the transformer file.
    Returns:
        tuple: A tuple containing the data with the transformer columns added and the list of transformer columns
    """

    if file_path_transformer != "": 
        trans_data = read_file(file_path_transformer)
        data = data.join(trans_data.drop("SET", "y_MEAN_ABNORM"), on="FINNGENID", how="left")
        trans_cols = trans_data.columns
        
    return data, trans_cols

import polars as pl
def get_pgs_data(data: pl.DataFrame, 
                 file_path_pgs1: str, 
                 file_path_pgs2: str) -> pl.DataFrame:
    """Returns the data with the PGS columns added if the file paths are provided."""

    if file_path_pgs1 != "":
        pgs = pl.read_csv(file_path_pgs1, separator="\t")
        if len(pgs.columns)==2: # Zhijian PGS for eGFR levels
            pgs.columns = ["FINNGENID", "PGS1"]
        else: # Zhiyu PGS for creatinine slopes
            pgs.columns = ["X1", "FINNGENID", "X2", "X3", "PGS1"] 
        data = data.join(pgs.select("FINNGENID", "PGS1"), on="FINNGENID", how="left")
        data = filter_out_no_pcs(data)

    if file_path_pgs2 != "":
        pgs = pl.read_csv(file_path_pgs2, separator="\t")
        if len(pgs.columns)==2: # Zhijian PGS for eGFR levels
            pgs.columns = ["FINNGENID", "PGS2"]
        else: # Zhiyu PGS for creatinine slopes
            pgs.columns = ["X1", "FINNGENID", "X2", "X3", "PGS2"] 
        data = data.join(pgs.select("FINNGENID", "PGS2"), on="FINNGENID", how="left")
        data = filter_out_no_pcs(data)

    return(data)

import polars as pl
def get_icd_atc_data(data: pl.DataFrame, 
                     file_path_preds: str) -> tuple[pl.DataFrame, list]:
    """Returns the data with the ICD/ATC columns added if the file path is provided, and the list of columns.

    Makes sure that only the columns with at least 5 cases to avoid individual level data.

    Args:
        data (pl.DataFrame): The data to which the ICD/ATC columns will be added.
        file_path_icds (str): The file path to the ICD codes file.
    Returns:
        tuple: A tuple containing the data with the ICD/ATC columns added and the list of ICD/ATC columns
    """

    if file_path_preds != "": 
        preds = read_file(file_path_preds).filter(pl.col.FINNGENID.is_in(data["FINNGENID"]))
        pred_cols = [c for c in preds.columns if c != "FINNGENID"]
        # count 1s per ICD column
        pred_counts = preds.select(pred_cols).sum()  # one-row df with sum per column
        # keep columns meeting threshold
        keep = [c for c in pred_cols if pred_counts[c].item() >= 5]
        preds = preds.select(["FINNGENID", *keep])

        data = data.join(preds, on="FINNGENID", how="left")
        pred_cols = preds.columns
        
    return data, pred_cols

import polars as pl
def get_sumstats_data(data: pl.DataFrame,
                      start_date: str,
                      file_path_sumstats: str,
                      is_second: bool=False) -> tuple[pl.DataFrame, list]:
    """Returns the data with the sumstats columns added if the file path is provided, and the list of columns.
    
    Args:
        data (pl.DataFrame): The data to which the sumstats columns will be added.
        start_date (str): The start date for the data, in the format "YYYY-MM-DD". 
                        Used to calculate the difference between the last value date and the start date.
        file_path_sumstats (str): The file path to the sumstats file.
        is_second (bool): Whether to process the second sumstats file. Which means renaming the columns with S_ and calculating the difference to start date with S_LAST_VAL_DATE. Default is False.

    Returns:
        tuple: A tuple containing the data with the sumstats columns added and the list of columns

    """

    if file_path_sumstats != "": 
        sumstats = read_file(file_path_sumstats,
                             schema={"SEQ_LEN": pl.Float64,
                                     "MIN_LOC": pl.Float64,
                                     "MAX_LOC": pl.Float64,
                                     "FIRST_LAST": pl.Float64})
        if "SET" in sumstats.columns: sumstats = sumstats.drop("SET") # dropping duplicate info on set in sumstats if present
        if is_second:
            sumstats = sumstats.rename({col: f"S_{col}" for col in sumstats.columns if col != "FINNGENID"})
        data = data.join(sumstats, on="FINNGENID", how="left")
        if start_date != "":
            if is_second:
                data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.S_LAST_VAL_DATE).dt.total_days().alias("S_LAST_VAL_DIFF"))
            else:
                data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.LAST_VAL_DATE).dt.total_days().alias("LAST_VAL_DIFF"))
        sumstats_cols = sumstats.columns

    return data, sumstats_cols

import polars as pl
def get_labs_data(data: pl.DataFrame,
                  lab_name: str,
                  clean: int,
                  fg_ver: str, 
                  file_path_labs: str) -> tuple[pl.DataFrame, list]:
    """Returns the data with the lab columns added if the file path is provided, and the list of columns.

    Makes sure that only the columns with at least 5 cases to avoid individual level data.

    Args:
        data (pl.DataFrame): The data to which the lab columns will be added.
        file_path_labs (str): The file path to the labs file.
    Returns:
        tuple: A tuple containing the data with the lab columns added and the list of lab columns
    """

    if file_path_labs != "": 
        labs = read_file(file_path_labs)
        # do not select OMOPs that are similar to the lab we are using as predictor to avoid data leakage
        not_select_omops = duplicate_lab_predictors(lab_name=lab_name, clean=clean, fg_ver=fg_ver)
        # labs is wider format so need to take out OMOP_ID_MEAN, OMOP_ID_QUANT25, OMOP_ID_QUANT75 for the not selected omops
        labs = labs.select(*[col for col in labs.columns if col.split("_")[0] not in not_select_omops])
        data = data.join(labs, on="FINNGENID", how="left")
        labs_cols = labs.columns

    return data, labs_cols

import polars as pl
def get_other_preds(data: pl.DataFrame,
                    preds: list,
                    start_date: str,
                    fill_missing: bool,
                    fg_ver: str) -> pl.DataFrame:
    """Returns the data with the other predictors added based on the selected predictors in preds.

    Args:
        data (pl.DataFrame): The data to which the other predictors will be added.
        preds (list): The list of selected predictors. Used to check which predictors to add.
        start_date (str): The start date for the data, in the format "YYYY-MM-DD". Used to filter the extra data based on the date.
        fill_missing (bool): Whether to fill missing values for BMI and EDU with the mean value. Default is False.
        fg_ver (str): The FinnGen version, used to get the correct education data.

    Returns:
        pl.DataFrame: The data with the other predictors added.
    """

    extra_data = None
    if "BMI" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "BMI")            
        data = data.join(extra_data.select("FINNGENID", "BMI"), how="left", on="FINNGENID")
        if fill_missing:
            data = data.with_columns(pl.when(pl.col.BMI.is_null()).then(pl.lit(22)).otherwise(pl.col.BMI).alias("BMI"))
    if "HEIGHT" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "HEIGHT")
        data = data.join(extra_data.select("FINNGENID", "HEIGHT"), how="left", on="FINNGENID")  
    if "WEIGHT" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "WEIGHT")
        data = data.join(extra_data.select("FINNGENID", "WEIGHT"), how="left", on="FINNGENID")  
    if "SMOKE" in preds:
        extra_data = get_smoke_data(datetime.strptime(start_date, "%Y-%m-%d"))
        data = data.join(extra_data.select("FINNGENID", "SMOKE"), how="left", on="FINNGENID")  
    if "SBP" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "SBP")
        data = data.join(extra_data.select("FINNGENID", "SBP"), how="left", on="FINNGENID")  
    if "DBP" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "DBP")
        data = data.join(extra_data.select("FINNGENID", "DBP"), how="left", on="FINNGENID")  
    if "ALCOHOL" in preds:
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "ALCOHOL")
        data = data.join(extra_data.select("FINNGENID", "ALCOHOL"), how="left", on="FINNGENID")  
    if "EDU" in preds:
        extra_data = get_edu_data(datetime.strptime(start_date, "%Y-%m-%d"), fg_ver=fg_ver)
        data = data.join(extra_data.select("FINNGENID", "EDU"), how="left", on="FINNGENID")
        if fill_missing:
            data = data.with_columns(pl.when(pl.col.EDU.is_null()).then(pl.lit(0.5)).otherwise(pl.col.EDU).alias("EDU"))
    
    return data

import polars as pl
def fix_sex_col(data: pl.DataFrame) -> pl.DataFrame:
    """Fixes the sex column in the data to be binary with 0 and 1, where 0 is female and 1 is male
    Args:
        data (pl.DataFrame): The data with the sex column to be fixed
    Returns:
        pl.DataFrame: The data with the sex column fixed.
    """
    if data["SEX"].dtype == pl.String:
        data = data.with_columns(pl.col("SEX").replace({"female": 0, "male": 1}).cast(pl.Int32).alias("SEX"))
    else:
        data = data.with_columns(pl.col.SEX.cast(pl.Int32).alias("SEX"))

    return data

def get_col_list(preds: list, 
                 icd_cols: list, 
                 atc_cols: list, 
                 sumstats_cols: list, 
                 second_sumstats_cols: list, 
                 labs_cols: list, 
                 trans_cols: list) -> list:
    X_cols = []
    for pred in preds:
        if pred == "ICD_MAT":
            [X_cols.append(ICD_CODE) for ICD_CODE in icd_cols if ICD_CODE != "FINNGENID" and ICD_CODE != "LAST_CODE_DATE"]
        elif pred == "ATC_MAT":
            [X_cols.append(ATC_CODE) for ATC_CODE in atc_cols if ATC_CODE != "FINNGENID" and ATC_CODE != "LAST_CODE_DATE"]
        elif pred == "SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in sumstats_cols if SUMSTAT != "FINNGENID" and SUMSTAT != "LAST_VAL_DATE"]
        elif pred == "TRANSFORMER":
            [X_cols.append(TRANSFORMER) for TRANSFORMER in trans_cols if TRANSFORMER != "FINNGENID" and TRANSFORMER != "SET" and TRANSFORMER != "y_MEAN_ABNORM"]
        elif pred == "SUMSTATS_MEAN":
            [X_cols.append(SUMSTAT) for SUMSTAT in sumstats_cols if SUMSTAT != "FINNGENID" and SUMSTAT != "LAST_VAL_DATE" and "MEAN" in SUMSTAT]
        elif pred == "SECOND_SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in second_sumstats_cols if SUMSTAT != "FINNGENID" and SUMSTAT != "S_LAST_VAL_DATE"]
        elif pred == "LAB_MAT":
            [X_cols.append(LAB_MAT) for LAB_MAT in labs_cols if LAB_MAT != "FINNGENID"]
        elif pred == "LAB_MAT_MEAN":
            [X_cols.append(LAB_MAT) for LAB_MAT in labs_cols if LAB_MAT != "FINNGENID" and "MEAN" in LAB_MAT]
        else:
            X_cols.append(pred)

    return X_cols
    

from datetime import datetime
import polars as pl
def get_smoke_data(start_date: datetime) -> pl.DataFrame:
    """"Returns the smoking data with the most recent value before the start date for each individual, and the values transformed to 0 for non-smokers, 1 for former smokers, and 2 for current smokers.
    Args:
        start_date (datetime): The start date for the data, in the format "YYYY-MM-DD". Used to filter the smoking data based on the date.
    Returns:
        pl.DataFrame: A DataFrame containing the smoking data with columns "FINNGENID" and "SMOKE", where SMOKE is 0 for non-smokers, 1 for former smokers, and 2 for current smokers.
    """

    smoke_data = pl.read_csv("/finngen/library-red/finngen_R14/harmonized_data/smoking_data/smoking_harmonized_longitudinal_v2.tsv.gz", separator="\t", columns=["FINNGENID", "APPROX_EVENT_DAY", "SMOKE"])
    smoke_data = smoke_data.with_columns(pl.when(pl.col.SMOKE.is_in(["NO", "NEVER", "PASSIVE"]))
                                         .then(pl.lit(0))
                                         .when(pl.col.SMOKE.is_in(["FORMER", "EVER"]))
                                         .then(pl.lit(1))
                                         .when(pl.col.SMOKE=="CURRENT")
                                         .then(pl.lit(2))
                                         .otherwise(pl.lit(None)).alias("SMOKE")
    )
    smoke_data = smoke_data.filter(pl.col.SMOKE.is_not_null(), 
                                   pl.col.APPROX_EVENT_DAY.str.to_date("%Y-%m-%d")<start_date)
    smoke_data = smoke_data.filter((pl.col.APPROX_EVENT_DAY==pl.col.APPROX_EVENT_DAY.max()).over("FINNGENID"))
    smoke_data = smoke_data.select("FINNGENID", "SMOKE")

    return(smoke_data)


from datetime import datetime
import polars as pl
def get_ext_data(start_date: datetime,
                 dtype="BMI"):

    """Returns extra predictor columns to the data based on the selected predictors.

        Args:
            start_date (datetime): The start date for the predictors. After which no data is used.
            dtype (str): The type of predictor to return. Can be "BMI", "ALCOHOL", "SMOKE", "SBP", or "DBP".
        Returns:
            pl.DataFrame: A DataFrame containing the extra predictor columns. With columns "FINNGENID", "DATE", and the respective predictor column.
    """
    if dtype not in ["HEIGHT", "WEIGHT"]:
        ext_file_name = "/finngen/library-red/finngen_R13/hilmo_avohilmo_extended_1.0/data/finngen_R13_hilmo_avohilmo_extended_1.0.txt.gz"
        ext_data = pl.read_csv(ext_file_name, 
                           separator="\t",
                           columns=["FINNGENID", "APPROX_EVENT_DAY", "CODE5", "CODE6", "CODE7", "CODE8", "CODE9"])
        ext_data = ext_data.rename({"CODE5": "BMI", "CODE6": "SMOKE", "CODE7": "ALCOHOL", "CODE8": "SBP", "CODE9":"DBP"})
        ext_data = ext_data.filter((pl.col.BMI != "NA")|(pl.col.SMOKE!="NA")|(pl.col.ALCOHOL!="NA")|(pl.col.SBP!="NA")|(pl.col.DBP!="NA"))
        
        ext_data = ext_data.with_columns(pl.col.BMI.cast(pl.Float64, strict=False).alias("BMI"),
                                     pl.col.SMOKE.cast(pl.Int32,strict=False).alias("SMOKE"),
                                     pl.col.ALCOHOL.cast(pl.Int32,strict=False).alias("ALCOHOL"),
                                     pl.col.SBP.cast(pl.Float64,strict=False).alias("SBP"),
                                     pl.col.DBP.cast(pl.Float64,strict=False).alias("DBP"),
                                     pl.col.APPROX_EVENT_DAY.str.to_date("%Y-%m-%d").alias("DATE")
                                    )
    minimum = pl.read_parquet("/finngen/library-red/finngen_R13/phenotype_1.0/data/finngen_R13_minimum_extended_1.0.parquet")
    if dtype=="HEIGHT":
        bb_data = minimum.select("FINNGENID", "HEIGHT").filter(pl.col.HEIGHT!="NA").with_columns(pl.col.HEIGHT.cast(pl.Int32))
        return(bb_data)
    if dtype=="WEIGHT":
        bb_data = minimum.select("FINNGENID", "WEIGHT").filter(pl.col.WEIGHT!="NA").with_columns(pl.col.WEIGHT.cast(pl.Float64))
        return(bb_data)
    if dtype=="BMI":
        long_data = (ext_data
                            .select("FINNGENID", "DATE", "BMI")
                            .filter(~pl.col.BMI.is_null(), pl.col.DATE<start_date)
                            .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                            .group_by("FINNGENID").agg(pl.col.BMI.mean().alias("BMI"))
                    )
        bb_data = (minimum
                   .select("FINNGENID", "APPROX_BIRTH_DATE", "BMI", "WEIGHT_AGE")
                   .filter(pl.col.BMI!="NA", ~pl.col.FINNGENID.is_in(long_data["FINNGENID"]))
                   .with_columns(pl.col.BMI.cast(pl.Float64),
                                 (pl.when(pl.col.WEIGHT_AGE!="NA")
                                     .then((pl.col.APPROX_BIRTH_DATE.cast(pl.Utf8).str.to_date()+(pl.duration(days=(pl.col("WEIGHT_AGE").cast(pl.Float64, strict=False)*365.25).round()))).cast(pl.Date))
                                     .otherwise(start_date-pl.duration(days=365.25))).alias("DATE").cast(pl.Date)
                                )
                   .filter(pl.col.DATE<start_date)
                   .select("FINNGENID", "BMI")
                  )
        return(pl.concat([long_data, bb_data]))
    if dtype=="SBP":
        return (ext_data
                .select("FINNGENID", "DATE", "SBP")
                .filter(~pl.col.SBP.is_null(), pl.col.DATE<start_date)
                .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                .group_by("FINNGENID").agg(pl.col.SBP.mean().alias("SBP"))
               )
    if dtype=="DBP":
        return (ext_data
                .select("FINNGENID", "DATE", "DBP")
                .filter(~pl.col.DBP.is_null(), pl.col.DATE<start_date)
                .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                .group_by("FINNGENID").agg(pl.col.DBP.mean().alias("DBP"))
               )

def get_edu_data(start_date: datetime,
                 fg_ver="R12") -> pl.DataFrame:
    """Returns the education data with the most recent value before the start date for each individual, and the values transformed to 0 for low education (comprehensive school or less) and 1 for high education (upper secondary school or more).
    Args:
        start_date (datetime): The start date for the data, in the format "YYYY-MM-DD". Used to filter the education data based on the date.
        fg_ver (str): The FinnGen version, used to get the correct education data. If R12, R13, or R14, uses the R12 education data. Otherwise, uses the ML4H education data.
    Returns:
        pl.DataFrame: A DataFrame containing the education data with columns "FINNGENID" and "EDU", where EDU is 0 for low education (comprehensive school or less) and 1 for high education (upper secondary school or more).
    """

    if fg_ver in ["R12", "R13", "R14", "r12", "r13", "r14"]:
        edu_data = pl.read_csv(f"/finngen/pipeline/finngen_R12/socio_register_1.0/data/finngen_R12_socio_register_1.0.txt.gz", separator="\t")
    else:
        edu_data = pl.read_csv(f"/finngen/red/ml4health/processed/socioeconomic_data/ml4health_socioeconomic_data.tsv.gz", 
                               separator="\t",
                               schema_overrides={"CODE1": pl.Utf8, "EVENT_AGE": pl.Float64}, null_values="NA")
        edu_data = edu_data.rename({"FID":"FINNGENID"})
    edu_data = (edu_data.filter(pl.col.CATEGORY=="EDUC").with_columns(pl.col.CODE2.str.head(1).cast(pl.Int32).alias("EDU"))
                    .filter(pl.col.YEAR<=start_date.year)
                    .filter((pl.col.EDU==pl.col.EDU.max()).over("FINNGENID"))
                    .select("FINNGENID", "EDU")
                    .unique()
                    .with_columns(pl.when(pl.col.EDU<=4).then(pl.lit(0)).otherwise(1).alias("EDU"))
    )

    return edu_data
    