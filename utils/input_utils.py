import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import read_file, print_count
from datetime import datetime
def get_data_and_pred_list(file_path_labels: str, 
                           file_path_icds: str,
                           file_path_atcs: str,
                           file_path_sumstats: str,
                           file_path_second_sumstats: str,
                           file_path_labs: str,
                           file_path_pgs1: str,
                           file_path_pgs2: str,
                           file_path_transformer: str,
                           preds: list,
                           start_date: str,
                           fill_missing: int=0,
                           fids_path: str="",
                           future_val: str="") -> tuple[pl.DataFrame, list]:
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
    #                 Getting Data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(file_path_labels)
    if future_val == "train" : data = data.filter(pl.col.SET==0)
    elif future_val == "val" or future_val == "final train": data = data.filter(pl.col.SET!=0)
    
    if fids_path != "": 
        fids = read_file(fids_path)
        print_count(data)
        data = data.filter(pl.col.FINNGENID.is_in(fids["FINNGENID"]))
    print_count(data)

    if file_path_transformer != "": 
        trans_data = read_file(file_path_transformer)
        data = data.join(trans_data.drop("SET", "y_MEAN_ABNORM"), on="FINNGENID", how="left")
    print_count(data)

    # # # # # # # # # PGS # # # # # # # # # # # # # # # # # # #
    if file_path_pgs1 != "":
        pgs = pl.read_csv(file_path_pgs1, separator="\t")
        if len(pgs.columns)==2: # Zhijian PGS for eGFR levels
            pgs.columns = ["FINNGENID", "PGS1"]
            #pgs = pgs.with_columns((-1*pl.col.PGS1).alias("PGS1"))
        else: # Zhiyu PGS for creatinine slopes
            pgs.columns = ["X1", "FINNGENID", "X2", "X3", "PGS1"] 
            pgs = pgs.with_columns((-1*pl.col.PGS1).alias("PGS1"))

        data = data.join(pgs.select("FINNGENID", "PGS1"), on="FINNGENID", how="left")
        pcs = pl.read_csv("/finngen/library-red/finngen_R13/analysis_covariates/data/R13_COV_PHENO_V0.FID.txt.gz", separator="\t", columns=["IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"])
        data = data.join(pcs.select("IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"), left_on="FINNGENID", right_on="IID", how="left")
        print("Number of individuals without PCs")
        print_count(data.filter(pl.col.PC1.is_null()))
        print("Number of individuals left")
        print_count(data.filter(~pl.col.PC1.is_null()))
        data = data.filter(~pl.col.PC1.is_null())
    # Adding other data modalities
    if file_path_pgs2 != "":
        pgs = pl.read_csv(file_path_pgs2, separator="\t")
        if len(pgs.columns)==2: # Zhijian PGS for eGFR levels
            pgs.columns = ["FINNGENID", "PGS2"]
            pgs = pgs.with_columns((-1*pl.col.PGS2).alias("PGS2"))
        else: # Zhiyu PGS for creatinine slopes
            pgs.columns = ["X1", "FINNGENID", "X2", "X3", "PGS2"] 
        data = data.join(pgs.select("FINNGENID", "PGS2"), on="FINNGENID", how="left")
        pcs = pl.read_csv("/finngen/library-red/finngen_R13/analysis_covariates/data/R13_COV_PHENO_V0.FID.txt.gz", separator="\t", columns=["IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"])
        data = data.join(pcs.select("IID", "PC1","PC2","PC3", "PC4","PC5","PC6","PC7","PC8","PC9","PC10"), left_on="FINNGENID", right_on="IID", how="left")
        print("Number of individuals without PCs")
        print_count(data.filter(pl.col.PC1.is_null()))
        print("Number of individuals left")
        print_count(data.filter(~pl.col.PC1.is_null()))
        data = data.filter(~pl.col.PC1.is_null())

    # # # # # # # # # ICD # # # # # # # # # # # # # # # # # # #
    if file_path_icds != "": 
        icds = read_file(file_path_icds)
        data = data.join(icds, on="FINNGENID", how="left")

    # # # # # # # # # ATC # # # # # # # # # # # # # # # # # # #
    if file_path_atcs != "": 
        atcs = read_file(file_path_atcs)
        data = data.join(atcs, on="FINNGENID", how="left")

    # # # # # # # # # Sumstats # # # # # # # # # # # # # # # # # # #
    if file_path_sumstats != "": 
        sumstats = read_file(file_path_sumstats,
                             schema={"SEQ_LEN": pl.Float64,
                                     "MIN_LOC": pl.Float64,
                                     "MAX_LOC": pl.Float64,
                                     "FIRST_LAST": pl.Float64})
        if "SET" in sumstats.columns: sumstats = sumstats.drop("SET") # dropping duplicate info on set in sumstats if present
        data = data.join(sumstats, on="FINNGENID", how="left")
        data = data.with_columns(pl.when(pl.col.MEAN.is_null()).then(pl.lit(1)).otherwise(pl.lit(0)).alias("NO_HISTORY"))
        if start_date != "":
            data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.LAST_VAL_DATE).dt.total_days().alias("LAST_VAL_DIFF"))

    # # # # # # # # # Second sumstats # # # # # # # # # # # # # # # # # # #
    if file_path_second_sumstats != "": 
        second_sumstats = read_file(file_path_second_sumstats)
        if "SET" in second_sumstats.columns: second_sumstats = second_sumstats.drop("SET")
        second_sumstats = second_sumstats.rename({col: f"S_{col}" for col in second_sumstats.columns if col != "FINNGENID"})
        data = data.join(second_sumstats, on="FINNGENID", how="left")
        if start_date != "":
            data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.S_LAST_VAL_DATE).dt.total_days().alias("S_LAST_VAL_DIFF"))

    # # # # # # # # # Labs # # # # # # # # # # # # # # # # # # #
    if file_path_labs != "": 
        labs = read_file(file_path_labs)
        data = data.join(labs, on="FINNGENID", how="left")

    # # # # # # # # # Extra # # # # # # # # # # # # # # # # # # #
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
        extra_data = get_ext_data(datetime.strptime(start_date, "%Y-%m-%d"), "SMOKE")
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
        extra_data = get_edu_data(datetime.strptime(start_date, "%Y-%m-%d"))
        data = data.join(extra_data.select("FINNGENID", "EDU"), how="left", on="FINNGENID")
        if fill_missing:
            data = data.with_columns(pl.when(pl.col.EDU.is_null()).then(pl.lit(0.5)).otherwise(pl.col.EDU).alias("EDU"))
            
    # Changing data-modality of sex column
    data = data.with_columns(pl.col("SEX").replace({"female": 0, "male": 1}).cast(pl.Int32).alias("SEX"))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Predictors                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    X_cols = []
    for pred in preds:
        if pred == "ICD_MAT":
            [X_cols.append(ICD_CODE) for ICD_CODE, _ in icds.schema.items() if ICD_CODE != "FINNGENID" and ICD_CODE != "LAST_CODE_DATE"]
        elif pred == "ATC_MAT":
            [X_cols.append(ATC_CODE) for ATC_CODE, _ in atcs.schema.items() if ATC_CODE != "FINNGENID" and ATC_CODE != "LAST_CODE_DATE"]
        elif pred == "SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT, _ in sumstats.schema.items() if SUMSTAT != "FINNGENID" and SUMSTAT != "LAST_VAL_DATE"]
        elif pred == "TRANSFORMER":
            [X_cols.append(TRANSFORMER) for TRANSFORMER, _ in trans_data.schema.items() if TRANSFORMER != "FINNGENID" and TRANSFORMER != "SET" and TRANSFORMER != "y_MEAN_ABNORM"]
        elif pred == "SUMSTATS_MEAN":
            [X_cols.append(SUMSTAT) for SUMSTAT, _ in sumstats.schema.items() if SUMSTAT != "FINNGENID" and SUMSTAT != "LAST_VAL_DATE" and "MEAN" in SUMSTAT]
        elif pred == "SECOND_SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT, _ in second_sumstats.schema.items() if SUMSTAT != "FINNGENID" and SUMSTAT != "S_LAST_VAL_DATE"]
        elif pred == "LAB_MAT":
            [X_cols.append(LAB_MAT) for LAB_MAT, _ in labs.schema.items() if LAB_MAT != "FINNGENID"]
        elif pred == "LAB_MAT_MEAN":
            [X_cols.append(LAB_MAT) for LAB_MAT, _ in labs.schema.items() if LAB_MAT != "FINNGENID" and "MEAN" in LAB_MAT]
        else:
            X_cols.append(pred)
    return(data.unique(), X_cols)

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
    if dtype=="ALCOHOL":
        return (ext_data
                    .select("FINNGENID", "DATE", "ALCOHOL")
                    .filter(~pl.col.ALCOHOL.is_null(), pl.col.DATE<start_date)
                    .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                    .filter(~pl.col.ALCOHOL.is_null())
                    .filter((pl.col.ALCOHOL==pl.col.ALCOHOL.max()).over("FINNGENID"))
                    .with_columns(pl.when(pl.col.ALCOHOL<=10)
                                  .then(pl.lit(0))
                                  .when((pl.col.ALCOHOL>10))
                                  .then(pl.lit(1))
                                  .alias("ALCOHOL"))
                )
    if dtype=="SMOKE":
        long_data = ((ext_data
                    .select("FINNGENID", "DATE", "SMOKE")
                    .filter(~pl.col.SMOKE.is_null(), pl.col.DATE<start_date)
                    .with_columns(pl.when(pl.col.SMOKE==9).then(pl.lit(None)).otherwise(pl.col.SMOKE).alias("SMOKE"))
                    .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                    .filter(~pl.col.SMOKE.is_null())
                    .filter((pl.col.SMOKE==pl.col.SMOKE.min()).over("FINNGENID"))
                     .unique()
                    .with_columns(pl.when(pl.col.SMOKE<=3).then(pl.lit(1)).otherwise(pl.lit(0)).alias("SMOKE"))
                )).select("FINNGENID", "SMOKE")
        bb_data = minimum.filter(pl.col.CURRENT_SMOKER!="NA", ~pl.col.FINNGENID.is_in(long_data["FINNGENID"])).with_columns(pl.col.CURRENT_SMOKER.cast(pl.Int32).alias("SMOKE")).select("FINNGENID", "SMOKE")
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

def get_edu_data(start_date: datetime):
    edu_data = pl.read_csv("/finngen/pipeline/finngen_R12/socio_register_1.0/data/finngen_R12_socio_register_1.0.txt.gz", separator="\t")
    edu_data = (edu_data.filter(pl.col.CATEGORY=="EDUC").with_columns(pl.col.CODE2.str.head(1).cast(pl.Int32).alias("EDU"))
                    .filter(pl.col.YEAR<=start_date.year)
                    .filter((pl.col.EDU==pl.col.EDU.max()).over("FINNGENID"))
                    .select("FINNGENID", "EDU")
                    .unique()
                    .with_columns(pl.when(pl.col.EDU<=4).then(pl.lit(0)).otherwise(1).alias("EDU"))
    )
    return edu_data
    