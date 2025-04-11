def get_data(lab_name="egfr"):
    if lab_name == "egfr":
        data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_labels/egfr_2025-04-04_data-diag-ctrlsample-start2019_2025-04-07.parquet")
        labels = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_labels/egfr_2025-04-04_data-diag-ctrlsample-start2019_2025-04-07_labels.parquet")
    elif lab_name == "hba1c":
        data = pd.read_csv("/home/ivm/valid/data/processed_data/step4_labels/hba1c_d1_2025-02-10_data-diag-noabnorm_2025-03-06.csv")
        labels = pd.read_csv("/home/ivm/valid/data/processed_data/step4_labels/hba1c_d1_2025-02-10_data-diag-noabnorm_2025-03-06_labels.csv")
        
    data = data.join(labels.select(["FINNGENID", "y_DIAG", "SET"]), on="FINNGENID", how="full")
    return(data)

def quantile_data(lab_name,
                  data):
    # Defining quantiles
    steps = 0.1
    if lab_name == "hba1c":
        steps = 0.2 # size of quantile steps
    quants = np.append(np.quantile(data.filter((pl.col.SET==0)&(~pl.col.VALUE.is_null()))["VALUE"], np.arange(0, 1, steps), method="higher"), 
                       data.filter(pl.col.SET==0)["VALUE"].max())
    
    #quants = [20, 30, 35, 40, 42, 48]
    # Adding column with cut data
    data = data.with_columns(pl.when(pl.col("VALUE").is_not_null())
                            .then(pl.col.VALUE.cut(quants))
                            .alias("QUANT")
    )
    
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
    
    quant_map = dict(zip(quant_df["QUANT"], quant_df["IDX"]))
    quant_df = quant_df.with_columns(pl.col.MID.map_elements(lambda x: "_".join([lab_name, str(x)]), return_dtype=pl.Utf8).alias("TEXT"))
    print(quant_df)
    data = data.join(quant_df.select(["QUANT", "TEXT"]).with_columns(pl.col.QUANT.cast((pl.Categorical))), on="QUANT", how="left")
    # Mapping quantiles to tokens in data
    
    labels_dict = dict(zip(labels["FINNGENID"], labels["y_DIAG"]))
    data["TEXT"].value_counts()

def get_icd_data():
    icds = pl.read_csv("/home/ivm/valid/data/extra_data/processed_data/step1_clean/icds_r12_2024-10-18_min1pct_sum_onttop_2025-02-10.csv", infer_schema_length=0)
    icds = icds.cast({"DATE": pl.Date, "EVENT_AGE": pl.Float64})
    icds = icds.filter(pl.col("FINNGENID").is_in(data["FINNGENID"].unique()))
    icds = icds.filter(pl.col("DATE").dt.year() >= 2012)
    icds = icds.sort(pl.col("DATE")).unique(subset=["DATE", "ICD_THREE"]).select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"])
    display(icds)

def get_atc_data():
    atcs = pl.read_csv("/home/ivm/valid/data/extra_data/processed_data/step1_clean/atcs_r12_2025-02-04_min1pct_sum_onttop_2025-02-18.csv")
    atcs = atcs.cast({"DATE": pl.Date, "EVENT_AGE": pl.Float64})
    atcs = atcs.filter(pl.col("FINNGENID").is_in(data["FINNGENID"].unique()))
    atcs = atcs.filter(pl.col("DATE").dt.year() >= 2012)
    atcs = atcs.sort(pl.col("DATE")).unique(subset=["DATE", "ATC_FIVE"]).select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"])
    display(atcs)

def create_map():
    list_codes = ["female", "male"]
    for code in list(data.filter(~pl.col.TEXT.is_null())["TEXT"].unique()):
        list_codes.append(code) 
    # for code in list(icds.get_column("ICD_THREE").unique()): 
    #      list_codes.append(code)
    # for code in list(atcs.get_column("ATC_FIVE").unique()): 
    #     list_codes.append(code)
    data_type = "lab"
    code_map = dict(zip(list_codes, range(2, len(list_codes)+2)))
    pickle.dump(code_map, open("/home/ivm/valid/data/processed_data/step5_data/pytorch_ehr/" + lab_name + "_" + get_date() + "_" + data_type + "_codemap.pkl", "wb"))

def merge_relevant_data():
    # all_data = pl.concat([pl.DataFrame(data[["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]]).cast({"DATE": pl.Date, "TEXT": str}), 
    #                       icds.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"}),
    #                       atcs.select(["FINNGENID", "EVENT_AGE", "DATE", "ATC_FIVE"]).rename({"ATC_FIVE":"TEXT"})])
    all_data = pl.DataFrame(data[["FINNGENID", "SEX", "EVENT_AGE", "DATE", "TEXT"]])
    # all_data = pl.concat([pl.DataFrame(data[["FINNGENID", "EVENT_AGE", "DATE", "TEXT"]]).cast({"DATE": pl.Date, "TEXT": str}), 
    #                       icds.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"})])
    #all_data = icds.select(["FINNGENID", "EVENT_AGE", "DATE", "ICD_THREE"]).rename({"ICD_THREE":"TEXT"})
    all_data = all_data.join(pl.DataFrame(labels[["FINNGENID", "START_DATE"]]), how="left", on="FINNGENID")
    all_data = all_data.filter(pl.col("DATE") < pl.col("START_DATE").cast(pl.Date), pl.col("TEXT").is_not_null())
    all_data = all_data.with_columns(pl.col("DATE").map_elements(lambda date: datetime.strptime(datetime.strftime(date, "%Y-%m"), "%Y-%m")))
    all_data

from collections import defaultdict
from tqdm import tqdm
import polars

def get_list_data(fids, 
                  all_data, 
                  labels, 
                  code_map):
    sex = (labels.select(["FINNGENID", "SEX"])
                 .with_columns(pl.col.SEX.map_elements(lambda x: 0 if x == "female" else 1).alias("SEX_TOKEN")))
    sex = dict(zip(sex.get_column("FINNGENID"), sex.get_column("SEX_TOKEN")))
    
    labels = labels.with_columns(pl.col("START_DATE").cast(pl.Utf8).str.to_date())
    print(all_data)
    all_data = all_data.with_columns(pl.col("DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", exact=False))
    labels_dict = dict(zip(labels["FINNGENID"], labels["y_DIAG"]))
    start_age = dict(zip(labels["FINNGENID"], labels["EVENT_AGE"]))
    start_date = dict(zip(labels["FINNGENID"], labels["START_DATE"]))

    first_age = (all_data.select(["FINNGENID", "EVENT_AGE", "DATE"])
                         .sort(["FINNGENID", "EVENT_AGE", "DATE"], descending=False)
                         .group_by("FINNGENID")
                         .head(1)
                         .rename({"EVENT_AGE": "MIN_AGE", "DATE": "MIN_DATE"}))
    min_age = dict(zip(first_age.get_column("FINNGENID"), first_age.get_column("MIN_AGE")))
    min_date =  dict(zip(first_age.get_column("FINNGENID"), first_age.get_column("MIN_DATE")))
    
    out_data = defaultdict(list)    
    last_dates = defaultdict(list)
    time_diffs = defaultdict(list)
    
    for fid in fids:
        out_data[fid] = [fid, labels_dict[fid], []]
        out_data[fid][2].append([[0], [sex[fid]]])
        if fid in min_age:
            time_diffs[fid] = np.round(round(min_age[fid])*365.25/30)
            last_dates[fid] = min_date[fid]
        else:
            out_data[fid][2].append([[np.round(round(start_age[fid])*365.25/30)], [1]])

    all_data = all_data.filter(pl.col("FINNGENID").is_in(fids)).sort("DATE", descending=False).with_columns(pl.col("TEXT").replace_strict(code_map))
    for fid, crnt_data in tqdm(all_data.group_by("FINNGENID"), total=len(fids)):
        fid = fid[0]
        last_date = last_dates[fid]
        time_diff = time_diffs[fid]
        last_codes = []
        start = True
        for date, date_data in crnt_data.sort("DATE", descending=False).group_by("DATE"):
            date = date[0]
            if not start: time_diff = np.round((date-last_date).days/30)
            else: start = False
            try:
                codes = list(set(date_data.get_column("TEXT").to_list()))
            except:
                codes = date_data.get_column("TEXT").to_list()
            last_date = date
            out_data[fid][2].append([[time_diff], codes])
            last_codes = codes
                
        out_data[fid][2].append([[np.round((start_date[fid]-last_date).days/30)], [1]])
    return(out_data)

train_data = get_list_data(labels.filter(pl.col.SET==0).get_column("FINNGENID").unique(), 
                           all_data, 
                           labels, 
                           code_map)
pickle.dump(list(train_data.values()), open("/home/ivm/valid/data/processed_data/step5_data/pytorch_ehr/"  + lab_name + "_" + get_date() + "_" + data_type + "_train.pkl", "wb"), -1)
test_data = get_list_data(labels.filter(pl.col.SET==2).get_column("FINNGENID").unique(), 
                           all_data, 
                           labels, 
                           code_map)
test_data
val_data = get_list_data(labels.filter(pl.col.SET==1).get_column("FINNGENID").unique(), 
                           all_data, 
                           labels, 
                           code_map)
val_data
pickle.dump(list(val_data.values()), open("/home/ivm/valid/data/processed_data/step5_data/pytorch_ehr/"  + lab_name + "_" + get_date() + "_" + data_type + "_valid.pkl", "wb"), -1)
pickle.dump(list(test_data.values()), open("/home/ivm/valid/data/processed_data/step5_data/pytorch_ehr/"  + lab_name + "_" + get_date() + "_" + data_type + "_test.pkl", "wb"), -1)