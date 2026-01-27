# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Util functions                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def pretty_int(int_no):
    """Round a number to a pretty string with K (N>1000), M (N>1000000)."""
    if int_no >= 1000000:
        return(str(round(int_no / 1000000,2)) + "M")
    elif int_no >= 1000:
        return(str(round(int_no / 1000)) + "K")
    else:
        return(str(round(int_no, 0)))
    
import polars as pl
from general_utils import read_file
def get_plot_names(col_names: list[str], 
                   lab_name: str,
                   lab_name_two: str="",
                   omop_mapping_path="/home/ivm/valid/data/extra_data/upload/kanta_omop_mappings_counts.csv") -> list[str]:
    """Get the plot names for the columns. 
       Prettier/more readable names for output. OMOP mapping is used for lab values."""
    kanta_omop_map = read_file(omop_mapping_path, schema={"measurement_concept_id": pl.Utf8, "concept_name": pl.Utf8})
    kanta_omop_map = dict(zip(kanta_omop_map["measurement_concept_id"].to_list(), kanta_omop_map["concept_name"].to_list()))
    new_names = []
    if lab_name == "hba1c": lab_name = "HbA1c"
    if lab_name == "egfr": lab_name = "eGFR"
    if lab_name == "ana": lab_name = "ANA"
    if lab_name == "tsh": lab_name = "TSH"
    if lab_name == "ldl": lab_name = "LDL"
    if lab_name_two == "ftri": lab_name_two == "Fasting TG"
    if lab_name_two == "t4": lab_name_two == "fT4"
    if lab_name_two == "cystc": lab_name_two == "Cystatin C"
    if lab_name_two == "fgluc": lab_name_two = "Fasting Glucose"
    for col_name in col_names:
        if col_name.startswith("S_"): 
            crnt_lab_name = lab_name_two
            col_name = col_name.replace("S_", "")
        else: 
            crnt_lab_name = lab_name
        if col_name.split("_")[0] in kanta_omop_map.keys():
            if "QUANT" in col_name:
                new_name = col_name.split("_")[1].replace("QUANT", "") + "% Quant - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
                
            else:
                new_name = col_name.split("_")[1].capitalize() + " - " + kanta_omop_map[col_name.split("_")[0]].split("[")[0]
            if len(kanta_omop_map[col_name.split("_")[0]].split("]")) > 1:
                new_name += kanta_omop_map[col_name.split("_")[0]].split("]")[1]
        else:
            if col_name.startswith("QUANT"):
                new_name = col_name.split("_")[1] + "% Quant - " + crnt_lab_name
            elif col_name in ["MIN", "MEAN", "MAX", "MEDIAN", ]:
                new_name = col_name.capitalize() + " - " + crnt_lab_name
            elif col_name == "LAST_VAL_DIFF":
                new_name = "Distance of last value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_100":
                new_name = "Last value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_50":
                new_name = "Mid value - " + crnt_lab_name
            elif col_name == "IDX_QUANT_0":
                new_name = "First value - " + crnt_lab_name
            elif col_name == "EVENT_AGE":
                new_name = "Age"
            elif col_name == "FIRST_LAST":
                new_name = "Days between first and last - " + crnt_lab_name
            elif col_name == "SEX":
                new_name = "Sex"
            elif col_name == "MIN_LOC":
                new_name = "Location of min (-days) - " + crnt_lab_name
            elif col_name == "MAX_LOC":
                new_name = "Location of max (-days) - " + crnt_lab_name
            elif col_name == "REG_COEF":
                new_name = "Regression coef - " + crnt_lab_name
            elif col_name == "SUM_ABS_CHANGE":
                new_name = "Sum of absolute change - " + crnt_lab_name
            elif col_name == "MAX_ABS_CHANGE":
                new_name = "Max of absolute change - " + crnt_lab_name
            elif col_name == "MEAN_ABS_CHANGE":
                new_name = "Mean of absolute change - " + crnt_lab_name
            elif col_name == "ABNORM":
                new_name = "Number of prior abnorm - " + crnt_lab_name
            elif col_name == "MEAN_CHANGE":
                new_name = "Mean change - " + crnt_lab_name
            elif col_name == "MAX_CHANGE":
                new_name = "Max change - " + crnt_lab_name
            elif col_name == "SUM":
                new_name = "Sum - " + crnt_lab_name
            elif col_name == "SD":
                new_name = "Standard deviation - " + crnt_lab_name
            elif col_name == "ABS_ENERG":
                new_name = "Absolute energy - " + crnt_lab_name
            elif col_name == "SKEW":
                new_name = "Skew - " + crnt_lab_name
            elif col_name == "KURT":
                new_name = "Kurtosis - " + crnt_lab_name
            elif col_name == "SEQ_LEN":
                new_name = "Number of measurements - " + crnt_lab_name
            elif col_name == "SUM_CHANGE":
                new_name = "Sum of change - " + crnt_lab_name
            elif col_name == "YEAR":
                new_name = "Year"
            else:
                new_name = col_name

        new_names.append(new_name)
    return(new_names)
    
import numpy as np
import polars as pl
from collections.abc import Iterable
def round_column_min5(col_data: Iterable) -> tuple[pl.Series, float, float]:
    """Rounds a column to the nearest 5 and replaces values with less than 5 counts with the nearest value with 5 counts.
        Returns the new column and the min and max values of the column."""
    # counting the frequencies of the values
    col_data = pl.Series(col_data)
    mean_freqs = col_data.round().value_counts()
    mean_freqs.columns = ["VALUE", "COUNT"]
    # map for values with too low counts
    value_map = dict()
    # min and max values with at least 5 counts
    min_val = mean_freqs.filter(pl.col("COUNT") >= 5).select(pl.col("VALUE").min()).to_numpy()[0][0]
    max_val = mean_freqs.filter(pl.col("COUNT") >= 5).select(pl.col("VALUE").max()).to_numpy()[0][0]

    # finding mapping
    for row in mean_freqs.rows():
        crnt_value = row[0]
        crnt_count = row[1]
        if crnt_count < 5:
            if abs(min_val-crnt_value) < abs(max_val-crnt_value):
                value_map[crnt_value] = min_val
            else:
                value_map[crnt_value] = max_val
        else: # No new mapping needed
            value_map[crnt_value] = crnt_value
    # mapping the values
    newcol = col_data.map_elements(lambda x: value_map.get(np.round(x), np.round(x)), return_dtype=pl.Float64)

    return(newcol, min_val, max_val)

