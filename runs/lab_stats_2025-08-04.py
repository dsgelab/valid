################################################################################################
############################# SETUP ############################################################
################################################################################################

#! source /home/ivm/envs/valid_env/bin/activate
import polars as pl
import sys
 
sys.path.append(("/home/ivm/valid/scripts/utils/"))
sys.path.append(("/home/ivm/valid/scripts/steps/"))

from general_utils import *

# For shap
from plot_utils import get_plot_names
import pickle
import shap
from model_eval_utils import compare_models
from step2_diags import get_diag_med_data


%load_ext autoreload
%autoreload 2

test_filter = (pl.col.SET==2)
valid_filter = (pl.col.SET==1)
train_filter = (pl.col.SET==0)
base_date = datetime(2021,10,1)
set_names = {0: "Train", 1: "Valid", 2: "Test"}
         
################################################################################################
############################# Get data ############################################################
################################################################################################
# aki_data = get_diag_med_data("(^N17)|(^N14[1|2])|(^N990)|(^O904)|(^R34)|(^R944)").filter(pl.col.CODE.str.contains("(^N17)|(^N14[1|2])|(^N990)|(^O904)|(^R34)|(^R944)"))
# ht_data = get_diag_med_data("(^I10)").filter(pl.col.CODE.str.contains("(^I10)"))
# ob_data = get_diag_med_data("(^E66)").filter(pl.col.CODE.str.contains("(^E66)"))
# gout_data = get_diag_med_data("(^M10)").filter(pl.col.CODE.str.contains("(^M10)"))
# diabetes = get_diag_med_data(med_regex="(^A10)").filter(pl.col.CODE.str.contains("(^A10)"))
# diuretics = get_diag_med_data(med_regex="(^C03)").filter(pl.col.CODE.str.contains("(^C03)"))
# nsaids = get_diag_med_data(med_regex="(^M01A)").filter(pl.col.CODE.str.contains("(^M01A)"))
# statins = get_diag_med_data(med_regex="(^C10AA)").filter(pl.col.CODE.str.contains("(^C10AA)"))
# bp = get_diag_med_data(med_regex="(^C02)|(^03)|(^C07)|(^C09)|(^C08)|(^C09)").filter(pl.col.CODE.str.contains("(^C02)|(^03)|(^C07)|(^C09)|(^C08)|(^C09)"))
# raas_data = get_diag_med_data(med_regex="(^C09AA)|(^C09BA)|(^C09CA)|(^C09DA)").filter(pl.col.CODE.str.contains("(^C09AA)|(^C09BA)|(^C09CA)|(^C09DA)"))
# sglt2_data = get_diag_med_data(med_regex="(^A10BK)").filter(pl.col.CODE.str.contains("(^A10BK)"))

aki_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ICD_aki.parquet")
ht_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ICD_ht.parquet")
diab_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/hba1c_R13_2025-08-01_diags.parquet")
ob_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ICD_obese.parquet")
gout_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ICD_gout.parquet")
raas_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_raas.parquet")
sglt2_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_sglt2.parquet")
diabetes = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_diab.parquet")
diuretics = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_diuretics.parquet")
nsaids = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_nsaids.parquet")
statins = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_statins.parquet")
bp = pl.read_parquet("/home/ivm/valid/data/processed_data/step4_data/atc_icd_codes/ATC_bp.parquet")
thyroid_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/tsh_R13_2025-08-01_diags.parquet")
ckd_data = get_diag_med_data("(^N18)").filter(pl.col.CODE.str.contains("(^N18)"))
################################################################################################
############################# Helpers ############################################################
################################################################################################
def add_info_columns(labels,
                    lab_name):
    icd_file_paths = {"eGFR": "/home/ivm/valid/data/processed_data/step4_data/egfr_d1_herold-part_ld_2025-07-23_filtered_2025-07-23_testv1_2022_w3_2025-07-23_icds_1pct_count_egfr_2025-07-29.parquet",
                      "HbA1c": "/home/ivm/valid/data/processed_data/step4_data/hba1c_d1_strong_2025-07-18_filtered_2025-07-18_testv1_2022_w3_2025-07-18_icds_1p0pct_count_hba1c_2025-08-04.parquet",
                       "TSH": "/home/ivm/valid/data/processed_data/step4_data/tsh_d1_multi_2025-07-18_filtered_2025-08-01_testv2_2022_w3_2025-08-01_icds_1p0pct_count_tsh_2025-08-04.parquet"}
    atc_file_paths = {"eGFR": "/home/ivm/valid/data/processed_data/step4_data/egfr_d1_herold-part_ld_2025-07-23_filtered_2025-07-23_testv1_2022_w3_2025-07-23_atcs_1pct_count_egfr_2025-07-29.parquet",
                      "HbA1c": "/home/ivm/valid/data/processed_data/step4_data/hba1c_d1_strong_2025-07-18_filtered_2025-07-18_testv1_2022_w3_2025-07-18_atcs_1p0pct_count_hba1c_2025-08-04.parquet",
                      "TSH": "/home/ivm/valid/data/processed_data/step4_data/tsh_d1_multi_2025-07-18_filtered_2025-08-01_testv2_2022_w3_2025-08-01_atcs_1p0pct_count_tsh_2025-08-04.parquet"}

    ###### ICD COUNTS ###################################################
    icd_data = pl.read_parquet(icd_file_paths[lab_name])
    icd_counts = icd_data.with_columns(
        pl.sum_horizontal([
            pl.col(col) for col in icd_data.columns if col != "FINNGENID" and col != "LAST_CODE_DATE"
        ]).alias("ICD_COUNT")
    ).select("FINNGENID", "ICD_COUNT").with_columns(pl.when(pl.col.ICD_COUNT==0).then(pl.lit(None)).otherwise(pl.col.ICD_COUNT).alias("ICD_COUNT"))
    labels = labels.join(icd_counts, on="FINNGENID", how="left")
    
    ###### ATC COUNTS ###################################################
    atc_data = pl.read_parquet(atc_file_paths[lab_name])
    atc_counts = atc_data.with_columns(
        pl.sum_horizontal([
            pl.col(col) for col in atc_data.columns if col != "FINNGENID" and col != "LAST_CODE_DATE"
        ]).alias("ATC_COUNT")
    ).select("FINNGENID", "ATC_COUNT").with_columns(pl.when(pl.col.ATC_COUNT==0).then(pl.lit(None)).otherwise(pl.col.ATC_COUNT).alias("ATC_COUNT"))
    labels = labels.join(atc_counts, on="FINNGENID", how="left")
    
    ###### Kanta lab COUNTS ###################################################
    kanta_data = pl.read_parquet("/home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet")
    kanta_data = kanta_data.group_by("FINNGENID").agg(pl.len().alias("LAB_COUNT"))
    labels = labels.join(kanta_data, on="FINNGENID", how="left")
    
    ###### Education Data ###################################################
    edu_data = get_edu_data(base_date)
    
    ###### Laboratory Data ###################################################
    egfr_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_herold-full_ld_2025-07-24.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "eGFR"})
                )
    egfr_abnorm_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_herold-full_ld_2025-07-24.parquet")
                 .filter(pl.col.DATE<base_date)
                 .group_by("FINNGENID").agg((pl.col.ABNORM_CUSTOM==1).any())
                 .select("FINNGENID", "ABNORM_CUSTOM")
                 .rename({"ABNORM_CUSTOM": "eGFR_ABNORM"})
                )
    hba1c_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/hba1c_d1_strong_2025-07-18.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "HbA1c"})
                )
    hba1c_abnorm_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/hba1c_d1_strong_2025-07-18.parquet")
                 .filter(pl.col.DATE<base_date)
                 .group_by("FINNGENID").agg((pl.col.ABNORM_CUSTOM==1).any())
                 .select("FINNGENID", "ABNORM_CUSTOM")
                 .rename({"ABNORM_CUSTOM": "HbA1c_ABNORM"})
                )
    cystc_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/cystc_d1_herold-part_ld_2025-07-23.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "CystC"})
                )
    uacr_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/uacr_d1_ld_2025-07-22.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "UACR"})
                )
    fg_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/fgluc_d1_2025-07-18.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "fGluc"})
                )
    g_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/gluc_d1_2025-07-18.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "Gluc"})
                )
    tsh_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/tsh_d1_multi_2025-07-22.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "TSH"})
                )
    tsh_abnorm_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/tsh_d1_multi_2025-07-22.parquet")
                 .filter(pl.col.DATE<base_date)
                 .group_by("FINNGENID").agg((pl.col.ABNORM_CUSTOM==1).any())
                 .select("FINNGENID", "ABNORM_CUSTOM")
                 .rename({"ABNORM_CUSTOM": "TSH_ABNORM"})
                )
    t4_data = (pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/t4_d1_2025-07-21.parquet")
                 .filter(pl.col.DATE<base_date)
                 .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                 .select("FINNGENID", "VALUE")
                 .rename({"VALUE": "T4"})
                )
    
    labels = labels.join(edu_data, on="FINNGENID", how="left")
    labels = labels.join(get_ext_data(base_date, "BMI"), on="FINNGENID", how="left")
    labels = labels.join(get_ext_data(base_date, "SBP"), on="FINNGENID", how="left")
    labels = labels.join(get_ext_data(base_date, "DBP"), on="FINNGENID", how="left")
    labels = labels.join(get_ext_data(base_date, "SMOKE"), on="FINNGENID", how="left")
    labels = labels.join(egfr_data, on="FINNGENID", how="left")
    labels = labels.join(egfr_abnorm_data, on="FINNGENID", how="left")
    labels = labels.join(hba1c_data, on="FINNGENID", how="left")
    labels = labels.join(hba1c_abnorm_data, on="FINNGENID", how="left")
    labels = labels.join(cystc_data, on="FINNGENID", how="left")
    labels = labels.join(uacr_data, on="FINNGENID", how="left")
    labels = labels.join(g_data, on="FINNGENID", how="left")
    labels = labels.join(fg_data, on="FINNGENID", how="left")
    labels = labels.join(tsh_data, on="FINNGENID", how="left")
    labels = labels.join(tsh_abnorm_data, on="FINNGENID", how="left")
    labels = labels.join(t4_data, on="FINNGENID", how="left")

    return(labels)

def prior_diags(df: pl.DataFrame,
                diag_data: pl.DataFrame) -> pl.DataFrame:
    if "DIAG_DATE" in diag_data.columns:
        diag_data = diag_data.rename({"DIAG_DATE": "APPROX_EVENT_DAY"})
    return (df.join(diag_data.select("FINNGENID", "APPROX_EVENT_DAY"), on="FINNGENID", how="left")
              .filter(pl.col("APPROX_EVENT_DAY") < base_date)
              .drop("APPROX_EVENT_DAY").unique()
    )

def goal_diags(df: pl.DataFrame,
                diag_data: pl.DataFrame) -> pl.DataFrame:
    if "DIAG_DATE" in diag_data.columns:
        diag_data = diag_data.rename({"DIAG_DATE": "APPROX_EVENT_DAY"})
    return (df.join(diag_data.select("FINNGENID", "APPROX_EVENT_DAY"), on="FINNGENID", how="left")
              .filter(pl.col("APPROX_EVENT_DAY").dt.year()>=2022)
              .drop("APPROX_EVENT_DAY").unique()
    )

################################################################################################
############################# Stats ############################################################
################################################################################################
from step5_fit_XGB import get_ext_data, get_edu_data
from collections import defaultdict
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest


outcomes = ["y_MEAN_ABNORM", "y_NEXT_ABNORM"]
base_date = datetime(2021,10,1)

objectives = {"AKI >= 2022": lambda df: goal_diags(df, aki_data),
              "CKD >= 2022": lambda df: goal_diags(df, ckd_data),
              "AKI|CKD >= 2022": lambda df: goal_diags(df, pl.concat([ckd_data.select(aki_data.columns), aki_data])),
              "Thyroid problems >= 2022": lambda df: goal_diags(df, thyroid_data),
              "Diabetes >= 2022": lambda df: goal_diags(df, diab_data),
               "AKI": lambda df: prior_diags(df, aki_data),
              "CKD": lambda df: prior_diags(df, ckd_data),
              "Hypertension": lambda df: prior_diags(df, ht_data),
              "Diabetes": lambda df: prior_diags(df, diab_data),
              "Overweight": lambda df: prior_diags(df, ob_data),
              "Gout": lambda df: prior_diags(df, gout_data),
              "RAAS": lambda df: prior_diags(df, raas_data),
              "SGLT2": lambda df: prior_diags(df, sglt2_data),
              "SGLT2 >= 2022": lambda df: goal_diags(df, sglt2_data),
              "Diabetes medication": lambda df: prior_diags(df, diabetes),
              "Statins": lambda df: prior_diags(df, statins),
              "Blood pressure medications": lambda df: prior_diags(df, bp),
              "Diuretics": lambda df: prior_diags(df, diuretics),
              "NSAIDs": lambda df: prior_diags(df, nsaids),
              "Education": lambda df: df.filter(pl.col.EDU==1),
              "Education missing": lambda df: df.filter(pl.col.EDU.is_null()),
              "Smoking": lambda df: df.filter(pl.col.SMOKE==1),
              "Smoking missing": lambda df: df.filter(pl.col.SMOKE.is_null()),
              "eGFR abnorm": lambda df: df.filter(pl.col.eGFR_ABNORM==1),
              "HbA1c abnorm": lambda df: df.filter(pl.col.HbA1c_ABNORM==1),
              "TSH abnorm": lambda df: df.filter(pl.col.TSH_ABNORM==1)
             }           
columns = {"Age": "EVENT_AGE",
           "BMI": "BMI",
           "SBP": "SBP",
           "DBP": "DBP",
           "eGFR": "eGFR",
           "Cystatin C": "CystC",
           "UACR": "UACR",
           "HbA1c": "HbA1c",
           "Fasting Glucose": "fGluc",
           "Glucose": "Gluc",
           "TSH": "TSH",
            "T4": "T4",
           "#ICDs": "ICD_COUNT",
           "#ATCs": "ATC_COUNT",
           "#Lab": "LAB_COUNT"

}             

results_bin = defaultdict(list)
results_cont = defaultdict(list)

label_files = {"eGFR": "/home/ivm/valid/data/processed_data/step3_labels/egfr_d1_herold-part_ld_2025-07-22_filtered_2025-07-23_testv1_2022_w3_2025-07-23_labels.parquet",
               "HbA1c": "/home/ivm/valid/data/processed_data/step3_labels/hba1c_d1_strong_2025-07-18_filtered_2025-07-18_testv1_2022_w3_2025-07-18_labels.parquet",
                "TSH": "/home/ivm/valid/data/processed_data/step3_labels/tsh_d1_multi_2025-07-18_filtered_2025-08-01_testv2_2022_w3_2025-08-01_labels.parquet"}
for lab_name, label_file in label_files.items():
    print(lab_name)
    labels = pl.read_parquet(label_file)
    labels = add_info_columns(labels, lab_name)
    
    for column_descr, column_name in columns.items():
        for outcome in outcomes: 
            for set_no in [0,1,2]:
                if set_no == 0: set_filter = train_filter
                elif set_no == 1: set_filter = valid_filter
                else: set_filter = test_filter
                results_bin["LAB_NAME"].append(lab_name)
                results_bin["SET"].append(set_names[set_no])
                results_bin["OBJECTIVE"].append(column_descr)
                results_bin["OUTCOME"].append(outcome)
    
                crnt_data = labels.filter(set_filter)
                results_bin["N_TOTAL"].append(crnt_data.height)
                crnt_column = crnt_data.get_column(column_name)
                results_bin["N_MISSING"].append(crnt_column.is_null().sum())
                results_bin["ALL_MEAN"].append(crnt_column.mean())
                results_bin["ALL_SD"].append(crnt_column.std())
                case_data = labels.filter(set_filter, pl.col(outcome)==1).get_column(column_name)
                results_bin["N_CASES"].append(case_data.len())
                results_bin["N_MISSING_CASES"].append(case_data.is_null().sum())
    
                results_bin["CASES_MEAN"].append(case_data.mean())
                results_bin["CASES_SD"].append(case_data.std())
        
                control_data = labels.filter(set_filter, pl.col(outcome)==0).get_column(column_name)
                results_bin["N_CONTROLS"].append(control_data.len())
                results_bin["N_MISSING_CONTROLS"].append(control_data.is_null().sum())
                
                results_bin["CONTROLS_MEAN"].append(control_data.mean())
                results_bin["CONTROLS_SD"].append(control_data.std())
                
                results_bin["PVAL_PRIOR"].append(ttest_ind(case_data.filter(case_data.is_not_null()),  
                                                       control_data.filter(control_data.is_not_null()), 
                                                       equal_var=False)[1])
                
    for objective_name in objectives:
        for outcome in outcomes: 
            for set_no in [0,1,2]:
                if set_no == 0: set_filter = train_filter
                elif set_no == 1: set_filter = valid_filter
                else: set_filter = test_filter
                results_cont["LAB_NAME"].append(lab_name)
                results_cont["SET"].append(set_names[set_no])
                results_cont["OBJECTIVE"].append(objective_name)
                results_cont["OUTCOME"].append(outcome)

                N_controls = labels.height-sum(labels.filter(set_filter)[outcome])
                results_cont["N_CONTROLS"].append(N_controls)
                N_cases = sum(labels.filter(set_filter)[outcome])
                results_cont["N_CASES"].append(N_cases)
        
                applied_data = objectives[objective_name](labels.filter(set_filter))

                if applied_data.height>0:
                    N_case_prior = (applied_data[outcome].value_counts().sort(outcome, descending=True)["count"][0])
                    N_control_prior = (applied_data[outcome].value_counts().sort(outcome, descending=True)["count"][1])
                else:
                    N_case_prior = 0
                    N_control_prior = 0
                results_cont["CASES_PRIOR"].append(N_case_prior)
                results_cont["CONTROLS_PRIOR"].append(N_control_prior)
            
                results_cont["PVAL_PRIOR"].append(proportions_ztest([N_case_prior, N_control_prior], [N_cases, N_controls])[1])
                
            display(pl.DataFrame(results_cont, strict=False).with_columns(
       (( (((pl.col.CASES_PRIOR/pl.col.N_CASES)*100).round(1)).cast(pl.Utf8)+"% (N="+pl.col.CASES_PRIOR.cast(pl.Utf8)) + ")").alias("CASES_PRIOR"),
       (( (((pl.col.CONTROLS_PRIOR/pl.col.N_CONTROLS)*100).round(1)).cast(pl.Utf8)+"% (N="+pl.col.CONTROLS_PRIOR.cast(pl.Utf8)) + ")").alias("CONTROLS_PRIOR"),
    
).filter(pl.col.OUTCOME=="y_MEAN_ABNORM", pl.col.SET=="Test"))

################################################################################################
############################# Saving ############################################################
################################################################################################
pl.DataFrame(results_bin).with_columns(
    pl.when(pl.col.N_CASES<=5).then(pl.lit(0)).otherwise(pl.col.N_CASES).alias("N_CASES"),
    pl.when(pl.col.N_MISSING_CASES<=5).then(pl.lit(0)).otherwise(pl.col.N_MISSING_CASES).alias("N_MISSING_CASES"),
    pl.when(pl.col.N_MISSING_CONTROLS<=5).then(pl.lit(0)).otherwise(pl.col.N_MISSING_CONTROLS).alias("N_MISSING_CONTROLS")
).with_columns(
        (pl.col.ALL_MEAN.round(2).cast(pl.Utf8) + " ± " + pl.col.ALL_SD.round(2).cast(pl.Utf8) + " (" + ((pl.col.N_MISSING/pl.col.N_TOTAL)*100).round(2).cast(pl.Utf8) + "%)").alias("ALL_MEAN_STR"),
        (pl.col.CASES_MEAN.round(2).cast(pl.Utf8) + " ± " + pl.col.CASES_SD.round(2).cast(pl.Utf8)+ " (" + ((pl.col.N_MISSING_CASES/pl.col.N_CASES)*100).round(2).cast(pl.Utf8) + "%)").alias("CASES_MEAN_STR"),
        (pl.col.CONTROLS_MEAN.round(2).cast(pl.Utf8) + " ± " + pl.col.CONTROLS_SD.round(2).cast(pl.Utf8)+ " (" + ((pl.col.N_MISSING_CONTROLS/pl.col.N_CONTROLS)*100).round(2).cast(pl.Utf8) + "%)").alias("CONTROLS_MEAN_STR")
).write_csv("/home/ivm/valid/results/comp_tables/cases_vs_controls_cont_2025-08-04.csv")

pl.DataFrame(results_cont, strict=False).with_columns(
    pl.when(pl.col.N_CASES<=5).then(pl.lit(0)).otherwise(pl.col.N_CASES).alias("N_CASES"),
    pl.when(pl.col.CASES_PRIOR<=5).then(pl.lit(0)).otherwise(pl.col.CASES_PRIOR).alias("CASES_PRIOR"),
    pl.when(pl.col.CONTROLS_PRIOR<=5).then(pl.lit(0)).otherwise(pl.col.CONTROLS_PRIOR).alias("CONTROLS_PRIOR")
).with_columns(
       (( (((pl.col.CASES_PRIOR/pl.col.N_CASES)*100).round(1)).cast(pl.Utf8)+"% (N="+pl.col.CASES_PRIOR.cast(pl.Utf8)) + ")").alias("CASES_PRIORS_STR"),
       (( (((pl.col.CONTROLS_PRIOR/pl.col.N_CONTROLS)*100).round(1)).cast(pl.Utf8)+"% (N="+pl.col.CONTROLS_PRIOR.cast(pl.Utf8)) + ")").alias("CONTROLS_PRIOR_STR"),
    
).write_csv("/home/ivm/valid/results/comp_tables/cases_vs_controls_bin_2025-08-04.csv")               