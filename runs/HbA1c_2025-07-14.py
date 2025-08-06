################################################################################################
############################# SETUP ############################################################
################################################################################################
! source /home/ivm/envs/valid_env/bin/activate
import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import *
from model_eval_utils import compare_models
# For shap
from plot_utils import get_plot_names
import pickle
import shap

%load_ext autoreload
%autoreload 2

base_path="/home/ivm/valid/data/processed_data"
from model_eval_utils import compare_models
goal="y_MEAN_ABNORM"
file_descr = "testv1_2022_w3"
lab_name = "hba1c"
data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/hba1c_d1_strong_2025-07-18.parquet")
base_date = datetime(2021,10,1)
train_dates = ["2025-07-18", "2025-07-22", "2025-07-23"]
def check_dated_path(file_path_start):
    for date in train_dates:
        if os.path.exists(file_path_start+date+"/preds_"+date+".parquet"):
            return True
    return False

def get_dated_path(file_path_start):
    for date in train_dates:
        if os.path.exists(file_path_start+date+"/preds_"+date+".parquet"):
            return file_path_start+date+"/preds_"+date+".parquet"
    return None

train_goals = {"logloss": "y_MEAN_ABNORM", "q75": "y_MEAN", "q25": "y_MEAN", "q10": "y_MEAN", "mae": "y_MEAN"}


lastval_long_filter = (pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID")).filter(pl.col.DATE.dt.year()<2020)["FINNGENID"]))
no_history_filter = (~pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date)["FINNGENID"]))
no_abnorm_filter = (~pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter(((pl.col.VALUE<43).all()).over("FINNGENID"))["FINNGENID"]))
thirty_filter = (((pl.col.EVENT_AGE>=30)&(pl.col.EVENT_AGE<40)))
fourty_filter = (((pl.col.EVENT_AGE>=40)&(pl.col.EVENT_AGE<50)))
fifty_filter = (((pl.col.EVENT_AGE>=50)&(pl.col.EVENT_AGE<60)))
sixty_filter = (((pl.col.EVENT_AGE>=60)&(pl.col.EVENT_AGE<=70)))
history_filter = (pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date)["FINNGENID"]))
test_filter = pl.col.SET==2

################################################################################################
############################# Initial prep #####################################################
################################################################################################
! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3004410 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=hba1c
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/hba1c_2025-07-18.parquet \
    --lab_name=hba1c \
    --fill_missing 1 \
    --dummies 18 37 51 -1 -1 \
    --ref_min 0.01 \
    --ref_max 200 \
    --main_unit mmol/mol \
    --abnorm_type strong \
    --plot 1

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=43054914 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=ogtt

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3018251 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=fgluc
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/fgluc_2025-07-18.parquet \
    --lab_name=fgluc \
    --fill_missing 1 \
    --dummies 3.8 5.5 6.8 1.9 22.2 \
    --ref_min 0.01 \
    --main_unit mmol/l \
    --plot 1

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3013826 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=gluc
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/gluc_2025-07-18.parquet \
    --lab_name=gluc \
    --fill_missing 1 \
    --dummies 3.6 5.5 7.8 1.8 24.3 \
    --ref_min 0.01 \
    --main_unit mmol/l \
    --plot 1

! python3 /home/ivm/valid/scripts/steps/step2_diags.py \
                --lab_name=hba1c \
                --res_dir=/home/ivm/valid/data/processed_data/step2_diags/  \
                --diag_regex="(^(E1[0-4]))|(^O24)" --med_regex="^A10" \
                --med_excl_regex="" \
                --diag_excl_regex="" \
                --fg_ver="R13"

################################################################################################
############################# Exclusions #######################################################
################################################################################################
diags_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/hba1c_R13_2025-07-18_diags.parquet")
meds_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/hba1c_R13_2025-07-18_meds.parquet")
hba1c_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/hba1c_d1_strong_2025-07-18.parquet")
fgluc_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/fgluc_d1_2025-07-18.parquet")
gluc_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/gluc_d1_2025-07-18.parquet")
ogtt_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step0_extract/ogtt_2025-07-18.parquet")

(hba1c_data
 .filter((~((pl.col.DATE<base_date)&(pl.col.ABNORM_CUSTOM==1)).any().over("FINNGENID")))
 .join(diags_data.select("FINNGENID", "DIAG_DATE", "DIAG"), on="FINNGENID", how="left")
 .filter(~(pl.col.DIAG_DATE<base_date).any().over("FINNGENID"))
 .drop("DIAG_DATE", "DIAG").unique()
 .join(meds_data.select("FINNGENID", "MED_DATE", "MED"), on="FINNGENID", how="left")
 .filter(~(pl.col.MED_DATE<base_date).any().over("FINNGENID"))
 .drop("MED_DATE", "MED").unique()
 .filter(~pl.col.FINNGENID.is_in(fgluc_diag["FINNGENID"]))
 .filter(~pl.col.FINNGENID.is_in(gluc_diag["FINNGENID"]))
 .filter(~pl.col.FINNGENID.is_in(ogtt_diag["FINNGENID"]))).write_parquet("/home/ivm/valid/data/processed_data/step2_diags/hba1c_d1_strong_2025-07-18_filtered_2025-07-18.parquet")

################################################################################################
############################# Labels ###########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; extra=d1_strong; diff=30; \
    date_1=2025-07-18; date_2=2025-07-18; \
    python3 /home/ivm/valid/scripts/steps/step3_labels_test_new.py \
        --data_path_full "$base_path"/step2_diags/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2".parquet \
        --res_dir "$base_path"/step3_labels/ \
        --lab_name "$lab_name" \
        --start_pred_date 2022-01-01 --end_pred_date 2022-12-31 \
        --min_age 30 --max_age 70 \
        --months_buffer 3 \
        --abnorm_type strong \
        --valid_pct 0.3 \
        --version v1

################################################################################################
############################# Extra data #######################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_data /home/ivm/valid/data/processed_data/step1_clean/"$lab_name_two"_"$extra_two".parquet \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --mean_impute 0

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_labs.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_lab /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/icds_r13_2025-06-06_min1p0pct_bin_onttop_2025-06-06.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ICD_THREE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 0 \
            --min_pct "1p0"

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/icds_r13_2025-06-06_min1p0pct_bin_onttop_2025-06-06.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ICD_THREE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 2020 \
            --min_pct "1p0"

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/atcs_r13_2025-06-12_min1p0pct_bin_onttop_2025-06-12.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ATC_FIVE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 0 \
            --min_pct "1p0"

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/atcs_r13_2025-06-12_min1p0pct_bin_onttop_2025-06-12.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ATC_FIVE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 2020 \
            --min_pct "1p0"
################################################################################################
############################# XGBoost ##########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --start_date 2021-10-01 \
            --pred_descriptor 1_clinpheno \
            --preds BMI SMOKE SBP DBP EDU EVENT_AGE SEX \
            --fg_ver r13 \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 2_lastval \
            --start_date 2021-10-01 \
            --preds IDX_QUANT_100 LAST_VAL_DIFF EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 2_sumstats \
            --start_date 2021-10-01 \
            --preds SUMSTATS LAST_VAL_DIFF EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --start_date 2021-10-01 \
            --pred_descriptor 3_twosumstats \
            --preds SECOND_SUMSTATS SUMSTATS LAST_VAL_DIFF EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_sumstats_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; date_5=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --start_date 2021-10-01 \
            --pred_descriptor 3_otherlabs \
            --preds S_MEAN LAB_MAT_MEAN EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_5".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 3_registry \
            --preds ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; date_5=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE EDU SBP DBP LAST_VAL_DIFF SECOND_SUMSTATS SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_5".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; date_5=2025-07-18;\
    GOAL=y_MEAN; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE EDU DBP SBP LAST_VAL_DIFF SECOND_SUMSTATS SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric mae  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --abnorm_extra_choice strong\
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_two=d1_2025-07-18; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; date_5=2025-07-18;\
    GOAL=y_MEAN; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE EDU DBP SBP LAST_VAL_DIFF SECOND_SUMSTATS SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1p0pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric q75  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --abnorm_extra_choice strong\
            --refit 1 \
            --n_trials 200

################################################################################################
############################# TLSTM ############################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step4_longitudina_2.py \
            --lab_name $lab_name \
            --preds LAB AGE SEX \
            --preds_name 1_lab \
            --data_path_dir "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --res_dir "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --goal "$GOAL" \
            --end_obs_date 2021-10-01 \
            --skip_rep_codes 0 \
            --quant_steps 20 36 40 41 42 43 44 45 46 47

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18; pred_descr=1_lab_manualquants \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_DL.py \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_long_"$pred_descr"_"$date_4" \
            --lab_name $lab_name \
            --pred_descriptor $pred_descr \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --model_name TLSTM \
            --refit 1 \
            --n_trials 100 \
            --train_epochs 20 \
            --goal "$GOAL" \
            --batch_size 256

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step4_longitudina_2.py \
            --lab_name $lab_name \
            --preds EXTRA_LAB LAB AGE SEX \
            --preds_name 2_lab_two \
            --data_path_dir "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --extra_lab_data_path /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --res_dir "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --goal "$GOAL" \
            --end_obs_date 2021-10-01 \
            --skip_rep_codes 0 \
            --quant_steps 20 36 40 41 42 43 44 45 46 47 \
            --extra_omop_ids "3013826"

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step4_longitudina_2.py \
            --lab_name $lab_name \
            --preds EXTRA_LAB LAB AGE SEX \
            --preds_name 2_lab_three \
            --data_path_dir "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --extra_lab_data_path /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --res_dir "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --goal "$GOAL" \
            --end_obs_date 2021-10-01 \
            --skip_rep_codes 0 \
            --quant_steps 20 36 40 41 42 43 44 45 46 47 \
            --extra_omop_ids "3013826" "3018251"

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=hba1c; lab_name_two=fgluc; extra=d1_strong; extra_labels=testv1_2022_w3;\
    date_1=2025-07-18; date_2=2025-07-18; date_3=2025-07-18; date_4=2025-07-18;\
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step4_longitudina_2.py \
            --lab_name $lab_name \
            --preds EXTRA_LAB LAB AGE SEX \
            --preds_name 2_lab_top \
            --data_path_dir "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --extra_lab_data_path /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --res_dir "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --goal "$GOAL" \
            --end_obs_date 2021-10-01 \
            --skip_rep_codes 0 \
            --quant_steps 20 36 40 41 42 43 44 45 46 47 \
            --extra_omop_ids "3013826" "3018251" "3048773" "3025839" "3006923" "3026910" "3010813" \
            --time_detail days

################################################################################################
############################# Evals ############################################################
from model_eval_utils import bootstrap_metric, bootstrap_difference, continuous_nri, bootstrap_nri
import sklearn.metrics as skm
import numpy as np
import os.path
from delong_utils import delong_roc_test

preds_descrs={"1_lab_manualquants": "lab sequences",
              "1_lab_manualquants_days": "lab sequences days",
              "2_lab_two_manualquants": "HbA1c+glucose sequences",
              "2_lab_three_manualquants":"HbA1c+(fasting)glucose sequences", 
              "2_lab_top_manualquants": "HbA1c+top labs",
              "1_clinpheno": "clinical phenotype", 
              "2_lastval": "last value",
              "2_sumstats": "sumstats", 
              "3_twosumstats": "two sumstats", 
              "3_otherlabs": "other labs", 
              "3_registry": "registry data", 
              "4_all": "all data",
              "5_icd": "ICD data", 
              "5_atc": "ATC data", 
              "6_atc2020": "ATC data >= 2020"}
metrics = ["logloss", "q75", "mae"]

train_goals = {"logloss": "y_MEAN_ABNORM", "q75": "y_MEAN", "mae": "y_MEAN"}
goal_names = {"y_MEAN_ABNORM": "Mean abnormal", "y_NEXT_ABNORM": "Next abnormal"}
goal_names_extra = {"y_MEAN_ABNORM": "Mean abnormal", "y_NEXT_ABNORM": "Next abnormal", "y_MEAN": "Mean"}
filters = {"All": True, 
           "History": history_filter, 
           "All normal": no_abnorm_filter|no_history_filter, 
           "Last <2020": lastval_long_filter|no_history_filter, 
           "No history": no_history_filter, 
           "30-40": thirty_filter, 
           "40-50": fourty_filter, 
           "50-60": fifty_filter, 
           "60-70": sixty_filter}
set_names = {0: "Train", 1: "Valid", 2: "Test"}
train_goal = "y_MEAN_ABNORM"
### ---- same as eGFR ----------------
# Except bug fix for F1 comp p-values
results = pl.DataFrame()

for combo_1, combo_2 in no_dups_combos:
    metric_1 = combo_1.split("_")[1]
    if metric_1 == "GRU": metric_1 = "logloss"
    file_path_1 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_1]+"/"+combo_1+"/models/"+lab_name+"/")
    metric_2 = combo_2.split("_")[1]
    if metric_2 == "GRU": metric_2 = "logloss"
    file_path_2 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_2]+"/"+combo_2+"/models/"+lab_name+"/")

    if metric_1 != "logloss" or metric_2 != "logloss":
        preds_1 = pl.read_parquet(file_path_1)
        preds_2 = pl.read_parquet(file_path_2)
        # Binarizing based on training with train goal
        if metric_1 == "logloss":
            precision_, recall_, proba = skm.precision_recall_curve(preds_1.filter(pl.col.SET==0)[train_goal], preds_1.filter(pl.col.SET==0)["ABNORM_PROBS"])
            optimal_proba_cutoff_1 = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
            preds_1 = preds_1.with_columns((pl.col.ABNORM_PROBS>optimal_proba_cutoff_1).alias("ABNORM_PREDS"))
        else:
            optimal_proba_cutoff_1 = 0.0
        # Binarizing based on training with train goal
        if metric_2 == "logloss":
            precision_, recall_, proba = skm.precision_recall_curve(preds_2.filter(pl.col.SET==0)[train_goal], preds_2.filter(pl.col.SET==0)["ABNORM_PROBS"])
            optimal_proba_cutoff_2 = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
            preds_2 = preds_2.with_columns((pl.col.ABNORM_PROBS>optimal_proba_cutoff_2).alias("ABNORM_PREDS"))
        else:
            optimal_proba_cutoff_2 = 0.0
            
        for crnt_filter_name in filters:
            for set_no in set_names:
                descriptors = {"MODEL_1": combo_1, "CUT_1": optimal_proba_cutoff_1,  "MODEL_2": combo_2,"CUT_2": optimal_proba_cutoff_2, "SET": set_names[set_no]}
                for goal_name in goal_names:
                    preds = preds_1.select("FINNGENID", "SET", "EVENT_AGE", goal_name, "ABNORM_PREDS").join(preds_2.select("FINNGENID", "ABNORM_PREDS"), on="FINNGENID", how="left")
                    crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter_name])
                    N_cases_1 = crnt_preds["ABNORM_PREDS"].sum()
                    if N_cases_1 < 5:
                        N_cases_1 = 0
                        crnt_preds = crnt_preds.with_columns((pl.lit(0)*crnt_preds.height).alias("ABNORM_PREDS"))
                    descriptors[goal_name+"_N_CASE_MODEL_1"] = N_cases_1
    
                    N_cases_2 = crnt_preds["ABNORM_PREDS_right"].sum()
                    if N_cases_2 < 5:
                        N_cases_2 = 0
                        crnt_preds = crnt_preds.with_columns((pl.lit(0)*crnt_preds.height).alias("ABNORM_PREDS_right"))
                    descriptors[goal_name+"_N_CASE_MODEL_2"] = N_cases_2
                        
                    diff_est, lowci, highci, pval_diff, f1_1, f1_2 = bootstrap_difference(metric_func = (lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0)),
                                                                                  preds_1=crnt_preds["ABNORM_PREDS"].to_numpy(), 
                                                                                  preds_2=crnt_preds["ABNORM_PREDS_right"].to_numpy(),
                                                                                  obs=crnt_preds[goal_name].to_numpy(),
                                                                                  n_boots=100)
                    descriptors["FILTER"] = crnt_filter_name
                    descriptors[goal_name+"_F1Diff"]=diff_est
                    descriptors[goal_name+"_F1Diff_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"
                    descriptors[goal_name+"_F1Diff_Pvalue"]=pval_diff
    
                    diff_est, lowci, highci, pval_diff, f1_1, f1_2 = bootstrap_difference(metric_func = (lambda x, y: skm.precision_score(x, y, average="macro", zero_division=0)),
                                                                                  preds_1=crnt_preds["ABNORM_PREDS"].to_numpy(), 
                                                                                  preds_2=crnt_preds["ABNORM_PREDS_right"].to_numpy(),
                                                                                  obs=crnt_preds[goal_name].to_numpy(),
                                                                                  n_boots=100)
                    descriptors[goal_name+"_PrecisionDiff"]=diff_est
                    descriptors[goal_name+"_PrecisionDiff_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"
                    descriptors[goal_name+"_PrecisionDiff_Pvalue"]=pval_diff
    
                    diff_est, lowci, highci, pval_diff, f1_1, f1_2 = bootstrap_difference(metric_func = (lambda x, y: skm.recall_score(x, y, average="macro", zero_division=0)),
                                                                                  preds_1=crnt_preds["ABNORM_PREDS"].to_numpy(), 
                                                                                  preds_2=crnt_preds["ABNORM_PREDS_right"].to_numpy(),
                                                                                  obs=crnt_preds[goal_name].to_numpy(),
                                                                                  n_boots=100)
                    descriptors[goal_name+"_RecallDiff"]=diff_est
                    descriptors[goal_name+"_RecallDiff_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"
                    descriptors[goal_name+"_RecallDiff_Pvalue"]=pval_diff
                descriptors = pl.DataFrame(descriptors)
                results = pl.concat([results, 
                                     descriptors.with_columns(pl.col(col_name).cast(pl.Float32) for col_name, dtype in zip(descriptors.columns, descriptors.dtypes) if dtype.is_numeric())])
        display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_f1etc_pvals_"+get_date()+".csv")