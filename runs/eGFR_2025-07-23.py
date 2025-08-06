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

goal="y_MEAN_ABNORM"
file_descr = "testv1_2022_w3"
lab_name = "egfr"
base_date = datetime(2021,10,1)

data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_herold-part_ld_2025-07-23.parquet")
train_dates = ["2025-07-23", "2025-07-24"]
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
history_filter = (pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date)["FINNGENID"]))
last_norm_filter = (pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID")).filter(pl.col.ABNORM_CUSTOM<1)["FINNGENID"]))
no_abnorm_filter = (~pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter(pl.col.ABNORM_CUSTOM==1)["FINNGENID"]))
thirty_filter = (((pl.col.EVENT_AGE>=30)&(pl.col.EVENT_AGE<40)))
fourty_filter = (((pl.col.EVENT_AGE>=40)&(pl.col.EVENT_AGE<50)))
fifty_filter = (((pl.col.EVENT_AGE>=50)&(pl.col.EVENT_AGE<60)))
sixty_filter = (((pl.col.EVENT_AGE>=60)&(pl.col.EVENT_AGE<=70)))
test_filter = pl.col.SET==2
valid_filer = pl.col.SET==1
train_filter = pl.col.SET==0

################################################################################################
############################# Initial prep #####################################################
################################################################################################
! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3020564 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=krea
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/krea_2025-07-21.parquet \
    --lab_name=egfr \
    --fill_missing 1 \
    --dummies 48 71 115 -1 625 \
    --abnorm_type=herold-part \
    --main_unit umol/l \
    --plot 1 \
    --keep_last_of_day 1 \
    --ref_min 2 \
    --max_z 4

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3020682 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=uacr
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/uacr_2025-07-22.parquet \
    --lab_name=uacr \
    --fill_missing 1 \
    --dummies -1 0.6 16.1 -1 -1 \
    --main_unit mg/mmol \
    --plot 1 \
    --keep_last_of_day 1 \
    --ref_min 0.01 

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3030366 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=cystc
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/cystc_2025-07-21.parquet \
    --lab_name=cystc \
    --fill_missing 1 \
    --dummies 0.61 0.89 1.79 -1 -1 \
    --abnorm_type=herold-part \
    --main_unit mg/l \
    --plot 1 \
    --keep_last_of_day 1 \
    --ref_min 2

! python3 /home/ivm/valid/scripts/steps/step2_diags.py \
                --lab_name=egfr \
                --res_dir=/home/ivm/valid/data/processed_data/step2_diags/  \
                --diag_regex="(^N18)|(^N19)|(^Z905)" --med_regex=""\
                --diag_excl_regex="" \
                --med_excl_regex="" \
                --fg_ver="R13"

################################################################################################
############################# Exclusions #######################################################
################################################################################################
diags_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/egfr_R13_2025-07-22_diags.parquet")
egfr_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/egfr_d1_herold-part_ld_2025-07-23.parquet")
cystc_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/cystc_d1_herold-part_ld_2025-07-23.parquet")
uacr_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/uacr_d1_ld_2025-07-22.parquet")

from diag_utils import get_abnorm_start_dates, get_data_diags
egfr_diag = get_data_diags(get_abnorm_start_dates(egfr_data), 90)
cystc_diag = get_data_diags(get_abnorm_start_dates(cystc_data), 90)
uacr_diag = get_data_diags(get_abnorm_start_dates(uacr_data), 90)


(egfr_data
  .join(egfr_diag.select("FINNGENID", "DATA_DIAG_DATE").unique(), on="FINNGENID", how="left")
  .filter(~(pl.col.DATA_DIAG_DATE<base_date)|pl.col.DATA_DIAG_DATE.is_null())
  .drop("DATA_DIAG_DATE")         
  .join(diags_data.select("FINNGENID", "DIAG_DATE", "DIAG"), on="FINNGENID", how="left")
  .filter(~(pl.col.DIAG_DATE<base_date).any().over("FINNGENID"))
  .drop("DIAG_DATE", "DIAG").unique()
  .join(cystc_diag.select("FINNGENID", "DATA_DIAG_DATE").unique(), on="FINNGENID", how="left")
  .filter(~(pl.col.DATA_DIAG_DATE<base_date)|pl.col.DATA_DIAG_DATE.is_null())
  .drop("DATA_DIAG_DATE")
  .join(uacr_diag.select("FINNGENID", "DATA_DIAG_DATE").unique(), on="FINNGENID", how="left")
  .filter(~(pl.col.DATA_DIAG_DATE<base_date)|pl.col.DATA_DIAG_DATE.is_null())
  .drop("DATA_DIAG_DATE")
).write_parquet("/home/ivm/valid/data/processed_data/step2_diags/egfr_d1_herold-part_ld_2025-07-23_filtered_2025-07-23.parquet")

################################################################################################
############################# Labels ###########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R12; lab_name=egfr; extra=d1_herold-part_ld; diff=90; \
    date_1=2025-07-23; date_2=2025-07-23; date_excl=2025-07-16; \
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
    fg_ver=R12; lab_name=egfr; extra=d1_herold-part_ld; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --mean_impute 0

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R12; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=uacr; extra_two=d1_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_data /home/ivm/valid/data/processed_data/step1_clean/"$lab_name_two"_"$extra_two"_"$date_two".parquet \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R12; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_data /home/ivm/valid/data/processed_data/step1_clean/"$lab_name_two"_"$extra_two"_"$date_two".parquet \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R12; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_labs.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_lab /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/icds_r13_2025-06-06_min1p0pct_2025-06-06.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ICD_THREE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 0 \
            --min_pct 1

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/atcs_r13_2025-06-12_min1p0pct_2025-06-12.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"\
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --col_name ATC_FIVE \
            --time 0 \
            --bin_count 1 \
            --months_before 0 \
            --start_year 0 \
            --min_pct 1

################################################################################################
############################# XGBoost ##########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 1_clinpheno \
            --start_date 2021-10-01 \
            --preds BMI SMOKE SBP DBP EDU EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
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
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=uacr; extra_two=d1_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --start_date 2021-10-01 \
            --pred_descriptor 3_twosumstats_2 \
            --preds SECOND_SUMSTATS SUMSTATS LAST_VAL_DIFF EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
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
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 3_otherlabs \
            --preds S_MEAN LAB_MAT_MEAN EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 3_registry \
            --preds ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN_ABNORM; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE SBP DBP EDU LAST_VAL_DIFF S_MEAN S_IDX_QUANT_100 SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric logloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE SBP DBP EDU LAST_VAL_DIFF S_MEAN S_IDX_QUANT_100 SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric mae  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE SBP DBP EDU LAST_VAL_DIFF S_MEAN S_IDX_QUANT_100 SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric q25  \
            --reweight 0 \
            --n_boots 500 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
    GOAL=y_MEAN; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_XGB.py \
            --lab_name $lab_name \
            --lab_name_two $lab_name_two \
            --lr 0.4 \
            --pred_descriptor 4_all \
            --start_date 2021-10-01 \
            --preds BMI SMOKE SBP DBP EDU LAST_VAL_DIFF S_MEAN S_IDX_QUANT_100 SUMSTATS LAB_MAT_MEAN ICD_MAT ATC_MAT EVENT_AGE SEX \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_sumstats_noimpute_"$date_4".parquet \
            --file_path_second_sumstats "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two"_"$date_two"_sumstats_"$date_4".parquet \
            --file_path_labs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labs_"$date_4".parquet \
            --file_path_icds "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_icds_1pct_bin_"$lab_name"_"$date_4".parquet \
            --file_path_atcs "$base_path"/step4_data/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_atcs_1pct_bin_"$lab_name"_"$date_4".parquet \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --goal "$GOAL" \
            --run_step0 0 \
            --metric q10  \
            --reweight 0 \
            --n_boots 500 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

################################################################################################
############################# TLSTM ############################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-23 \
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
            --quant_steps 2 20 40 50 52 54 56 58 60 62 64 66 68 70 90 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=egfr; extra=d1_herold-part_ld; lab_name_two=cystc; extra_two=d1_herold-part_ld; date_two=2025-07-22; extra_labels=testv1_2022_w3;\
    date_1=2025-07-23; date_2=2025-07-23; date_3=2025-07-23;date_4=2025-07-24 \
    GOAL=y_MEAN_ABNORM; pred_descr=1_lab_manualquants_month ; \
        python3 /home/ivm/valid/scripts/steps/step5_fit_DL.py \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path "$base_path"/step4_data/"$extra_labels"/"$GOAL"/tlstm/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_long_"$pred_descr"_"$date_4" \
            --lab_name $lab_name \
            --pred_descriptor $pred_descr \
            --res_dir "$base_path"/step5_predict/"$extra_labels"/"$GOAL"/ \
            --model_name TLSTM \
            --refit 1 \
            --n_trials 50 \
            --train_epochs 10 \
            --goal "$GOAL" \
            --batch_size 256

################################################################################################
############################# Evals ############################################################
################################################################################################
from model_eval_utils import bootstrap_metric, bootstrap_difference, continuous_nri, bootstrap_nri
import sklearn.metrics as skm
import numpy as np
import os.path
from delong_utils import delong_roc_test

preds_descrs={"1_lab_manualquants_month": "lab sequence", "1_clinpheno": "clinical phenotype", "2_lastval": "last value", "2_sumstats": "sumstats", "3_twosumstats": "eGFR+Cystatin C", "3_twosumstats_2": "eGFR+UACR", "3_otherlabs": "other labs", "3_registry": "registry data", "4_all": "all data", "5_icd": "ICD data", "5_atc": "ATC data"}
metrics = ["logloss", "q10", "q25", "mae"]

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

metric = "logloss"
results = pl.DataFrame()

options = ["xgb_"+metric+"_"+pred_descr for pred_descr in preds_descrs for metric in metrics if check_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/xgb_"+metric+"_"+pred_descr+"/models/"+lab_name+"/")]
for pred_descr in preds_descrs:
    print(("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/TLSTM_GRU_"+pred_descr+"/models/"+lab_name+"/"))
    if check_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/TLSTM_GRU_"+pred_descr+"/models/"+lab_name+"/"):
        options.append("TLSTM_GRU_"+pred_descr)

combos = [(x,y) for x in options for y in options if x!=y]
no_dups_combos = []
for combo_1, combo_2 in combos:
    if (combo_2, combo_1) not in no_dups_combos and ((combo_1, combo_2) not in no_dups_combos):
        no_dups_combos.append( (combo_1, combo_2) )


        metric = "logloss"
results = pl.DataFrame()
for combo_1, combo_2 in no_dups_combos:
    metric_1 = combo_1.split("_")[1]
    if metric_1 == "GRU": metric_1 = "logloss"
    file_path_1 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_1]+"/"+combo_1+"/models/"+lab_name+"/")
    metric_2 = combo_2.split("_")[1]
    if metric_2 == "GRU": metric_2 = "logloss"
    file_path_2 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_2]+"/"+combo_2+"/models/"+lab_name+"/")

    preds_1 = pl.read_parquet(file_path_1)
    preds_2 = pl.read_parquet(file_path_2)

    if metric_1 == "logloss" and metric_2 == "logloss":
        for set_no in set_names:
            descriptors = {"MODEL_1": combo_1, "MODEL_2": combo_2, "SET": set_names[set_no]}
            for crnt_filter_name in {"History": history_filter}:
                descriptors["FILTER"] = crnt_filter_name

                for goal_name in goal_names:
                    preds = preds_1.select("FINNGENID", "SET", "EVENT_AGE", goal_name, "ABNORM_PROBS").join(preds_2.select("FINNGENID", "ABNORM_PROBS"), on="FINNGENID", how="left")
                    crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter_name])

                    ### P-values for AUCs with DeLong
                    pval_diff = 10**delong_roc_test(crnt_preds[goal_name].to_numpy(), crnt_preds["ABNORM_PROBS"].to_numpy(), crnt_preds["ABNORM_PROBS_right"].to_numpy())[0]
                    descriptors[goal_name+"_AUCDiff_Pvalue"]=pval_diff

                    ### P-values for Average Precision with Bootstrapping
                    diff_est, lowci, highci, pval_diff, avg_1, avg_2 = bootstrap_difference(metric_func = (skm.average_precision_score),
                                                                              preds_1=crnt_preds["ABNORM_PROBS"].to_numpy(), 
                                                                              preds_2=crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                                              obs=crnt_preds[goal_name].to_numpy(),
                                                                              n_boots=100)
                    descriptors[goal_name+"_AvgPrecDiff_Pvalue"]=pval_diff

                    ### NRI with CI measure of if new model is better at reclassification <0 -> worst and >0 -> better
                    nri, lowci, highci = bootstrap_nri(continuous_nri, 
                                                       crnt_preds[goal_name].to_numpy(), 
                                                       crnt_preds["ABNORM_PROBS"].to_numpy(),
                                                       crnt_preds["ABNORM_PROBS_right"].to_numpy(),
                                                       n_boots=100)
                    descriptors[goal_name+"_NRI"]=nri
                    descriptors[goal_name+"_NRI_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"

                results = pl.concat([results, pl.DataFrame(descriptors)])
    display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_aucetc_pvals_"+get_date()+".csv")

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
        # Binarizing based on training with train goal
        if metric_2 == "logloss":
            precision_, recall_, proba = skm.precision_recall_curve(preds_2.filter(pl.col.SET==0)[train_goal], preds_2.filter(pl.col.SET==0)["ABNORM_PROBS"])
            optimal_proba_cutoff_2 = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
            preds_2 = preds_2.with_columns((pl.col.ABNORM_PROBS>optimal_proba_cutoff_2).alias("ABNORM_PREDS"))
    
            
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
    
                results = pl.concat([results, pl.DataFrame(descriptors)])
        display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_f1etc_pvals_"+get_date()+".csv")

results = pl.DataFrame()

for crnt_option in options:
    ### Getting info
    metric = crnt_option.split("_")[1]
    if metric == "GRU": metric = "logloss"
    if metric != "logloss": continue
    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
        
    # Getting data
    file_path = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/"+crnt_option+"/models/"+lab_name+"/")
    if not file_path: continue
    date = file_path.split(".")[0].split("preds_")[1]
    preds = pl.read_parquet(file_path)

    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
    descriptors = {"Date": date, 
                   "Model": mdl_name, 
                   "File Description": file_descr, 
                   "Predictors": preds_descrs[crnt_pred_descr],
                   "Outcome": goal_names[train_goal]}
    
    for set_no in set_names.keys():
        descriptors["SET"] = set_names[set_no]
        for goal in ["y_MEAN_ABNORM", "y_NEXT_ABNORM"]:
            N_total = preds.filter(pl.col.SET==set_no).height
            N_cases = preds.filter(pl.col.SET==set_no).filter(pl.col(goal)==1).height
            descriptors[goal+"_N"] = N_cases
            AUC = bootstrap_metric(skm.roc_auc_score, 
                                   preds.filter(pl.col.SET==set_no)[goal],
                                   preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[goal+"_AUC"] = AUC[0]
            descriptors[goal+"_AUC_CI"] = "("+str(round(AUC[1], 2))+ "-"+ str(round(AUC[2], 2)) + ")"
    
            averagePrec = bootstrap_metric(skm.average_precision_score, 
                                   preds.filter(pl.col.SET==set_no)[goal],
                                   preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[goal+"_avgPrec"] = averagePrec[0]
            descriptors[goal+"_avgPrec_CI"] = "("+str(round(averagePrec[1], 2))+ "-"+ str(round(averagePrec[2], 2)) + ")"

            descriptors[goal+"_Brier"] = skm.brier_score_loss(preds.filter(pl.col.SET==set_no)[goal], preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"])
    
        results = pl.concat([results, pl.DataFrame(descriptors)])
display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_aucsetc_"+get_date()+".csv")

results = pl.DataFrame()

for crnt_option in options:
    ### Getting info
    metric = crnt_option.split("_")[1]
    if metric == "GRU": metric = "logloss"
    if metric != "logloss": continue

    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
        
    # Getting data
    file_path = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/"+crnt_option+"/models/"+lab_name+"/")
    if not file_path: continue
    date = file_path.split(".")[0].split("preds_")[1]
    preds = pl.read_parquet(file_path)

    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
    descriptors = {"Date": date, 
                   "Model": mdl_name, 
                   "File Description": file_descr, 
                   "Predictors": preds_descrs[crnt_pred_descr],
                   "Outcome": goal_names[train_goal]}
    
    for set_no in set_names.keys():
        descriptors["SET"] = set_names[set_no]
        for crnt_filter in filters.keys():
            crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter])
            N_total = crnt_preds.filter(pl.col.SET==set_no).height
            N_cases = crnt_preds.filter(pl.col.SET==set_no).filter(pl.col(goal)==1).height
            descriptors[crnt_filter+"_N"] = N_cases
            AUC = bootstrap_metric(skm.roc_auc_score, 
                                   crnt_preds.filter(pl.col.SET==set_no)[goal],
                                   crnt_preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[crnt_filter+"_AUC"] = AUC[0]
            descriptors[crnt_filter+"_AUC_CI"] = "("+str(round(AUC[1], 2))+ "-"+ str(round(AUC[2], 2)) + ")"
    
            averagePrec = bootstrap_metric(skm.average_precision_score, 
                                   crnt_preds.filter(pl.col.SET==set_no)[goal],
                                   crnt_preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[crnt_filter+"_avgPrec"] = averagePrec[0]
            descriptors[crnt_filter+"_avgPrec_CI"] = "("+str(round(averagePrec[1], 2))+ "-"+ str(round(averagePrec[2], 2)) + ")"


            descriptors[goal+"_Brier"] = skm.brier_score_loss(preds.filter(pl.col.SET==set_no)[goal], 
                                                              preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"])
    
        results = pl.concat([results, pl.DataFrame(descriptors)])
display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_aucsetc_filtered_"+get_date()+".csv")

results = pl.DataFrame()

for crnt_option in options:
    ### Getting info
    metric = crnt_option.split("_")[1]
    if metric == "GRU": metric = "logloss"
    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
        
    # Getting data
    file_path = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/"+crnt_option+"/models/"+lab_name+"/")
    if not file_path: continue
    date = file_path.split(".")[0].split("preds_")[1]
    preds = pl.read_parquet(file_path)
    
    # Binarizing based on training with train goal
    if metric == "logloss":
        precision_, recall_, proba = skm.precision_recall_curve(preds.filter(pl.col.SET==0)[train_goal], preds.filter(pl.col.SET==0)["ABNORM_PROBS"])
        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
    else:
        optimal_proba_cutoff = None

    # Binarizing
    descriptors = {"Date": date, 
                       "Model": mdl_name, 
                       "File Description": file_descr, 
                       "Predictors": preds_descrs[crnt_pred_descr],
                       "Outcome": goal_names[train_goal],
                       "Metric": metric,
                       "Cut-off": optimal_proba_cutoff}
    #### Goint through sets
    for set_no in set_names.keys():
        descriptors["SET"] = set_names[set_no]
        #### And different goals for prediction
        for goal in ["y_MEAN_ABNORM", "y_NEXT_ABNORM"]:
            crnt_preds = preds.filter(pl.col.SET==set_no)
            if metric == "logloss":
                case_preds = crnt_preds["ABNORM_PROBS"]>optimal_proba_cutoff
            else:
                case_preds = crnt_preds["ABNORM_PREDS"]
                
            N_total = crnt_preds.height
            N_cases = crnt_preds.filter(pl.col(goal)==1).height
            N_true = np.logical_and(crnt_preds[goal].to_numpy(), case_preds.to_numpy()).sum()
            if N_true < 5: 
                N_true = 0
                case_preds = np.zeros(N_total)
            descriptors[goal+"_N"] = N_cases
            descriptors[goal+"_NTP"] = N_true
    
            F1 = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), 
                                           crnt_preds[goal],
                                           case_preds,
                                           n_boots=100)
            descriptors[goal+"_F1"] = F1[0]
            descriptors[goal+"_F1_CI"] = "("+str(round(F1[1], 2))+ "-"+ str(round(F1[2], 2)) + ")"
            
            precision = bootstrap_metric(lambda x, y: skm.precision_score(x, y, average="macro", zero_division=0), 
                                           crnt_preds[goal],
                                           case_preds,
                                           n_boots=100)
            descriptors[goal+"_Precision"] = precision[0]
            descriptors[goal+"_Precision_CI"] = "("+str(round(precision[1], 2))+ "-"+ str(round(precision[2], 2)) + ")"
        
            recall = bootstrap_metric(lambda x, y: skm.recall_score(x, y, average="macro", zero_division=0), 
                                           crnt_preds[goal],
                                           case_preds,
                                           n_boots=100)
            descriptors[goal+"_Recall"] = recall[0]
            descriptors[goal+"_Recall_CI"] = "("+str(round(recall[1], 2))+ "-"+ str(round(recall[2], 2)) + ")"

        results = pl.concat([results, pl.DataFrame(descriptors)])
    display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_f1setc_"+get_date()+".csv")

results = pl.DataFrame()

for crnt_option in options:
    ### Getting info
    metric = crnt_option.split("_")[1]
    if metric == "GRU": metric = "logloss"
    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
        
    # Getting data
    file_path = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/"+crnt_option+"/models/"+lab_name+"/")
    if not file_path: continue
    date = file_path.split(".")[0].split("preds_")[1]
    preds = pl.read_parquet(file_path)
    
    # Binarizing based on training with train goal
    if metric == "logloss":
        precision_, recall_, proba = skm.precision_recall_curve(preds.filter(pl.col.SET==0)[train_goal], preds.filter(pl.col.SET==0)["ABNORM_PROBS"])
        optimal_proba_cutoff = sorted(list(zip(np.abs(precision_ - recall_), proba)), key=lambda i: i[0], reverse=False)[0][1]
    else:
        optimal_proba_cutoff = None

    # Binarizing
    descriptors = {"Date": date, 
                       "Model": mdl_name, 
                       "File Description": file_descr, 
                       "Predictors": preds_descrs[crnt_pred_descr],
                       "Outcome": goal_names[train_goal],
                       "Metric": metric,
                       "Cut-off": optimal_proba_cutoff}
    #### Goint through sets
    for set_no in set_names.keys():
        for crnt_filter in filters.keys():
            crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter])
            if metric == "logloss":
                case_preds = crnt_preds["ABNORM_PROBS"]>optimal_proba_cutoff
            else:
                case_preds = crnt_preds["ABNORM_PREDS"]
        
            descriptors["SET"] = set_names[set_no]
            N_total = crnt_preds.height
            N_cases = crnt_preds.filter(pl.col(goal)==1).height
            N_true = np.logical_and(crnt_preds[goal].to_numpy(), case_preds.to_numpy()).sum()
            if N_true < 5: 
                N_true = 0
                case_preds = np.zeros(N_total)  
            descriptors[crnt_filter+"_N"] = N_cases
            descriptors[crnt_filter+"_NTP"] = N_true
    
            F1 = bootstrap_metric(lambda x, y: skm.f1_score(x, y, average="macro", zero_division=0), 
                                      crnt_preds[goal],
                                      case_preds,
                                      n_boots=100)
            descriptors[crnt_filter+"_F1"] = F1[0]
            descriptors[crnt_filter+"_F1_CI"] = "("+str(round(F1[1], 2))+ "-"+ str(round(F1[2], 2)) + ")"
            
            precision = bootstrap_metric(lambda x, y: skm.precision_score(x, y, average="macro", zero_division=0), 
                                             crnt_preds[goal],
                                             case_preds,
                                             n_boots=100)
            descriptors[crnt_filter+"_Precision"] = precision[0]
            descriptors[crnt_filter+"_Precision_CI"] = "("+str(round(precision[1], 2))+ "-"+ str(round(precision[2], 2)) + ")"
        
            recall = bootstrap_metric(lambda x, y: skm.recall_score(x, y, average="macro", zero_division=0), 
                                          crnt_preds[goal],
                                          case_preds,
                                          n_boots=100)
            descriptors[crnt_filter+"_Recall"] = recall[0]
            descriptors[crnt_filter+"_Recall_CI"] = "("+str(round(recall[1], 2))+ "-"+ str(round(recall[2], 2)) + ")"
        results = pl.concat([results, pl.DataFrame(descriptors)])
    display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_f1setc_filtered_"+get_date()+".csv")
