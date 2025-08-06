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
base_date = datetime(2021,10,1)
goal="y_MEAN_ABNORM"
file_descr = "testv2_2022_w3"
lab_name = "tsh"
data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/tsh_d1_multi_2025-07-18.parquet")
train_dates = ["2025-08-01", "2025-08-04"]
def check_dated_path(file_path_start):
    for date in train_dates:
        if os.path.exists(file_path_start+date+"/model_"+date+".pkl"):
            return True
    return False

def get_dated_path(file_path_start):
    for date in train_dates:
        if os.path.exists(file_path_start+date+"/preds_"+date+".parquet"):
            return file_path_start+date+"/preds_"+date+".parquet"
        if os.path.exists(file_path_start+date+"/model_"+date+".pkl"):
            return file_path_start+date+"/model_"+date+".pkl"
    return None

train_goals = {"logloss": "y_MEAN_ABNORM","mlogloss": "y_MEAN_ABNORM", "q75": "y_MEAN", "q25": "y_MEAN", "q10": "y_MEAN", "mae": "y_MEAN", "q90": "y_MEAN"}

lastval_long_filter = (pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID")).filter(pl.col.DATE.dt.year()<2020)["FINNGENID"]))
no_history_filter = (~pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date)["FINNGENID"]))
no_abnorm_filter = (~pl.col.FINNGENID.is_in(data.filter(pl.col.DATE<base_date).filter(((pl.col.VALUE<2.5).all()).over("FINNGENID"))["FINNGENID"]))
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
    --omop=3009201 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=tsh
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/tsh_2025-07-18.parquet \
    --lab_name=tsh \
    --fill_missing 1 \
    --dummies 0.19 1.74 4.63 -1 126 \
    --abnorm_type multi \
    --max_z 60 \
    --main_unit mu/l \
    --plot 1

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3008486 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=t4
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/t4_2025-07-18.parquet \
    --lab_name=t4 \
    --fill_missing 1 \
    --dummies 10 15.1 22.4 3.4 53 \
    --ref_min 0.01 \
    --main_unit pmol/l \
    --plot 1 \
    --max_z 32

! python3 /home/ivm/valid/scripts/steps/step0_extract.py \
    --omop=3026989 \
    --res_dir=/home/ivm/valid/data/processed_data/step0_extract/ \
    --lab_name=t3
! python3 /home/ivm/valid/scripts/steps/step1_clean.py \
    --res_dir=/home/ivm/valid/data/processed_data/step1_clean/ \
    --file_path=/home/ivm/valid/data/processed_data/step0_extract/t3_2025-07-18.parquet \
    --lab_name=t3 \
    --fill_missing 1 \
    --dummies 2.7 4.7 7.1 -1 -1 \
    --ref_min 0.01 \
    --main_unit pmol/l \
    --plot 1 \
    --max_z 25

! python3 /home/ivm/valid/scripts/steps/step2_diags.py \
                --lab_name=tsh \
                --res_dir=/home/ivm/valid/data/processed_data/step2_diags/  \
                --diag_regex="^E0[0-7]" --med_regex="^H03" \
                --med_excl_regex="" \
                --diag_excl_regex="" \
                --fg_ver="R13"
################################################################################################
############################# Exclusions #######################################################
################################################################################################
diags_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/tsh_R13_2025-08-01_diags.parquet")
meds_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step2_diags/tsh_R13_2025-08-01_meds.parquet")
tsh_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/tsh_d1_multi_2025-07-22.parquet")
t4_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/t4_d1_2025-07-21.parquet")
t3_data = pl.read_parquet("/home/ivm/valid/data/processed_data/step1_clean/t3_d1_2025-07-21.parquet")

(tsh_data
 .filter((~((pl.col.DATE<base_date)&(pl.col.ABNORM_CUSTOM!=0)).any().over("FINNGENID")))
 .join(diags_data.select("FINNGENID", "DIAG_DATE", "DIAG"), on="FINNGENID", how="left")
 .filter(~(pl.col.DIAG_DATE<base_date).any().over("FINNGENID"))
 .drop("DIAG_DATE", "DIAG").unique()
 .join(meds_data.select("FINNGENID", "MED_DATE", "MED"), on="FINNGENID", how="left")
 .filter(~(pl.col.MED_DATE<base_date).any().over("FINNGENID"))
 .drop("MED_DATE", "MED").unique()
 .filter(~pl.col.FINNGENID.is_in(t4_diag["FINNGENID"]))
 .filter(~pl.col.FINNGENID.is_in(t3_data["FINNGENID"]))
).write_parquet("/home/ivm/valid/data/processed_data/step2_diags/tsh_d1_multi_2025-07-18_filtered_2025-08-01.parquet")

################################################################################################
############################# Labels ###########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; extra=d1_multi;\
    date_1=2025-07-18; date_2=2025-08-01; \
    python3 /home/ivm/valid/scripts/steps/step3_labels_test_new.py \
        --data_path_full "$base_path"/step2_diags/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2".parquet \
        --res_dir "$base_path"/step3_labels/ \
        --lab_name "$lab_name" \
        --start_pred_date 2022-01-01 --end_pred_date 2022-12-31 \
        --min_age 30 --max_age 70 \
        --months_buffer 3 \
        --abnorm_type multi \
        --version v2

################################################################################################
############################# Extra data #######################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_labels "$base_path"/step3_labels/"$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_labels.parquet \
            --file_path_data /home/ivm/valid/data/processed_data/step1_clean/"$lab_name_two"_"$extra_two".parquet \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3"_"$lab_name_two"_"$extra_two" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; \
        python3 /home/ivm/valid/scripts/steps/step4_sumstats.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path "$base_path"/step3_labels/ \
            --file_name_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01 \
            --mean_impute 0

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; \
        python3 /home/ivm/valid/scripts/steps/step4_labs.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_lab /home/ivm/valid/data/extra_data/processed_data/step1_clean/R13_kanta_lab_min1pct_18-70-in-2026-293629total_2025-04-17.parquet \
            --dir_path_labels "$base_path"/step3_labels/ \
            --file_name_labels_start "$lab_name"_"$extra"_"$date_1"_filtered_"$date_2"_"$extra_labels"_"$date_3" \
            --lab_name "$lab_name" \
            --start_date 2021-10-01

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; \
        python3 /home/ivm/valid/scripts/steps/step4_atcsicds.py \
            --res_dir "$base_path"/step4_data/ \
            --file_path_preds /home/ivm/valid/data/extra_data/processed_data/step1_clean/icds_r13_2025-06-06_min1p0pct_2025-06-06.parquet \
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
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; \
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
            --min_pct "1p0"
################################################################################################
############################# XGBoost ##########################################################
################################################################################################
! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;date_5=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --n_boots 5 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

! base_path=/home/ivm/valid/data/processed_data; \
    fg_ver=R13; lab_name=tsh; lab_name_two=t4; extra=d1_multi; extra_two=d1_2025-07-21; extra_labels=testv2_2022_w3;\
    date_1=2025-07-18; date_2=2025-08-01; date_3=2025-08-01; date_4=2025-08-01;date_5=2025-08-01;\
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
            --metric mlogloss  \
            --reweight 0 \
            --low_lr 0.01 \
            --refit 1 \
            --n_trials 200

################################################################################################
############################# Evals ############################################################
from model_eval_utils import bootstrap_metric, bootstrap_difference, continuous_nri, bootstrap_nri
import sklearn.metrics as skm
import numpy as np
import os.path
from delong_utils import delong_roc_test


preds_descrs={"1_clinpheno": "clinical phenotype", 
              "2_lastval": "last value", 
              "2_sumstats": "sumstats", 
              "3_twosumstats": "two sumstats", 
              "3_otherlabs": "other labs", 
              "3_registry": "registry data", 
              "4_all": "all data", 
              "5_icd": "ICD data",
              "5_atc": "ATC data", 
              "6_atc2020": "ATC data >= 2020",
              "6_icd2020": "ICD data >= 2020"}
metrics = ["mlogloss", "q75", "mae", "q90"]

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

metric = "mlogloss"
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

metric = "mlogloss"
results = pl.DataFrame()

for combo_1, combo_2 in no_dups_combos:
    metric_1 = combo_1.split("_")[1]
    if metric_1 == "GRU": metric_1 = "mlogloss"
    file_path_1 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_1]+"/"+combo_1+"/models/"+lab_name+"/")
    metric_2 = combo_2.split("_")[1]
    if metric_2 == "GRU": metric_2 = "mlogloss"
    file_path_2 = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric_2]+"/"+combo_2+"/models/"+lab_name+"/")

    preds_1 = pl.read_parquet(file_path_1)
    preds_2 = pl.read_parquet(file_path_2)

    if metric_1 == "mlogloss" and metric_2 == "mlogloss":
        for set_no in set_names:
            descriptors = {"MODEL_1": combo_1, "MODEL_2": combo_2, "SET": set_names[set_no]}
            for crnt_filter_name in filters:
                descriptors["FILTER"] = crnt_filter_name
                for goal_name in goal_names:
                    preds_1 = preds_1.with_columns(pl.when(pl.col(goal_name)==1).then(1).otherwise(0).alias(goal_name))

                    preds = preds_1.select("FINNGENID", "SET", "EVENT_AGE", goal_name, "ABNORM_PROBS_1").join(preds_2.select("FINNGENID", "ABNORM_PROBS_1"), on="FINNGENID", how="left")
                    preds = preds.filter(~pl.col.ABNORM_PROBS_1.is_null(), ~pl.col.ABNORM_PROBS_1_right.is_null())
                    crnt_preds = preds.filter(pl.col.SET==set_no).filter(filters[crnt_filter_name])

                    ### P-values for AUCs with DeLong
                    pval_diff = 10**delong_roc_test(crnt_preds[goal_name].to_numpy(), 
                                                    crnt_preds["ABNORM_PROBS_1"].to_numpy(), 
                                                    crnt_preds["ABNORM_PROBS_1_right"].to_numpy())[0]
                    descriptors[goal_name+"_AUCDiff_Pvalue"]=pval_diff

                    ### P-values for Average Precision with Bootstrapping
                    diff_est, lowci, highci, pval_diff, avg_1, avg_2 = bootstrap_difference(metric_func = (skm.average_precision_score),
                                                                              preds_1=crnt_preds["ABNORM_PROBS_1"].to_numpy(), 
                                                                              preds_2=crnt_preds["ABNORM_PROBS_1_right"].to_numpy(),
                                                                              obs=crnt_preds[goal_name].to_numpy(),
                                                                              n_boots=100)
                    descriptors[goal_name+"_AvgPrecDiff_Pvalue"]=pval_diff

                    ### NRI with CI measure of if new model is better at reclassification <0 -> worst and >0 -> better
                    nri, lowci, highci = bootstrap_nri(continuous_nri, 
                                                       crnt_preds[goal_name].to_numpy(), 
                                                       crnt_preds["ABNORM_PROBS_1"].to_numpy(),
                                                       crnt_preds["ABNORM_PROBS_1_right"].to_numpy(),
                                                       n_boots=100)
                    descriptors[goal_name+"_NRI"]=nri
                    descriptors[goal_name+"_NRI_CI"]="("+str(round(lowci, 2))+ "-"+ str(round(highci, 2)) + ")"

                results = pl.concat([results, pl.DataFrame(descriptors)])
    display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_aucetc_pvals_"+get_date()+".csv")

results = pl.DataFrame()
train_goal = "y_MEAN_ABNORM"

for crnt_option in options:
    ### Getting info
    metric = crnt_option.split("_")[1]
    if metric == "GRU": metric = "mlogloss"
    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "logloss", "TLSTM", "mae", "q75", "q10", "q25"]])
        
    # Getting data
    file_path = get_dated_path("/home/ivm/valid/data/processed_data/step5_predict/"+file_descr+"/"+train_goals[metric]+"/"+crnt_option+"/models/"+lab_name+"/")
    if not file_path: continue
    date = file_path.split(".")[0].split("preds_")[1]
    preds = pl.read_parquet(file_path)

    mdl_name = crnt_option.split("_")[0]
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "mlogloss", "TLSTM", "mae", "q75", "q10", "q25"]])
    descriptors = {"Date": date, 
                   "Model": mdl_name, 
                   "File Description": file_descr, 
                   "Predictors": preds_descrs[crnt_pred_descr],
                   "Outcome": goal_names[train_goal]}
    
    crnt_pred_descr = "_".join([elem for elem in crnt_option.split("_") if elem not in ["xgb", "GRU", "mlogloss", "TLSTM", "mae", "q75", "q10", "q25"]])

    #### Have to remove multi levels for this
    if metric=="mlogloss": preds = preds.rename({"ABNORM_PROBS_1": "ABNORM_PROBS"})
    
    for set_no in set_names.keys():
        descriptors["SET"] = set_names[set_no]
        for goal in ["y_MEAN_ABNORM", "y_NEXT_ABNORM"]:
            preds = preds.with_columns(pl.when(pl.col(goal)==1).then(1).otherwise(0).alias(goal))
            
            N_total = preds.filter(pl.col.SET==set_no).height
            N_cases = preds.filter(pl.col.SET==set_no).filter(pl.col(goal)==1).height
            descriptors[goal_names[goal]+"_N"] = N_cases
            AUC = bootstrap_metric(skm.roc_auc_score, 
                                   preds.filter(pl.col.SET==set_no)[goal],
                                   preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[goal_names[goal]+"_AUC"] = AUC[0]
            descriptors[goal_names[goal]+"_AUC_CI"] = "("+str(round(AUC[1], 2))+ "-"+ str(round(AUC[2], 2)) + ")"
    
            averagePrec = bootstrap_metric(skm.average_precision_score, 
                                   preds.filter(pl.col.SET==set_no)[goal],
                                   preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"],
                                   n_boots=100)
            descriptors[goal_names[goal]+"_avgPrec"] = averagePrec[0]
            descriptors[goal_names[goal]+"_avgPrec_CI"] = "("+str(round(averagePrec[1], 2))+ "-"+ str(round(averagePrec[2], 2)) + ")"


            descriptors[goal+"_Brier"] = skm.brier_score_loss(preds.filter(pl.col.SET==set_no)[goal], 
                                                              preds.filter(pl.col.SET==set_no)["ABNORM_PROBS"])
    
        results = pl.concat([results, pl.DataFrame(descriptors)])
    display(results)
results.write_csv("/home/ivm/valid/results/model_evals/"+lab_name+"/"+lab_name+"_"+file_descr+"_aucsetc_"+get_date()+".csv")