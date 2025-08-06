# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, make_dir, init_logging, Timer, read_file
from model_eval_utils import get_train_type, create_report, eval_subset, get_optim_precision_recall_cutoff
from processing_utils import get_abnorm_func_based_on_name 
from plot_utils import get_plot_names
from model_fit_utils import get_cont_goal_col_name
from model_eval_utils import save_all_report_plots, get_all_eval_metrics
from optuna_utils import run_optuna_optim
# time
from datetime import datetime
# Standard stuff
import polars as pl
import numpy as np
import xgboost as xgb
import optuna
import pickle
import shap
# Logging and input
import argparse
import logging
logger = logging.getLogger(__name__)
# Training and eval
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def get_ext_data(start_date: datetime,
                 dtype="BMI"):
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
    if dtype=="BMI":
        return (ext_data
                .select("FINNGENID", "DATE", "BMI")
                .filter(~pl.col.BMI.is_null(), pl.col.DATE<datetime(2021,10,1))
                .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                .group_by("FINNGENID").agg(pl.col.BMI.mean().alias("BMI"))
               )
    if dtype=="ALCOHOL":
        return (ext_data
                    .select("FINNGENID", "DATE", "ALCOHOL")
                    .filter(~pl.col.ALCOHOL.is_null(), pl.col.DATE<datetime(2021,10,1))
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
        return ((ext_data
                    .select("FINNGENID", "DATE", "SMOKE")
                    .filter(~pl.col.SMOKE.is_null(), pl.col.DATE<datetime(2021,10,1))
                    .with_columns(pl.when(pl.col.SMOKE==9).then(pl.lit(None)).otherwise(pl.col.SMOKE).alias("SMOKE"))
                    .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                    .filter(~pl.col.SMOKE.is_null())
                    .filter((pl.col.SMOKE==pl.col.SMOKE.min()).over("FINNGENID"))
                     .unique()
                    .with_columns(pl.when(pl.col.SMOKE<=3).then(pl.lit(1)).otherwise(pl.lit(0)).alias("SMOKE"))
                ))
    if dtype=="SBP":
        return (ext_data
                .select("FINNGENID", "DATE", "SBP")
                .filter(~pl.col.SBP.is_null(), pl.col.DATE<datetime(2021,10,1))
                .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                .group_by("FINNGENID").agg(pl.col.SBP.mean().alias("SBP"))
               )
    if dtype=="DBP":
        return (ext_data
                .select("FINNGENID", "DATE", "DBP")
                .filter(~pl.col.DBP.is_null(), pl.col.DATE<datetime(2021,10,1))
                .filter((pl.col.DATE==pl.col.DATE.max()).over("FINNGENID"))
                .group_by("FINNGENID").agg(pl.col.DBP.mean().alias("DBP"))
               )

def get_edu_data(start_date: datetime):
    edu_data = pl.read_csv("/finngen/pipeline/finngen_R12/socio_register_1.0/data/finngen_R12_socio_register_1.0.txt.gz", separator="\t")
    edu_data = (edu_data.filter(pl.col.CATEGORY=="EDUC").with_columns(pl.col.CODE2.str.head(1).cast(pl.Int32).alias("EDU"))
                    .filter(pl.col.YEAR<=2021)
                    .filter((pl.col.EDU==pl.col.EDU.max()).over("FINNGENID"))
                    .select("FINNGENID", "EDU")
                    .unique()
                    .with_columns(pl.when(pl.col.EDU<=4).then(pl.lit(0)).otherwise(1).alias("EDU"))
    )
    return edu_data
    
        
def get_out_data(data: pl.DataFrame, 
                 model_final: xgb.XGBClassifier, 
                 X_all: pl.DataFrame, 
                 y_all: pl.DataFrame, 
                 metric: str,
                 lab_name: str,
                 goal: str,
                 abnorm_extra_choice: str="") -> pl.DataFrame:
    """Returns the relevant columns from the input data with the predictions of the model.
       Not that abnormality here is the case/control status.
       Columns:
            - `FINNGENID`: Individual ID
            - `EVENT_AGE`: Age at event (for controls last lab record and for cases )
            - `LAST_VAL_DATE`: Prediction date (TODO need to check what exact time this is)
            - `SET`: Train, validation or test set
            - `N_PRIOR_ABNORMS`: Number of prior abnormal values
            - `VALUE` (if continuous prediction): Predicted value
            - `ABNORM_PROBS` (if binary prediction): Probability of abnormality
            - `ABNORM_PREDS` (if binary prediction): Predicted abnormality
            - `TRUE_VALUE` (if continuous prediction): True value
            - `TRUE_ABNORM` (if binary prediction): True abnormality"""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Base data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # data = data.with_columns([
    #     pl.when(pl.col(col)>0).then(1).otherwise(0).alias(col) for col in data.columns if "ABNORM" in col
    # ])
    out_data = (data
                .select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM", "EVENT_AGE", "SET", "LAST_VAL_DATE", "ABNORM")
                .rename({"LAST_VAL_DATE": "DATE", "ABNORM": "N_PRIOR_ABNORMS"})
               )
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Continuous prediction                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ################# PROBABLY DEPRECATED #####################################
    if get_train_type(metric) == "cont":    
        y_pred = model_final.predict(X_all.to_numpy())
        out_data = out_data.with_columns([
                            pl.Series("ABNORM_PROBS", y_pred),
                            pl.Series("TRUE_VALUE", y_all),
        ])
        # Abnormality prediction based on the continuous value
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "ABNORM_PROBS").rename({"ABNORM_CUSTOM": "ABNORM_PREDS"})
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "TRUE_VALUE").rename({"ABNORM_CUSTOM": "TRUE_ABNORM"})
        print(out_data["TRUE_ABNORM"].value_counts())
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Binary prediction of "abnormality"                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    else:        
        # # # # # # # # # Continuous column # # # # # # # # # # # # # # # # # # # #
        new_goal = get_cont_goal_col_name(goal, data.columns)
        if new_goal is not None:
            y_cont_all = data.select(new_goal)

            out_data = out_data.with_columns([
                pl.Series("TRUE_VALUE", y_cont_all),
            ])
        out_data = out_data.with_columns([pl.col(goal).alias("TRUE_ABNORM")])

        # # # # # # # # # Abnormality columns # # # # # # # # # # # # # # # # # # #
        if get_train_type(metric) == "bin":  
            y_pred = model_final.predict_proba(X_all.to_numpy())[:,1]
            out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS")])
            # Binary abnormality prediction based on optimal cut-off
            optimal_proba_cutoff = get_optim_precision_recall_cutoff(out_data)
            logging.info(f"Optimal cut-off for prediction based on PR {optimal_proba_cutoff}")
            out_data = out_data.with_columns(
                (pl.col("ABNORM_PROBS") > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS")
            )
        elif get_train_type(metric) == "multi":  
            for pred_val in out_data["TRUE_ABNORM"].unique():
                y_pred = model_final.predict_proba(X_all.to_numpy())[:,pred_val]
                out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS_"+str(pred_val))])
                # Binary abnormality prediction based on optimal cut-off
                temp_data = out_data.with_columns(pl.when((pl.col.TRUE_ABNORM!=1)&(pl.col.TRUE_ABNORM!=0)).then(pl.lit(0)).otherwise(pl.col.TRUE_ABNORM).alias("TRUE_ABNORM"))
                temp_data = temp_data.rename({"ABNORM_PROBS_"+str(pred_val): "ABNORM_PROBS"})
                optimal_proba_cutoff = get_optim_precision_recall_cutoff(temp_data)
                logging.info(f"Optimal cut-off for {pred_val} prediction based on PR {optimal_proba_cutoff}")
                out_data = out_data.with_columns(
                (pl.col("ABNORM_PROBS_"+str(pred_val)) > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS_"+str(pred_val)))
    return(out_data)

def get_shap_importances(X_in: pl.DataFrame,
                         explainer: shap.TreeExplainer,
                         lab_name: str,
                         lab_name_two: str="") -> tuple[pl.DataFrame, list]:
    """Creates a dataframe based on the mean absolute SHAP values of the model. 
       Also returns the (readable) names of the features in the same order as the SHAP values."""
    
    new_names = get_plot_names(X_in.columns, lab_name, lab_name_two)
    explanations = explainer.shap_values(X_in)
    mean_shaps = np.abs(explanations).mean(0)
    shap_importance = pl.DataFrame({"mean_shap": mean_shaps}, schema=["mean_shap"]).with_columns(pl.Series("labels", new_names)).sort("mean_shap", descending=True)

    return(shap_importance, new_names)

def save_importances(top_gain,
                     out_down_path: str,
                     subset: str="all") -> None:
    """Saves the feature importances of the model to a csv file. """
    print()
    try:
        logging.info(top_gain.head(10))
        top_gain.write_csv(out_down_path + "shap_importance_" + subset + "_" + get_date() + ".csv")
    except:
        top_gain = pl.DataFrame(top_gain).with_columns(pl.col(top_gain.columns[0]).arr.to_struct().alias(top_gain.columns[0])).unnest(top_gain.columns[0])
        logging.info(top_gain.head(10))
        top_gain.write_csv(out_down_path + "shap_importance_" + subset + "_" + get_date() + ".csv")

from model_eval_utils import get_score_func_based_on_metric
def quantile_eval(metric):
    def eval_metric(x, y):
        loss = get_score_func_based_on_metric(metric)(x, y)
        print(loss)
        return np.mean(loss)
    return eval_metric

def xgb_final_fitting(best_params: dict, 
                      X_train: pl.DataFrame, 
                      y_train: pl.DataFrame, 
                      X_valid: pl.DataFrame, 
                      y_valid: pl.DataFrame, 
                      metric: str,
                      low_lr: float,
                      early_stop: int,
                      n_classes: int=2):
    """Fits the final XGB model with the best hyperparameters found in the optimization step."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    timer = Timer()
    params_fin = {}
    params_fin.update(get_xgb_base_params(metric, low_lr, n_classes))
    params_fin.update(best_params)
    print(params_fin)
    np.random.seed(9234)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fitting                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if get_train_type(metric) == "bin" or get_train_type(metric) == "multi":
        clf = xgb.XGBClassifier(**params_fin, early_stopping_rounds=early_stop, n_estimators=10000)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
    else:
        if metric in ["q10", "q05", "q25", "q50", "q75", "q90", "q95"]: del params_fin["eval_metric"]
        clf = xgb.XGBRegressor(**params_fin, early_stopping_rounds=early_stop, n_estimators=10000)
        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Final fitting ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {params_fin["learning_rate"]}')
    logging.info(f'best boosting round: {clf.best_iteration}')
    logging.info("time taken {timer.get_elapsed()}")

    return(clf)    

def get_xgb_base_params(metric: str, 
                        lr: float,
                        n_classes: int=2) -> dict:
    """Returns the base parameters for the XGBoost model."""

    base_params = {'tree_method': 'approx', 'learning_rate': lr, 'seed': 1239}
    if metric == "q50":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.5, "eval_metric": "q50"})
    elif metric == "q75":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.75, "eval_metric": metric})
    elif metric == "q90":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.90, "eval_metric": metric})
    elif metric == "q95":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.95, "eval_metric": metric})
    elif metric == "q25":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.25, "eval_metric": "q25"})
    elif metric == "q05":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.05, "eval_metric": "q05"})
    elif metric == "q10":
        base_params.update({"objective": "reg:quantileerror", "quantile_alpha": 0.1, "eval_metric": "q10"})
    elif metric == "tweedie":
        base_params.update({"objective": "reg:tweedie", "eval_metric": "tweedie-nloglik@1.99"})
    elif get_train_type(metric) == "cont":
        base_params.update({"objective": "reg:squarederror", "eval_metric": metric})
    elif get_train_type(metric) == "bin":
        base_params.update({"objective": "binary:logistic", "eval_metric": metric})
    elif get_train_type(metric) == "multi":
        base_params.update({"objective": "multi:softprob", "num_class": n_classes, "eval_metric": metric})
    return(base_params)

def run_speed_test(dtrain: xgb.DMatrix, 
                   dvalid: xgb.DMatrix, 
                   metric: str,
                   lr: float,
                   n_classes: int) -> None:
    """Runs a quick test to see how fast the model can learn with a high learning rate."""
    base_params = get_xgb_base_params(metric, lr, n_classes)
    timer = Timer()
    model = xgb.train(params=base_params, dtrain=dtrain,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      num_boost_round=10,
                      early_stopping_rounds=5,
                      verbose_eval=1)
    print("Fit took: " + timer.get_elapsed())
    print(model.best_iteration)
    
def get_weighting(reweight, y_train):
    """Returns the weighting for the positive class."""
    if reweight: return(y_train.shape[0] - np.sum(y_train)) / np.sum(y_train)
    else: return(1)

def get_weights(pos_weight, y):
    """Returns the weights for the positive class."""
    return np.where(y == 1, pos_weight, 1)

def create_xgb_dts(data: pl.DataFrame, 
                   X_cols: list, 
                   y_goal: str, 
                   reweight: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, xgb.DMatrix, xgb.DMatrix, RobustScaler]:
    """Creates the XGBoost data matrices and scales the data."""
    # Need later to be Float for scaling
    data = data.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col, dtype in data.schema.items() if isinstance(dtype, pl.Int64) or isinstance(dtype, pl.Int32)
    ])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Replace -1s with 2s                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if -1 in data.get_column(y_goal):
        data = data.with_columns(pl.when(pl.col(y_goal)==-1).then(pl.lit(2)).otherwise(pl.col(y_goal)).alias(y_goal))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Split data                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    train_data = data.filter(pl.col("SET")==0).drop("SET")
    valid_data = data.filter(pl.col("SET")==1).drop("SET")
    test_data = data.filter(pl.col("SET")==2).drop("SET")

    # XGB datatype prep
    X_train = train_data.select(X_cols); y_train = train_data.select(y_goal)
    X_valid = valid_data.select(X_cols); y_valid = valid_data.select(y_goal)
    X_test = test_data.select(X_cols); y_test = test_data.select(y_goal)
    X_all = data.select(X_cols); y_all = data.select(y_goal)
    #print(f"have {round(y_train.sum()/len(y_train)*100, 2)}% (N= {y_train.sum()} )in train and {round(y_valid.sum()/len(y_valid)*100, 2)}% (N={y_valid.sum()}) in validation ")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Scaling                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    scaler_base = RobustScaler()
    scaler_base.set_output(transform="polars")
    X_train=pl.DataFrame(scaler_base.fit_transform(X_train), schema=X_train.schema)
    X_valid=pl.DataFrame(scaler_base.transform(X_valid), schema=X_valid.schema)
    X_test=pl.DataFrame(scaler_base.transform(X_test), schema=X_test.schema)
    X_all_scaled=pl.DataFrame(scaler_base.transform(X_all), schema=X_all.schema)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 XGBoost datatype                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    pos_weight = get_weighting(reweight, y_train)
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True, weight=get_weights(pos_weight, y_train))
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True, weight=get_weights(pos_weight, y_valid))
    
    return(data["FINNGENID"], X_train, y_train, X_valid, y_valid, X_test, y_test, X_all_scaled, y_all, X_all, dtrain, dvalid, scaler_base)

def get_relevant_label_data_cols(data: pl.DataFrame, 
                                 goal: str) -> pl.DataFrame:
    """Returns the relevant columns for the prediction task. 
       If prediction column is ABNORM, keeping column on which ABNORMALITY is based on.
       If prediciton column is y_DIAG, keeping column for mean value of the lab value (y_MEAN)."""
    new_goal = get_cont_goal_col_name(goal, data.columns)
    if "START_DATE" in data.columns:
        data = data.with_columns(pl.col.START_DATE.dt.year().alias("YEAR"))
        if new_goal is not None:
            data = data.select(["FINNGENID", "SET", "YEAR", "START_DATE", "EVENT_AGE", "SEX", goal, new_goal])
        else:
            data = data.select(["FINNGENID", "SET", "YEAR", "START_DATE", "EVENT_AGE", "SEX", goal])
    else:
        if new_goal is not None:
            data = data.select(["FINNGENID", "SET", "EVENT_AGE", "SEX", goal, new_goal])
        else:
            data = data.select(["FINNGENID", "SET",  "EVENT_AGE", "SEX", goal])
    return data

def get_data_and_pred_list(file_path_labels: str, 
                           file_path_icds: str,
                           file_path_atcs: str,
                           file_path_sumstats: str,
                           file_path_second_sumstats: str,
                           file_path_labs: str,
                           goal: str,
                           preds: list,
                           start_date: str) -> tuple[pl.DataFrame, list]:
    """Reads in label data and merges it with other data modalities. Returns the data and the predictors."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting Data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(file_path_labels)
    #data = get_relevant_label_data_cols(data, goal)

    # Adding other data modalities
    if file_path_icds != "": 
        icds = read_file(file_path_icds)
        data = data.join(icds, on="FINNGENID", how="left")
    if file_path_atcs != "": 
        atcs = read_file(file_path_atcs)
        data = data.join(atcs, on="FINNGENID", how="left")
    if file_path_sumstats != "": 
        sumstats = read_file(file_path_sumstats,
                             schema={"SEQ_LEN": pl.Float64,
                                     "MIN_LOC": pl.Float64,
                                     "MAX_LOC": pl.Float64,
                                     "FIRST_LAST": pl.Float64})
        if "SET" in sumstats.columns: sumstats = sumstats.drop("SET") # dropping duplicate info if present
        data = data.join(sumstats, on="FINNGENID", how="left")
        if start_date != "":
            data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.LAST_VAL_DATE).dt.total_days().alias("LAST_VAL_DIFF"))
    if file_path_second_sumstats != "": 
        second_sumstats = read_file(file_path_second_sumstats)
        if "SET" in second_sumstats.columns: second_sumstats = second_sumstats.drop("SET")
        second_sumstats = second_sumstats.rename({col: f"S_{col}" for col in second_sumstats.columns if col != "FINNGENID"})
        data = data.join(second_sumstats, on="FINNGENID", how="left")
        if start_date != "":
            data = data.with_columns((datetime.strptime(start_date, "%Y-%m-%d")-pl.col.S_LAST_VAL_DATE).dt.total_days().alias("S_LAST_VAL_DIFF"))
    if file_path_labs != "": 
        labs =read_file(file_path_labs)
        data = data.join(labs, on="FINNGENID", how="left")
    if "BMI" in preds:
        extra_data = get_ext_data(start_date, "BMI")
        data = data.join(extra_data.select("FINNGENID", "BMI"), how="left", on="FINNGENID")
    if "SMOKE" in preds:
        extra_data = get_ext_data(start_date, "SMOKE")
        data = data.join(extra_data.select("FINNGENID", "SMOKE"), how="left", on="FINNGENID")  
    if "SBP" in preds:
        extra_data = get_ext_data(start_date, "SBP")
        data = data.join(extra_data.select("FINNGENID", "SBP"), how="left", on="FINNGENID")  
    if "DBP" in preds:
        extra_data = get_ext_data(start_date, "DBP")
        data = data.join(extra_data.select("FINNGENID", "DBP"), how="left", on="FINNGENID")  
    if "ALCOHOL" in preds:
        extra_data = get_ext_data(start_date, "ALCOHOL")
        data = data.join(extra_data.select("FINNGENID", "ALCOHOL"), how="left", on="FINNGENID")  
    if "EDU" in preds:
        extra_data = get_edu_data(start_date)
        data = data.join(extra_data.select("FINNGENID", "EDU"), how="left", on="FINNGENID")
    # Changing data-modality of sex column
    data = data.with_columns(pl.col("SEX").replace({"female": 0, "male": 1}))

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

    return(data, X_cols)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--model_fit_date", type=str, help="Original date of model fitting.", default="")
    parser.add_argument("--fg_ver", type=str, help="FinnGen version needed for import of BMI or SMOKE.", default="r13")

    # Data paths
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument("--file_path_icds", type=str, help="Path to ICD data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_atcs", type=str, help="Path to ATC data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_labs", type=str, help="Path to Lab data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_sumstats", type=str, help="Path to summary statistics of a single lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_second_sumstats", type=str, help="Path to summary statistics of a another lab value data. Each column is a predictor. [default: '' = not loaded]", default="")

    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--lab_name_two", type=str, help="Readable name of the second most relevant measurement value for file naming.", default="")
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--start_date", type=str, default="", help="Date to filter before")

    # Prediction task
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--abnorm_extra_choice", default="", type=str, help="Extra choice for abnormality prediction. [default: ''] ")
    parser.add_argument("--preds", type=str, help="List of predictors. Special options: ICD_MAT, ATC_MAT, SUMSTATS, SECOND_SUMSTATS, LAB_MAT. Taking all columns from the respective data file.", 
                        default=["SUMSTATS", "EVENT_AGE", "SEX"], nargs="+")
    
    # Model fitting parameters
    parser.add_argument("--lr", type=float, help="Learning rate for hyperparamter optimization, can be high so this goes fast.", default=0.4)
    parser.add_argument("--low_lr", type=float, help="Learning rate for final model training.", default=0.001)
    parser.add_argument("--reweight", type=int, default=0)
    parser.add_argument("--early_stop", type=int, help="Early stopping for the final fitting round. Currently, early stopping fixed at 5 for hyperparameter optimization.", default=5)
    parser.add_argument("--metric", type=str, help="Which metric to optimize based on.", default="mse")

    # Hyperparameter optimization parameters
    parser.add_argument("--run_step0", type=int, help="Whether to run quick trial run [deafult: 0]", default=0)
    parser.add_argument("--n_trials", type=int, help="Number of hyperparameter optimizations to run [default: 1 = running based on time_step1 instead]", default=1)
    parser.add_argument("--time_optim", type=int, help="Number of seconds to run hyperparameter optimizations for, instead of basing it on the number of traisl. [run when n_trials=1]", default=300)
    parser.add_argument("--refit", type=int, help="Whether to rerun the hyperparameter optimization", default=1)

    # Final model fitting and evaluation
    parser.add_argument("--skip_model_fit", type=int, help="Whether to rerun the final model fitting, or load a prior model fit.", default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics.", default=500)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Initial setup                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #     
    timer = Timer()
    args = get_parser_arguments()

    # File names
    study_name = "xgb_" + str(args.metric) + "_" + args.pred_descriptor
    if args.model_fit_date == "": args.model_fit_date = get_date()

    out_dir = args.res_dir + study_name + "/"; 
    out_model_dir = out_dir + "models/" + args.lab_name + "/" + args.model_fit_date + "/" 
    out_plot_dir = out_dir + "plots/" + args.model_fit_date + "/"
    out_plot_path = out_plot_dir + args.lab_name + "_"
    out_down_dir = out_dir + "down/" + args.model_fit_date + "/"
    out_down_path = out_down_dir + args.lab_name + "_" + study_name + "_"
    init_logging(out_dir, args.lab_name, logger, args)
    make_dir(out_model_dir); make_dir(out_plot_dir); make_dir(out_down_dir)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data, X_cols = get_data_and_pred_list(file_path_labels=args.file_path_labels, 
                                          file_path_icds=args.file_path_icds, 
                                          file_path_atcs=args.file_path_atcs, 
                                          file_path_sumstats=args.file_path_sumstats, 
                                          file_path_second_sumstats=args.file_path_second_sumstats, 
                                          file_path_labs=args.file_path_labs, 
                                          goal=args.goal, 
                                          preds=args.preds,
                                          start_date=args.start_date)
    fgids, X_train, y_train, X_valid, y_valid, \
        X_test, y_test, X_all, y_all, X_all_unscaled, \
        dtrain, dvalid, scaler_base = create_xgb_dts(data=data, 
                                                     X_cols=X_cols, 
                                                     y_goal=args.goal, 
                                                     reweight=args.reweight)
    print(X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).head(2))
    X_all_unscaled.with_columns(pl.Series("FINNGENID", fgids)).write_parquet(out_model_dir + "Xall_unscaled_" + get_date() + ".parquet")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Hyperparam optimization with optuna                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.run_step0 == 1: run_speed_test(dtrain=dtrain, 
                                           dvalid=dvalid, 
                                           metric=args.metric, 
                                           lr=args.lr) # to test learning rate
    else:
        if not args.skip_model_fit:
            base_params = get_xgb_base_params(metric=args.metric, 
                                              lr=args.lr, 
                                              n_classes=len(y_train.unique()))
            best_params = run_optuna_optim(train=dtrain, 
                                           valid=dvalid, 
                                           test=None,
                                           lab_name=args.lab_name, 
                                           refit=args.refit, 
                                           time_optim=args.time_optim, 
                                           n_trials=args.n_trials, 
                                           study_name=study_name,
                                           res_dir=args.res_dir,
                                           model_type="xgb",
                                           model_fit_date=args.model_fit_date,
                                           base_params=base_params)
            logging.info(timer.get_elapsed())
            model_final = xgb_final_fitting(best_params=best_params, 
                                            X_train=X_train, y_train=y_train, 
                                            X_valid=X_valid, y_valid=y_valid, 
                                            metric=args.metric,
                                            low_lr=args.low_lr,
                                            early_stop=args.early_stop,
                                            n_classes=len(y_train.unique()))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving or loading                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

            pickle.dump(model_final, open(out_model_dir + "model_" + get_date() + ".pkl", "wb"))
            pickle.dump(scaler_base, open(out_model_dir + "scaler_" + get_date() + ".pkl", "wb"))
        else:
            model_final = pickle.load(open(out_model_dir + "model_" + get_date() + ".pkl", "rb"))

        if not args.skip_model_fit:
            # SHAP explainer for model
            shap_explainer = shap.TreeExplainer(model_final)
            train_importances, _ = get_shap_importances(X_train, shap_explainer, args.lab_name, args.lab_name_two)
            print(train_importances)
            test_importances, _ = get_shap_importances(X_test, shap_explainer, args.lab_name, args.lab_name_two)
            save_importances(top_gain=train_importances,
                             out_down_path=out_down_path,
                             subset="train")
            save_importances(top_gain=test_importances,
                             out_down_path=out_down_path,
                             subset="test")
            # Model predictions
            out_data = get_out_data(data=data, 
                                    model_final=model_final, 
                                    X_all=X_all, 
                                    y_all=y_all, 
                                    metric=args.metric,
                                    lab_name=args.lab_name,
                                    goal=args.goal,
                                    abnorm_extra_choice=args.abnorm_extra_choice)
            out_data.write_parquet(out_model_dir + "preds_" + get_date() + ".parquet")  
        else:
            train_importances = pl.read_csv(out_down_path + "shap_importance_train_" + args.model_fit_date + ".csv")
            test_importances = pl.read_csv(out_down_path + "shap_importance_test_" + args.model_fit_date + ".csv")
            out_data = pl.read_parquet(out_model_dir + "preds_" + get_date() + ".parquet")


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # #                 Evaluations                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
    #     crnt_report = create_report(model_final, 
    #                                 out_data, 
    #                                 display_scores=[args.metric, "aucpr"], 
    #                                 metric=args.metric)
    #     pickle.dump(crnt_report, open(out_model_dir + "report_" + get_date() + ".pkl", "wb"))  


        save_all_report_plots(out_data=out_data,
                              train_importances=train_importances,
                              valid_importances=train_importances,
                              test_importances=test_importances,
                              out_plot_path=out_plot_path,
                              out_down_path=out_down_path,
                              train_type=get_train_type(args.metric))


    #     eval_metrics = get_all_eval_metrics(data=out_data,
    #                                         plot_path=out_plot_path, 
    #                                         down_path=out_down_path, 
    #                                         n_boots=args.n_boots,
    #                                         train_type=get_train_type(args.metric))
                                                
    #     if args.save_csv == 1:
    #         eval_metrics.filter(pl.col("F1").is_not_null()).write_csv(out_down_path + "evals_" + get_date() + ".csv", separator=",")
