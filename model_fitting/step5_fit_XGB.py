# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import get_date, get_datetime, make_dir, init_logging, Timer, read_file
from model_eval_utils import get_train_type, create_report, eval_subset
from model_fit_utils import create_optuna_study, get_optim_precision_recall_cutoff
from processing_utils import get_abnorm_func_based_on_name 
from plot_utils import create_report_plots, get_plot_names
# Standard stuff
import time 
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

def get_cont_goal_col_name(goal: str,
                           col_names: list) -> str:
    """Returns the column name of the continuous goal.
       If first part of goal name in column names, i.e. y_MEDIAN in the case of y_MEDIAN_ABNORM, returns the first part.
       Otherwise returns y_MEAN. If not present, certain plotting will be skipped"""
    # Binary prediction tasks
    if goal_split[-1] == "ABNORM" or args.goal == "y_DIAG":
        goal_split = goal.split("_")
        new_goal = "_".join(goal_split[0:len(goal_split)-1])
        if not new_goal in col_names: new_goal = "y_MEAN"

        # If y_MEAN still not in column names, return None
        if not new_goal in col_names: 
            print("y_MEAN not in column names. Certain plots cannot be created.")
            return None
        else:
            return(new_goal)
    # Continuous prediction tasks
    else:
        print("Warning: Continuous prediction probably deprecated at the moment.")
        return None

def get_out_data(data: pl.DataFrame, 
                 model_final: xgb.XGBClassifier, 
                 X_all: pl.DataFrame, 
                 y_all: pl.DataFrame, 
                 metric: str,
                 lab_name: str,
                 goal: str):
    """Returns the relevant columns from the input data with the predictions of the model.
       Not that abnormality here is the case/control status.
       Columns:
            - `FINNGENID`: Individual ID
            - `EVENT_AGE`: Age at event (TODO need to check what exact time this is)
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
    out_data = (data.select(["FINNGENID", "EVENT_AGE", "LAST_VAL_DATE", "SET", "ABNORM"])
                    .rename({"DATE": "LAST_VAL_DATE", "ABNORM": "N_PRIOR_ABNORMS"})
                    .unique())
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Continuous prediction                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    ################# PROBABLY DEPRECATED #####################################
    if get_train_type(metric) == "cont":    
        y_pred = model_final.predict(X_all.to_numpy())
        out_data = out_data.assign(VALUE = y_pred)
        out_data = get_abnorm_func_based_on_name(lab_name)(out_data, "VALUE").rename(columns={"ABNORM_CUSTOM": "ABNORM_PREDS"})
        out_data = out_data.assign(TRUE_VALUE = y_all)
        out_data = get_abnorm_func_based_on_name(lab_name)(out_data, "TRUE_VALUE").rename(columns={"ABNORM_CUSTOM": "TRUE_ABNORM"})

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Binary prediction of "abnormality"                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    else:        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # Continuous column # # # # # # # # # # # # # # # # # # # #
        new_goal = get_cont_goal_col_name(goal, data.columns)
        if new_goal is not None:
            y_cont_all = data.select(new_goal)

            out_data = out_data.with_columns([
                pl.Series("TRUE_VALUE", y_cont_all),
            ])

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # # # # # # # # # Abnormality columns # # # # # # # # # # # # # # # # # # #
        y_pred = model_final.predict_proba(X_all.to_numpy())[:,1]
        out_data = out_data.with_columns([
                        pl.Series("TRUE_ABNORM", y_all),
                        pl.Series("ABNORM_PROBS", y_pred),
                    ])
        # Binary abnormality prediction based on optimal cut-off
        optimal_proba_cutoff = get_optim_precision_recall_cutoff(out_data)
        logging.info(f"Optimal cut-off for prediction based on PR {optimal_proba_cutoff}")
        out_data = out_data.with_columns(
            (pl.col("ABNORM_PROBS") > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS")
        )
    return(out_data)

def get_shap_importances(X_in: pl.DataFrame,
                         explainer: shap.TreeExplainer,
                         lab_name: str):
    new_names = get_plot_names(X_in.columns, lab_name)
    explanations = explainer.shap_values(X_in.to_numpy())
    mean_shaps = np.abs(explanations.values).mean(0)
    shap_importance = pl.DataFrame({"mean_shap": mean_shaps}, schema=["mean_shap"]).with_columns(pl.Series("labels", new_names)).sort("mean_shap", descending=True)

    return(shap_importance, new_names)

def save_importances(X_all: pl.DataFrame, 
                     model_final: xgb.XGBClassifier, 
                     out_file_name: str,
                     lab_name: str,
                     date_model_fit: str,
                     pred_descriptor: str) -> None:
    """Saves the feature importances of the model to a csv file. """
    top_gain, _ = get_shap_importances(X_all, model_final.get_booster().get_dump()[0], lab_name)
    logging.info(top_gain.head(10))
    top_gain.write_csv(out_file_name + "down/" + date_model_fit + "/" + lab_name + "xgb_" + pred_descriptor + "_shap_importance_" + get_date() + ".csv")
      
def xgb_final_fitting(best_params: dict, 
                      X_train: pl.DataFrame, 
                      y_train: pl.DataFrame, 
                      X_valid: pl.DataFrame, 
                      y_valid: pl.DataFrame, 
                      metric: str,
                      low_lr: float,
                      early_stop: int) -> xgb.XGBClassifier:
    """Fits the final XGB model with the best hyperparameters found in the optimization step."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    timer = Timer()
    params_fin = {}
    params_fin.update(get_xgb_base_params(metric, low_lr))
    params_fin.update(best_params)
    print(params_fin)
    np.random.seed(9234)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fitting                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    clf = xgb.XGBClassifier(**params_fin, early_stopping_rounds=early_stop, n_estimators=1000)
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

def optuna_objective(trial: optuna.Trial, 
                     base_params: dict, 
                     dtrain: xgb.DMatrix, 
                     dvalid: xgb.DMatrix, 
                     eval_metric="tweedie") -> float:
    """Objective function for the Optuna optimization. Returns the last value of the metric on the validation set."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Suggested hyperparameters                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ["approx", "hist"]),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int("min_child_weight", 5, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 15, log=True),
        'gamma': trial.suggest_float('gamma', 1, 15, log=True),
    }

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Setting up trial                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{eval_metric}')
    evals_result = dict()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Train model                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    model = xgb.train(params=params, 
                      dtrain=dtrain, 
                      num_boost_round=200, 
                      evals=[(dtrain, "train"), (dvalid, "valid")], 
                      evals_result=evals_result,
                      early_stopping_rounds=5, 
                      verbose_eval=0,
                      callbacks=[pruning_callback])
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    
    # Return the last value of the metric on the validation set
    return evals_result["valid"][eval_metric][-1] 


def get_xgb_base_params(metric, lr):
    """Returns the base parameters for the XGBoost model."""

    base_params = {'tree_method': 'approx', 'learning_rate': lr, 'seed': 139}

    if metric == "tweedie":
        base_params.update({"objective": "reg:tweedie", "eval_metric": "tweedie-nloglik@1.99"})
    elif get_train_type(metric) == "cont":
        base_params.update({"objective": "reg:squarederror", "eval_metric": metric})
    elif get_train_type(metric) == "bin":
        base_params.update({"objective": "binary:logistic", "eval_metric": metric})

    return(base_params)

def run_optuna_optim(dtrain: xgb.DMatrix, 
                     dvalid: xgb.DMatrix, 
                     metric: str,
                     lr: float,
                     lab_name: str,
                     refit: bool,
                     time_optim: int,
                     n_trials: int,
                     study_name: str) -> dict:   
    """Runs the first step of the XGBoost optimization, which is to find the best hyperparameters for the model on a high learning rate.
       Uses Optuna to optimize the hyperparameters. The function returns the best hyperparameters found.
       Logs the best hyperparameters found and the best boosting round."""     
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    base_params = get_xgb_base_params(metric, lr)
    sampler = optuna.samplers.TPESampler(seed=429)
    study = create_optuna_study(study_name, lab_name, sampler, refit)
    tic = time.time()
    timer = Timer()
    np.random.seed(9234)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Running trials                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if n_trials == 1:
        while time.time() - tic < time_optim:
            study.optimize(lambda trial: optuna_objective(trial, base_params, dtrain, dvalid, metric) , n_trials=1)
    else:
        study.optimize(lambda trial: optuna_objective(trial, base_params, dtrain, dvalid, metric) , n_trials=n_trials)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info('Optuna optimization ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info(f'best score = {study.best_trial.value}')
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {lr}')
    logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
    logging.info('best tree params --------------------------')
    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
        
    return(study.best_trial.params)

def run_speed_test(dtrain: xgb.DMatrix, 
                   dvalid: xgb.DMatrix, 
                   metric: str,
                   lr: float) -> None:
    """Runs a quick test to see how fast the model can learn with a high learning rate."""
    base_params = get_xgb_base_params(metric, lr)
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
    for col in data: 
        if data.schema[col] == pl.Int64: data.with_columns(pl.col(col).cast(pl.Float64))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Split data                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    train_data = data.filter(pl.col("SET")==0).drop("SET")
    valid_data =  data.filter(pl.col("SET")==1).drop("SET")
    test_data =  data.filter(pl.col("SET")==2).drop("SET")

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
    
    X_train= pl.DataFrame(scaler_base.fit_transform(X_train.to_numpy()), schema=X_train.schema)
    X_valid= pl.DataFrame(scaler_base.transform(X_valid.to_numpy()), schema=X_valid.schema)
    X_test= pl.DataFrame(scaler_base.transform(X_test.to_numpy()), schema=X_test.schema)
    X_all_scaled = pl.DataFrame(scaler_base.transform(X_all.to_numpy()), schema=X_all.schema)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 XGBoost datatype                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    pos_weight = get_weighting(reweight, y_train)
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True, weight=get_weights(pos_weight, y_train))
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True, weight=get_weights(pos_weight, y_valid))
    #dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True, weight=get_weights(pos_weight, y_test))
    #dall = xgb.DMatrix(data=X_all, label=y_all, enable_categorical=True, weight=get_weights(pos_weight, y_all))
    
    return(X_train, y_train, X_valid, y_valid, X_test, y_test, X_all_scaled, y_all, X_all, dtrain, dvalid, scaler_base)
    

def get_relevant_label_data_cols(data: pl.DataFrame, 
                                 goal: str) -> pl.DataFrame:
    """Returns the relevant columns for the prediction task. 
       If the prediction goal is binary, i.e. name ends with ABNORM, or is y_DIAG, keeping also a continuous value column.
       If prediction column is ABNORM, keeping column on which ABNORMALITY is based on.
       If prediciton column is y_DIAG, keeping column for mean value of the lab value (y_MEAN)."""
    new_goal = get_cont_goal_col_name(goal, data.columns)
    if new_goal is not None:
        data = data.select(["FINNGENID", "SET", "START_DATE", "EVENT_AGE", "SEX", goal, new_goal])
    else:
        data = data.select(["FINNGENID", "SET", "START_DATE", "EVENT_AGE", "SEX", goal])
    return data

def get_data_and_pred_list(file_path_labels: str, 
                           file_path_icds: str,
                           file_path_atcs: str,
                           file_path_sumstats: str,
                           file_path_second_sumstats: str,
                           file_path_labs: str,
                           goal: str,
                           preds: list) -> tuple[pl.DataFrame, list]:
    """Reads in label data and merges it with other data modalities. Returns the data and the predictors."""
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Getting Data                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    data = read_file(file_path_labels)
    data = get_relevant_label_data_cols(data, goal)
    # Adding other data modalities
    if file_path_icds != "": 
        icds = read_file(file_path_icds)
        data = data.join(icds, on="FINNGENID", how="left")
    if file_path_atcs != "": 
        atcs = read_file(file_path_atcs)
        data = data.join(atcs, on="FINNGENID", how="left")
    if file_path_sumstats != "": 
        sumstats = read_file(file_path_sumstats)
        if "SET" in sumstats.columns: sumstats = sumstats.drop("SET") # dropping duplicate info if present
        data = data.join(sumstats, on="FINNGENID", how="left")
    if file_path_second_sumstats != "": 
        second_sumstats = read_file(file_path_second_sumstats)
        if "SET" in second_sumstats.columns: second_sumstats = second_sumstats.drop("SET")
        second_sumstats = second_sumstats.rename({col: f"S_{col}" for col in second_sumstats.columns if col != "FINNGENID"})
        data = data.join(second_sumstats, on="FINNGENID", how="left")
    if file_path_labs != "": 
        labs =read_file(file_path_labs)
        data = data.join(labs, on="FINNGENID", how="left")
    # Changing data-modality of sex column
    data = data.with_columns(pl.col("SEX").replace({"female": 0, "male": 1}))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Predictors                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    X_cols = []
    for pred in preds:
        if pred == "ICD_MAT":
            [X_cols.append(ICD_CODE) for ICD_CODE in icds.columns[np.logical_and(icds.columns != "FINNGENID", icds.columns != "LAST_ICD_DATE")]]
        elif pred == "ATC_MAT":
            [X_cols.append(ATC_CODE) for ATC_CODE in atcs.columns[np.logical_and(atcs.columns != "FINNGENID", atcs.columns != "LAST_ATC_DATE")]]
        elif pred == "SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in sumstats.columns[np.logical_and(sumstats.columns != "FINNGENID", sumstats.columns != "LAST_VAL_DATE")]]
        elif pred == "SECOND_SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in second_sumstats.columns[np.logical_and(second_sumstats.columns != "FINNGENID", second_sumstats.columns != "S_LAST_VAL_DATE")]]
        elif pred == "LAB_MAT":
            [X_cols.append(LAB_MAT) for LAB_MAT in labs.columns[labs.columns != "FINNGENID"]]
        else:
            X_cols.append(pred)

    return(data, X_cols)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    # Saving info
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", required=True)
    parser.add_argument("--date_model_fit", type=str, help="Original date of model fitting.", required=True)

    # Data paths
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument("--file_path_icds", type=str, help="Path to ICD data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_atcs", type=str, help="Path to ATC data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_labs", type=str, help="Path to Lab data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_sumstats", type=str, help="Path to summary statistics of a single lab value data. Each column is a predictor. [default: '' = not loaded]", default="")
    parser.add_argument("--file_path_second_sumstats", type=str, help="Path to summary statistics of a another lab value data. Each column is a predictor. [default: '' = not loaded]", default="")

    # Extra info
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value for file naming.", required=True)
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)

    # Prediction task
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--preds", type=str, help="List of predictors. Special options: ICD_MAT, ATC_MAT, SUMSTATS, SECOND_SUMSTATS, LAB_MAT. Taking all columns from the respective data file.", 
                        default=["SUMSTATS", "EVENT_AGE", "SEX"], nargs="+")
    
    # Model fitting parameters
    parser.add_argument("--lr", type=float, help="Learning rate for hyperparamter optimization, can be high so this goes fast.", default=0.4)
    parser.add_argument("--low_lr", type=float, help="Learning rate for final model training.", default=0.001)
    parser.add_argument("--reweight", type=int, default=1)
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
    study_name = "xgb_" + str(args.metric) + "_" + args.pred_descriptor +  "_reweight" + str(args.reweight) # for optuna
    out_dir = args.res_dir + study_name + "/"; out_file_name = out_dir + args.lab_name; out_plot_name = out_dir + "plots/" + args.lab_name
    log_file_name = args.lab_name + "_" + args.pred_descriptor + "_preds_" + get_datetime()
    if args.date_model_fit == "": args.date_model_fit = get_date()
    init_logging(out_dir, log_file_name, logger, args)
    make_dir(out_dir + "plots/"); make_dir(out_dir + "down/"); make_dir(out_dir + "down/" + args.date_model_fit + "/")
    
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
                                          preds=args.preds)
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, X_all_unscaled, dtrain, dvalid, scaler_base = create_xgb_dts(data=data, X_cols=X_cols, goal=args.goal, reweight=args.reweight)
    X_all_unscaled.write_csv(out_file_name + "_Xall_unscaled_" + get_date() + ".csv")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Hyperparam optimization with optuna                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if args.run_step0 == 1: run_speed_test(dtrain=dtrain, dvalid=dvalid, metric=args.metric, lr=args.lr) # to test learning rate
    else:
        if not args.skip_model_fit:
            best_params = run_optuna_optim(dtrain=dtrain, dvalid=dvalid, 
                                           metric=args.metric, 
                                           lr=args.lr, 
                                           lab_name=args.lab_name, 
                                           refit=args.refit, 
                                           time_optim=args.time_optim, 
                                           n_trials=args.n_trials, 
                                           study_name=study_name)
            logging.info(timer.get_elapsed())
            model_final = xgb_final_fitting(best_params=best_params, 
                                            X_train=X_train, y_train=y_train, 
                                            X_valid=X_valid, y_valid=y_valid, 
                                            metric=args.metric,
                                            low_lr=args.low_lr,
                                            early_stop=args.early_stop)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Saving or loading                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            save_importances(labels=X_train.columns, 
                             model_final=model_final, 
                             out_file_name=out_file_name,
                             lab_name=args.lab_name,
                             date_model_fit=args.date_model_fit,
                            pred_descriptor=args.pred_descriptor)
            pickle.dump(model_final, open(out_file_name + "_model_" + get_date() + ".pkl", "wb"))
            pickle.dump(scaler_base, open(out_file_name + "_scaler_" + get_date() + ".pkl", "wb"))
        else:
            model_final = pickle.load(open(out_file_name + "_model_" + get_date() + ".pkl", "rb"))
        # SHAP explainer for model
        shap_explainer = shap.TreeExplainer(model_final)
        # Model predictions
        out_data = get_out_data(data, model_final, X_all, y_all, args)
        out_data.write_csv(out_file_name + "_preds_" + get_date() + ".csv")  

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Evaluations                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #         
        crnt_report = create_report(model_final, out_data, display_scores=[args.metric, "aucpr"], metric=args.metric)
        pickle.dump(crnt_report, open(out_file_name + "_report_" + get_date() + ".pkl", "wb"))  

        importances, new_names = get_shap_importances(X_all, shap_explainer, args.lab_name)
        fig = create_report_plots(out_data.filter(pl.col("SET") == 1).select("TRUE_ABNORM"), 
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PREDS"),
                                  importances=importances)
        
        
        fig.savefig(out_plot_name + "_report_" + get_date() + ".png"); fig.savefig(out_plot_name + "_report_" + get_date() + ".pdf")   
        
        eval_metrics, all_conf_mats = eval_subset(out_data, "ABNORM_PREDS", "ABNORM_PROBS", "TRUE_ABNORM", "TRUE_VALUE", out_plot_name, out_dir, "all", args.n_boots, get_train_type(args.metric))
        if args.save_csv == 1:
            eval_metrics.loc[eval_metrics.F1.notnull()].write_csv(out_file_name + "_evals_" + get_date() + ".csv", sep=",")
            all_conf_mats.write_csv(out_file_name + "_confmats_" + get_date() + ".csv", sep=",")