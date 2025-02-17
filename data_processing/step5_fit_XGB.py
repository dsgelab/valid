# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from utils import *
from processing_utils import * 
from model_eval_utils import *
# Standard stuff
import pandas as pd
# Logging and input
import logging
logger = logging.getLogger(__name__)
import argparse
from datetime import datetime
import xgboost as xgb
import optuna
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, r2_score, average_precision_score, d2_tweedie_score, log_loss

def score_model_tweedie(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label()
    y_pred = model.predict(dmat)
    y_pred = np.clip(y_pred, a_min=1e-10, a_max=None)
    try:
        return(d2_tweedie_score(y_true, y_pred, power=3))
    except:
        return(np.nan)

def score_model(model: xgb.core.Booster, dmat: xgb.core.DMatrix, metric: str) -> (str, float):
    y_true = dmat.get_label()
    y_pred = model.predict(dmat)
    if get_train_type(metric) == "bin": y_pred = (y_pred > 0.5).astype(int)
    scores = get_score_func_based_on_metric(metric)(y_true, y_pred)
    return(metric, scores)
    
def get_weighting(reweight, y_train):
    if reweight: return(y_train.shape[0] - np.sum(y_train)) / np.sum(y_train)
    else: return(1)

def get_weights(pos_weight, y):
    return np.where(y == 1, pos_weight, 1)

def optuna_objective(trial, base_params, dtrain, dvalid, eval_metric="tweedie", num_boost_round = 10):
    # Suggest hyperparameters

    params = {
        'tree_method': trial.suggest_categorical('tree_method', ["approx", "hist"]),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int("min_child_weight", 5, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10, log=True),
        'gamma': trial.suggest_float('gamma', 1, 10, log=True),
    }
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{eval_metric}')

    # Train model
    evals_result = dict()
    model = xgb.train(params=params, 
                      dtrain=dtrain, 
                      num_boost_round=200, 
                      evals=[(dtrain, "train"), (dvalid, "valid")], 
                      evals_result=evals_result,
                      early_stopping_rounds=5, 
                      verbose_eval=0,
                      callbacks=[pruning_callback])
    trial.set_user_attr("best_iteration", int(model.best_iteration))
    return evals_result["valid"][eval_metric][-1]

def get_data(goal, preds, file_path_data, file_path_labels, file_path_icds, file_path_sumstats, file_path_atcs):
    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",").drop(columns=["EVENT_AGE"])
    labels = pd.read_csv(file_path_labels, sep=",")
    if file_path_icds != "": 
        icds = pd.read_csv(file_path_icds, sep=",")
        data = pd.merge(data, icds, on="FINNGENID")
    if file_path_atcs != "": 
        atcs = pd.read_csv(file_path_atcs, sep=",")
        data = pd.merge(data, atcs, on="FINNGENID")
    if file_path_sumstats != "": 
        sumstats = pd.read_csv(file_path_sumstats, sep=",")
        data = pd.merge(data, sumstats, on="FINNGENID")

    data = data.assign(SEX=data.SEX.map({"female": 0, "male": 1}))
    #age_last = data.groupby("FINNGENID").agg({"EVENT_AGE":"max"}).reset_index()

    #data = pd.merge(data, age_last, on="FINNGENID")
    # keeping also continuous goal col in case of binary predictions
    goal_split = goal.split("_")
    if goal_split[-1] == "ABNORM" or goal == "y_DIAG":
        new_goal = "_".join(goal_split[0:len(goal_split)-1])
        if not new_goal in data.columns: new_goal = "y_MEAN"
        # Merging
        data = pd.merge(data, labels[[*["FINNGENID", "SET", "START_DATE", "EVENT_AGE"], goal, new_goal]], on="FINNGENID")
    else:
        data = pd.merge(data, labels[[*["FINNGENID", "SET", "START_DATE", "EVENT_AGE"], goal]], on="FINNGENID")
    if "LAST_ICD_DATE" in data.columns:
        data["TIME_LAST_ICD"] = (data.START_DATE.astype("datetime64[ns]") - data.LAST_ICD_DATE.astype("datetime64[ns]")).dt.days.astype(int)
    if "LAST_ATC_DATE" in data.columns:
        data["LAST_ATC_DATE"] = (data.START_DATE.astype("datetime64[ns]") - data.LAST_ICD_DATE.astype("datetime64[ns]")).dt.days.astype(int)

    ## Predictors and outcome
    X_cols = []
    for pred in preds:
        if pred == "ICD_MAT":
            [X_cols.append(ICD_CODE) for ICD_CODE in icds.columns[np.logical_and(icds.columns != "FINNGENID", icds.columns != "LAST_ICD_DATE")]]
        elif pred == "ATC_MAT":
            [X_cols.append(ATC_CODE) for ATC_CODE in atcs.columns[np.logical_and(atcs.columns != "FINNGENID", atcs.columns != "LAST_ATC_DATE")]]
        elif pred == "SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in sumstats.columns[np.logical_and(sumstats.columns != "FINNGENID", sumstats.columns != "LAST_VAL_DATE")]]
        else:
            X_cols.append(pred)
    return(data, X_cols)

def create_xgb_dts(data, X_cols, y_goal, reweight):
    # Need later to be Float for scaling
    for col in data: 
        if data[col].dtype == "int64": data[col] = data[col].astype("float64")

    #X_train, X_rest, y_train, y_rest = train_test_split(data[X_cols], data[y_goal], shuffle=True, random_state=3291, test_size=0.4, train_size=0.6, stratify=data[y_goal])
    #X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, shuffle=True, random_state=391, test_size=0.5, train_size=0.5, stratify=y_rest)

    # # Split data
    train_data = data.loc[data.SET == 0].drop(columns="SET")
    valid_data = data.loc[data.SET == 1].drop(columns="SET")
    test_data = data.loc[data.SET == 2].drop(columns="SET")
    ## XGB datatype prep
    X_train = train_data[X_cols]
    y_train = train_data[y_goal]
    X_valid = valid_data[X_cols]
    y_valid = valid_data[y_goal]
    X_test = test_data[X_cols]
    y_test = test_data[y_goal]
    
    X_all = data.drop(columns="SET")[X_cols]
    y_all = data.drop(columns="SET")[y_goal]
    #print(f"have {round(y_train.sum()/len(y_train)*100, 2)}% (N= {y_train.sum()} )in train and {round(y_valid.sum()/len(y_valid)*100, 2)}% (N={y_valid.sum()}) in validation ")

    ## Scaling
    scaler_base = RobustScaler()
    
    transform_cols = []
    for col in X_train.columns:
        if (X_train[col].dtype == "float64" or X_train[col].dtype == "int64") and not X_train[col].isin([0,1,np.nan]).all():
            transform_cols.append(col)
            
    X_train.loc[:,transform_cols] = scaler_base.fit_transform(X_train[transform_cols])
    X_valid.loc[:,transform_cols] = scaler_base.transform(X_valid[transform_cols])
    X_test.loc[:,transform_cols] = scaler_base.transform(X_test[transform_cols])
    X_all.loc[:,transform_cols] = scaler_base.transform(X_all[transform_cols])
    # XGBoost datatype
    pos_weight = get_weighting(reweight, y_train)
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True, weight=get_weights(pos_weight, y_train))
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True, weight=get_weights(pos_weight, y_valid))
    dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True, weight=get_weights(pos_weight, y_test))
    dall = xgb.DMatrix(data=X_all, label=y_all, enable_categorical=True, weight=get_weights(pos_weight, y_all))
    
    return(X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, dtrain, dvalid, scaler_base, transform_cols)
    
def get_base_params(metric, lr):
    base_params = {'tree_method': 'approx', 'learning_rate': lr, 'seed': 139}

    if metric == "tweedie":
        base_params.update({"objective": "reg:tweedie", "eval_metric": "tweedie-nloglik@1.99"})
    elif get_train_type(metric) == "cont":
        base_params.update({"objective": "reg:squarederror", "eval_metric": metric})
    elif get_train_type(metric) == "bin":
        base_params.update({"objective": "binary:logistic", "eval_metric": metric})

    return(base_params)

def run_speed_test(dtrain, dvalid, args):
    base_params = get_base_params(args.metric, args.lr)

    tic = time.time()

    model = xgb.train(params=base_params, dtrain=dtrain,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      num_boost_round=10,
                      early_stopping_rounds=5,
                      verbose_eval=1)
        
    print(f'{time.time() - tic:.1f} seconds')
    print(model.best_iteration)
    
def run_step1(dtrain, dvalid, args, study_name):   
    """Runs the first step of the XGBoost optimization, which is to find the best hyperparameters for the model on a high learning rate.
       The function uses Optuna to optimize the hyperparameters. The function returns the best hyperparameters found.
       The function also logs the best hyperparameters found and the best boosting round."""     
    base_params = get_base_params(args.metric, args.lr)

    sampler = optuna.samplers.TPESampler(seed=429)
    if args.refit: optuna.delete_study(study_name=study_name, storage="sqlite:///" + args.lab_name + "_optuna.db")
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name, storage="sqlite:///" + args.lab_name + "_optuna.db", load_if_exists=True)
    tic = time.time()
    timer = Timer()
    np.random.seed(9234)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    if args.n_trials == 1:
        while time.time() - tic < args.time_step1:
            study.optimize(lambda trial: optuna_objective(trial, base_params, dtrain, dvalid, args.metric) , n_trials=1)
    else:
        study.optimize(lambda trial: optuna_objective(trial, base_params, dtrain, dvalid, args.metric) , n_trials=args.n_trials)

    
    logging.info('Stage 1 ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info(f'best score = {study.best_trial.value}')
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {args.lr}')
    logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
    logging.info('best tree params --------------------------')
    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
        
    return(study.best_trial.params)

def run_step2(best_params, X_train, y_train, X_valid, y_valid, args):
    timer = Timer()
    params_fin = {}
    params_fin.update(get_base_params(args.metric, args.low_lr))
    params_fin.update(best_params)
    print(params_fin)
    np.random.seed(9234)

    clf = xgb.XGBClassifier(**params_fin, early_stopping_rounds=args.early_stop, n_estimators=1000)
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Stage 2 ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {params_fin["learning_rate"]}')
    logging.info(f'best boosting round: {clf.best_iteration}')
    logging.info("time taken {timer.get_elapsed()}")
    return(clf)    

def save_importances(labels, model_final, out_dir, args):
    values = list(model_final.feature_importances_)
        
    top_gain = pd.DataFrame(data=values, index=labels, columns=["score"]).sort_values(by="score", ascending=False)
    logging.info(top_gain[0:10])
    top_gain.to_csv(out_dir + args.lab_name + "_importance_" + get_date() + ".csv")
        
def get_out_data(data, model_final, X_all, y_all, args):
    out_data = data[["FINNGENID", "EVENT_AGE", "LAST_VAL_DATE", "SET", "ABNORM"]].rename(columns={"DATE": "LAST_VAL_DATE", "ABNORM": "N_PRIOR_ABNORMS"})

    if get_train_type(args.metric) == "cont":    
        y_pred = model_final.predict(X_all)
        out_data = out_data.assign(VALUE = y_pred)
        out_data = get_abnorm_func_based_on_name(args.lab_name)(out_data, "VALUE").rename(columns={"ABNORM_CUSTOM": "ABNORM_PREDS"})
    else:
        y_pred = model_final.predict_proba(X_all)[:,1]
        out_data = out_data.assign(ABNORM_PROBS = y_pred)
        out_data = out_data.assign(ABNORM_PREDS = (out_data.ABNORM_PROBS>0.5).astype("int64"))

    if get_train_type(args.metric) == "cont":
        out_data = out_data.assign(TRUE_VALUE = y_all)
        out_data = get_abnorm_func_based_on_name(args.lab_name)(out_data, "TRUE_VALUE").rename(columns={"ABNORM_CUSTOM": "TRUE_ABNORM"})
    else:
        goal_split = args.goal.split("_")
        new_goal = "_".join(goal_split[0:len(goal_split)-1])
        if not new_goal in data.columns: new_goal = "y_MEAN"

        y_cont_all = data[new_goal]
        out_data = out_data.assign(TRUE_VALUE = y_cont_all)
        out_data = out_data.assign(TRUE_ABNORM = y_all)
    return(out_data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step5_predict/1_year_buffer/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step3_abnorm/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="")
    parser.add_argument("--file_path_icds", type=str, default="")
    parser.add_argument("--file_path_atcs", type=str, default="")
    parser.add_argument("--file_path_sumstats", type=str, default="")
    parser.add_argument("--source_file_date", type=str, default="")

    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--preds", type=str, help="List of predictors. ICD_MAT takes all ICD-codes in the ICD-code file. SUMSTATS takes all summary statistics in the sumstats file.", default=["SUMSTATS", "EVENT_AGE", "SEX"], nargs="+")
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--lr", type=float, help="Learning rate for hyperparamter optimization, can be high so this goes fast.", default=0.4)
    parser.add_argument("--low_lr", type=float, default=0.001, help="Learning rate for final model training.")
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--run_step0", type=int, default=0)
    parser.add_argument("--time_step1", type=int, default=300)
    parser.add_argument("--metric", type=str, default="mse")
    parser.add_argument("--refit", type=int, default=1)
    parser.add_argument("--reweight", type=int, default=1)
    parser.add_argument("--skip_model_fit", type=int, default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics", default=500)
    parser.add_argument("--n_cv", type=int, help="Number of random samples for bootstrapping of metrics", default=5)
    parser.add_argument("--n_trials", type=int, help="Number of random samples for bootstrapping of metrics", default=1)

    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()

    study_name = "xgb_" + str(args.metric) + "_" + args.pred_descriptor +  "_reweight" + str(args.reweight)

    out_dir = args.res_dir + study_name + "/"
    out_name = args.lab_name 
    log_file_name = args.lab_name + "_" + args.pred_descriptor + "_preds_" + get_datetime()
    make_dir(out_dir + "plots/")
    
    init_logging(out_dir, log_file_name, logger, args)
    print(args)

    data, X_cols = get_data(args.goal, args.preds, args.file_path, args.file_path_labels, args.file_path_icds, args.file_path_sumstats, args.file_path_atcs)
    X_train, y_train, X_valid, y_valid, X_test, y_test, X_all, y_all, dtrain, dvalid, scaler_base, transform_cols = create_xgb_dts(data, X_cols, args.goal, args.reweight)

    if args.run_step0 == 1: run_speed_test(dtrain, dvalid, args)
    else:
        if not args.skip_model_fit:
            best_params = run_step1(dtrain, dvalid, args, study_name)
            logging.info(timer.get_elapsed())
            model_final = run_step2(best_params, X_train, y_train, X_valid, y_valid, args)
            save_importances(X_train.columns, model_final, out_dir, args)
            pickle.dump(model_final, open(out_dir + args.lab_name + "_model_" + get_date() + ".pkl", "wb"))
            pickle.dump(scaler_base, open(out_dir + args.lab_name + "_scaler_" + get_date() + ".pkl", "wb"))
        else:
            model_final = pickle.load(open(out_dir + args.lab_name + "_model_" + get_date() + ".pkl", "rb"))

        out_data = get_out_data(data, model_final, X_all, y_all, args)
        out_data.to_csv(out_dir + out_name + "_preds_" + get_date() + ".csv", index=False)   
        ## Report on all data
        crnt_report, fig = create_report(model_final, out_data, feature_labels=X_cols, display_scores=[args.metric], metric=args.metric, importance_plot=True)
        fig.savefig(out_dir + "plots/" + out_name + "_report_" + get_date() + ".png")   
        fig.savefig(out_dir + "plots/" + out_name + "_report_" + get_date() + ".pdf")   
        pickle.dump(crnt_report, open(out_dir + args.lab_name + "_report_" + get_date() + ".pkl", "wb"))  
        
        eval_metrics, all_conf_mats = eval_subset(out_data, "ABNORM_PREDS", "ABNORM_PROBS", "TRUE_ABNORM", "TRUE_VALUE", out_dir + "plots/" + out_name, out_dir, "all", args.n_boots, get_train_type(args.metric))

        ## Report on individuals without prior abnorm
        crnt_report, fig = create_report(model_final, out_data.query("N_PRIOR_ABNORMS==0"), feature_labels=X_cols, display_scores=[args.metric], metric=args.metric)
        fig.savefig(out_dir + "plots/" + out_name + "_report_noabnorm_" + get_date() + ".png")   
        fig.savefig(out_dir + "plots/" + out_name + "_report_noabnorm_" + get_date() + ".pdf")   
        pickle.dump(crnt_report, open(out_dir + out_name + "_report_noabnorm_" + get_date() + ".pkl", "wb"))  

        eval_metrics_2, all_conf_mats_2 = eval_subset(out_data.query("N_PRIOR_ABNORMS==0"), "ABNORM_PREDS", "ABNORM_PROBS", "TRUE_ABNORM", "TRUE_VALUE", out_dir + "plots/" + out_name, out_dir, "noabnorm", args.n_boots, get_train_type(args.metric))
        eval_metrics = pd.concat([eval_metrics, eval_metrics_2])
        all_conf_mats = pd.concat([all_conf_mats, all_conf_mats_2])
        if args.save_csv == 1:
            eval_metrics.loc[eval_metrics.F1.notnull()].to_csv(out_dir + out_name + "_evals_" + get_date() + ".csv", sep=",", index=False)
            all_conf_mats.to_csv(out_dir + out_name + "_confmats_" + get_date() + ".csv", sep=",", index=False)
            