# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from step3_abnorm import egfr_abnorm
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

def optuna_objective(trial, base_params, X_train, y_train, eval_metric="tweedie", reweight=False, n_cv=5, cv_seed=2834, num_boost_round = 10000):
    # Suggest hyperparameters
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ["approx", "hist"]),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 250),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 25, log=True)
    }
    params.update(base_params)

    # Prepare Cross-Validation
    kf = KFold(n_splits=n_cv, shuffle=True, random_state=cv_seed)
    mse_scores = []
    tweedie_scores = []
    loglosses = []
    f1s = []
    opt_boosts = []
    scaler_X_cv = MinMaxScaler()

    for train_idx, val_idx in kf.split(X_train, y_train): 
        X_now_train, X_now_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_now_train, y_now_val = y_train.values[train_idx], y_train.values[val_idx]


        X_now_train.loc[:,transform_cols] = scaler_X_cv.fit_transform(X_now_train[transform_cols])
        X_now_val.loc[:,transform_cols] = scaler_X_cv.transform(X_now_val[transform_cols])

        pos_weight = get_weighting(reweight, y_now_train)
        dnowtrain = xgb.DMatrix(data=X_now_train, label=y_now_train, enable_categorical=True, weight=get_weights(pos_weight, y_now_train))
        dnowvalid = xgb.DMatrix(data=X_now_val, label=y_now_val, enable_categorical=True, weight=get_weights(pos_weight, y_now_val))
        # Train model
        # problems with CV and pruning callback
        #pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{eval_metric}')
        model = xgb.train(params=params, 
                          dtrain=dnowtrain, 
                          num_boost_round=num_boost_round, 
                          evals=[(dnowtrain, "train"), (dnowvalid, "valid")], 
                          early_stopping_rounds=args.early_stop, 
        #                  callbacks=[pruning_callback],
                          verbose_eval=0)
        opt_boosts.append(model.best_iteration)
        if get_train_type(eval_metric) == "cont":
            mse_scores.append(score_model(model, dnowvalid, "mse")[1])
            tweedie_scores.append(score_model_tweedie(model, dnowvalid))
        else:
            loglosses.append(score_model(model, dnowvalid, "logloss")[1])
            f1s.append(score_model(model, dnowvalid, "f1")[1])

    trial.set_user_attr("best_iteration", int(np.mean(opt_boosts)))

    if eval_metric == "tweedie": return -np.mean(tweedie_scores)
    if eval_metric == "mse": return np.mean(mse_scores)
    if eval_metric == "logloss": return np.mean(loglosses)
    if eval_metric == "f1": return -np.mean(f1s)

def get_data(goal, preds, file_path_data, file_path_meta, file_path_labels, file_path_icds, file_path_sumstats, exclude_diag=False):
    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    labels = pd.read_csv(file_path_labels, sep=",")
    icds = pd.read_csv(file_path_icds, sep=",")
    sumstats = pd.read_csv(file_path_sumstats, sep=",")

    # Preprocessing data
    excludes = []
    if exclude_diag:
        excludes = pd.merge(metadata[["FINNGENID", "DIAG_DATE"]], labels[["FINNGENID", "START_DATE"]], on="FINNGENID", how="left")
        excludes = list(excludes.loc[excludes.DIAG_DATE < excludes.START_DATE].copy().FINNGENID)

    sex = metadata[["FINNGENID", "SEX"]]
    sex = sex.assign(SEX=sex.SEX.map({"female": 0, "male": 1}))
    age_last = data.groupby("FINNGENID").agg({"EVENT_AGE":"max"}).reset_index()

    # Merging
    data = pd.merge(sex, age_last, on="FINNGENID")
    data = pd.merge(data, labels[[*["FINNGENID", "SET", "START_DATE"], goal]], on="FINNGENID")
    data = pd.merge(data, icds, on="FINNGENID")
    data = pd.merge(data, sumstats, on="FINNGENID")

    if "LAST_ICD_DATE" in data.columns:
        data["TIME_LAST_ICD"] = (data.START_DATE.astype("datetime64[ns]") - data.LAST_ICD_DATE.astype("datetime64[ns]")).dt.days.astype(int)

    ## Predictors and outcome
    X_cols = []
    for pred in preds:
        if pred == "ICD_MAT":
            [X_cols.append(ICD_CODE) for ICD_CODE in icds.columns[np.logical_and(icds.columns != "FINNGENID", icds.columns != "LAST_ICD_DATE")]]
        elif pred == "SUMSTATS":
            [X_cols.append(SUMSTAT) for SUMSTAT in sumstats.columns[np.logical_and(sumstats.columns != "FINNGENID", sumstats.columns != "LAST_VAL_DATE")]]
        else:
            X_cols.append(pred)

    return(data, excludes, X_cols)


def create_xgb_dts(data, X_cols, y_goal, reweight):
    # Need later to be Float for scaling
    for col in data: 
        if data[col].dtype == "int64": data[col] = data[col].astype("float64")

    ## Split data
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
    
    ## Scaling
    scaler_base = MinMaxScaler()
    
    transform_cols = []
    for col in X_train.columns:
        if (X_train[col].dtype == "float64" or X_train[col].dtype == "int64") and not X_train[col].isin([0,1,np.nan]).all():
            transform_cols.append(col)
    X_train_scaled = X_train.copy()
    X_train_scaled.loc[:,transform_cols] = scaler_base.fit_transform(X_train_scaled[transform_cols])
    X_valid.loc[:,transform_cols] = scaler_base.transform(X_valid[transform_cols])
    X_test.loc[:,transform_cols] = scaler_base.transform(X_test[transform_cols])
    X_all.loc[:,transform_cols] = scaler_base.transform(X_all[transform_cols])
    # XGBoost datatype
    pos_weight = get_weighting(reweight, y_train)
    dtrain = xgb.DMatrix(data=X_train_scaled, label=y_train, enable_categorical=True, weight=get_weights(pos_weight, y_train))
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True, weight=get_weights(pos_weight, y_valid))
    dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True, weight=get_weights(pos_weight, y_test))
    dall = xgb.DMatrix(data=X_all, label=y_all, enable_categorical=True, weight=get_weights(pos_weight, y_all))
    
    return(X_train, y_train, y_valid, y_all, dtrain, dvalid, dtest, dall, scaler_base, transform_cols)
    
def get_base_params(metric, lr):
    base_params = {'tree_method': 'approx', 'learning_rate': lr, 'seed': 1239}

    if metric == "tweedie":
        base_params.update({"objective": "reg:tweedie", "eval_metric": "tweedie-nloglik@1.99"})
    elif get_train_type(metric) == "cont":
        base_params.update({"objective": "reg:squarederror", "eval_metric": metric})
    elif get_train_type(metric) == "bin":
        base_params.update({"objective": "binary:logistic", "eval_metric": metric})

    return(base_params)

def run_speed_test(dtrain, y_train, dvalid, args):
    base_params = get_base_params(args.metric, args.lr)

    tic = time.time()

    model = xgb.train(params=base_params, dtrain=dtrain,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      num_boost_round=10000,
                      early_stopping_rounds=args.early_stop,
                      verbose_eval=1)
        
    print(f'{time.time() - tic:.1f} seconds')
    print(model.best_iteration)
    
def run_step1(X_train, y_train, args, study_name):   
    """Runs the first step of the XGBoost optimization, which is to find the best hyperparameters for the model on a high learning rate.
       The function uses Optuna to optimize the hyperparameters. The function returns the best hyperparameters found.
       The function also logs the best hyperparameters found and the best boosting round."""     
    base_params = get_base_params(args.metric, args.lr)

    sampler = optuna.samplers.TPESampler(seed=42)
    if args.refit: optuna.delete_study(study_name=study_name, storage="sqlite:///optuna.db")
    study = optuna.create_study(direction='minimize', sampler=sampler, study_name=study_name, storage="sqlite:///optuna.db", load_if_exists=True)
    tic = time.time()
    while time.time() - tic < args.time_step1:
        study.optimize(lambda trial: optuna_objective(trial, base_params, X_train, y_train, args.metric, args.reweight) , n_trials=1)

    logging.info('Stage 1 ==============================')
    logging.info(f'best score = {study.best_trial.value}')
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {args.lr}')
    logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
    logging.info('best tree params --------------------------')

    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
    return(study.best_trial.params)

def run_step2(best_params, dtrain, y_train, dvalid, args):
    timer = Timer()
    params_fin = {}
    params_fin.update(get_base_params(args.metric, args.low_lr))
    params_fin.update(best_params)

    model_final = xgb.train(params=params_fin, 
                            dtrain=dtrain, 
                            num_boost_round=10000,
                            evals=[(dtrain, 'train'), (dvalid, 'valid')],
                            early_stopping_rounds=5,
                            verbose_eval=100)
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Stage 2 ==============================')
    #### TODO
    
    logging.info('boosting params ---------------------------')
    logging.info(f'fixed learning rate: {params_fin["learning_rate"]}')
    logging.info(f'best boosting round: {model_final.best_iteration}')
    logging.info("time taken {timer.get_elapsed()}")
    return(model_final)    

def save_importances(model_final, out_dir, args):
    keys = list(model_final.get_score(importance_type="gain").keys())
    values = list(model_final.get_score(importance_type="gain").values())
        
    top_gain = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    logging.info(top_gain[0:10])
    top_gain.to_csv(out_dir + args.lab_name + "_gain_" + args.source_file_date + "_" + get_date() + ".csv")
        
    keys = list(model_final.get_score(importance_type="weight").keys())
    values = list(model_final.get_score(importance_type="weight").values())
        
    top_weight = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
    logging.info(top_weight[0:10])
    top_weight.to_csv(out_dir + args.lab_name + "_weight_" + args.source_file_date + "_" + get_date() + ".csv")

def get_out_data(metric, model_final, dall, y_all, lab_name):
    out_data = data[["FINNGENID", "EVENT_AGE", "LAST_VAL_DATE", "SET", "ABNORM"]].rename(columns={"LAST_VAL_DATE": "DATE"})
    y_pred = model_final.predict(dall)

    if get_train_type(metric) == "cont":
        out_data = out_data.assign(VALUE = y_pred)
        out_data = get_abnorm_func_based_on_name(args.lab_name)(out_data, "VALUE").rename(columns={"ABNORM_CUSTOM": "ABNORM_PREDS"})
    else:
        y_pred = model_final.predict(dall)
        out_data = out_data.assign(ABNORM_PROBS = y_pred)
        out_data = out_data.assign(ABNORM_PREDS = (out_data.ABNORM_PROBS>0.5).astype("int64"))

    out_data = out_data.assign(TRUE_VALUE = y_all)
    out_data = get_abnorm_func_based_on_name(args.lab_name)(out_data, "TRUE_VALUE").rename(columns={"ABNORM_CUSTOM": "TRUE_ABNORM"})

    return(out_data)

def get_parser_arguments():
    #### Parsing and logging
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_dir", help="Path to the results directory", default="/home/ivm/valid/data/processed_data/step5_predict/1_year_buffer/")
    parser.add_argument("--file_path", type=str, help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end", default="/home/ivm/valid/data/processed_data/step3_abnorm/")
    parser.add_argument("--lab_name", type=str, help="Readable name of the measurement value.", required=True)
    parser.add_argument("--source_file_date", type=str, help="Date of file.", required=True)
    parser.add_argument("--file_path_labels", type=str, help="Path to outcome label data.", default="/home/ivm/valid/data/processed_data/step4_labels/krea_labels_2024-10-18_1-year.csv")
    parser.add_argument("--file_path_icds", type=str, default="/home/ivm/valid/icd_data/data/processed_data/step0/ICD_wide_r12_2024-10-18_min1pct_2024-12-09_krea.csv")
    parser.add_argument("--file_path_sumstats", type=str, default="/home/ivm/valid/data/processed_data/step5_predict/1_year_buffer/krea_2024-10-18_sumstats_2025-01-27.csv")
    parser.add_argument("--goal", type=str, help="Column name in labels file used for prediction.", default="y_MEAN")
    parser.add_argument("--preds", type=str, help="List of predictors. ICD_MAT takes all ICD-codes in the ICD-code file. SUMSTATS takes all summary statistics in the sumstats file.", default=["SUMSTATS", "EVENT_AGE", "SEX"], nargs="+")
    parser.add_argument("--pred_descriptor", type=str, help="Description of model predictors short.", required=True)
    parser.add_argument("--lr", type=float, help="Learning rate for hyperparamter optimization, can be high so this goes fast.", default=0.4)
    parser.add_argument("--low_lr", type=float, default=0.001, help="Learning rate for final model training.")
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--run_step0", type=int, default=0)
    parser.add_argument("--time_step1", type=int, default=300)
    parser.add_argument("--metric", type=str, default="mse")
    parser.add_argument("--exclude_diag", type=int, default=0)
    parser.add_argument("--refit", type=int, default=1)
    parser.add_argument("--reweight", type=int, default=1)
    parser.add_argument("--skip_model_fit", type=int, default=0)
    parser.add_argument("--save_csv", type=int, help="Whether to save the eval metrics file. If false can do a rerun on low boots.", default=1)
    parser.add_argument("--n_boots", type=int, help="Number of random samples for bootstrapping of metrics", default=500)


    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    study_name = args.lab_name + "_xgb_" + args.metric + "_" + args.pred_descriptor 
    out_dir = args.res_dir + study_name + "/"
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_" + args.pred_descriptor + "_preds_" + get_datetime()
    make_dir(out_dir)
    
    init_logging(out_dir, log_file_name, logger, args)
    data, excludes, X_cols = get_data(args.goal, args.preds, file_path_data, file_path_meta, args.file_path_labels, args.file_path_icds, args.file_path_sumstats)
    logging.info(f"Excluding {len(excludes)} individuals with prior diagnosis.")
    data = data.loc[~data.FINNGENID.isin(excludes)].copy()

    X_train, y_train, y_valid, y_all, dtrain, dvalid, dtest, dall, scaler_base, transform_cols = create_xgb_dts(data, X_cols, args.goal, args.reweight)

    if args.run_step0 == 1: run_speed_test(dtrain, y_train, dvalid, args)
    else:
        if not args.skip_model_fit:
            best_params = run_step1(X_train, y_train, args, study_name)
            model_final = run_step2(best_params, dtrain, y_train, dvalid, args)
            save_importances(model_final, out_dir, args)
            pickle.dump(model_final, open(out_dir + args.lab_name + "_" + args.source_file_date + "_model_" + get_date() + ".pkl", "wb"))
            pickle.dump(scaler_base, open(out_dir + args.lab_name + "_" + args.source_file_date + "_scaler_" + get_date() + ".pkl", "wb"))
        else:
            model_final = pickle.load(open(out_dir + args.lab_name + "_" + args.source_file_date + "_model_" + get_date() + ".pkl", "rb"))

        out_data = get_out_data(args.metric, model_final, dall, y_all, args.lab_name)
        out_data.to_csv(out_dir + args.lab_name + "_" + args.source_file_date + "_preds_" + get_date() + ".csv", index=False)   

        ## Report on all data
        crnt_report, fig = report(model_final, out_data, feature_labels=X_cols, metric=args.metric)
        fig.savefig(out_dir + args.lab_name + "_" + args.source_file_date + "_report_" + get_date() + ".png")   
        fig.savefig(out_dir + args.lab_name + "_" + args.source_file_date + "_report_" + get_date() + ".pdf")   
        pickle.dump(crnt_report, open(out_dir + args.lab_name + "_" + args.source_file_date + "_report_" + get_date() + ".pkl", "wb"))  
        eval_metrics, all_conf_mats = eval_preds(out_data, "ABNORM_PREDS", "ABNORM_PROBS", "TRUE_ABNORM", "TRUE_VALUE", out_dir + "/plots/", out_dir, args.n_boots, get_train_type(args.metric))
        if args.save_csv == 1:
            eval_metrics.loc[eval_metrics.F1.notnull()].to_csv(table_path + "_evals.csv", sep=",", index=False)
            all_conf_mats.to_csv(table_path + "_confmats.csv", sep=",", index=False)
        ## Report on individuals without prior abnorm
        crnt_report, fig = report(model_final, out_data.query("ABNORM==0"), feature_labels=X_cols, metric=args.metric)
        fig.savefig(out_dir + args.lab_name + "_" + args.source_file_date + "_report_noabnorm_" + get_date() + ".png")   
        fig.savefig(out_dir + args.lab_name + "_" + args.source_file_date + "_report_noabnorm_" + get_date() + ".pdf")   
        pickle.dump(crnt_report, open(out_dir + args.lab_name + "_" + args.source_file_date + "_report_noabnorm_" + get_date() + ".pkl", "wb"))  


