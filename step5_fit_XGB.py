# Utils
import sys
sys.path.append(("/home/ivm/valid/scripts/"))
from utils import *
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

def score_model_rmse(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label()
    y_pred = model.predict(dmat)
    return(mean_squared_error(y_true, y_pred))

def score_model_logloss(model: xgb.core.Booster, dmat: xgb.core.DMatrix) -> float:
    y_true = dmat.get_label()
    y_pred = model.predict(dmat)
    return(log_loss(y_true, y_pred))
    
def optuna_objective(trial, base_params, X_train, y_train, eval_metric="tweedie", n_cv=5, cv_seed=2834, num_boost_round = 10000):
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
    opt_boosts = []
    scaler_X_cv = MinMaxScaler()

    for train_idx, val_idx in kf.split(X_train, y_train): 
        X_now_train, X_now_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_now_train, y_now_val = y_train.values[train_idx], y_train.values[val_idx]

        X_now_train.loc[:,transform_cols] = scaler_X_cv.fit_transform(X_now_train[transform_cols])
        X_now_val.loc[:,transform_cols] = scaler_X_cv.transform(X_now_val[transform_cols])

        dnowtrain = xgb.DMatrix(data=X_now_train, label=y_now_train, enable_categorical=True)
        dnowvalid = xgb.DMatrix(data=X_now_val, label=y_now_val, enable_categorical=True)
        # Train model
        model = xgb.train(params=params, 
                          dtrain=dnowtrain, 
                          num_boost_round=num_boost_round, 
                          evals=[(dnowtrain, "train"), (dnowvalid, "valid")], 
                          early_stopping_rounds=args.early_stop, 
                          verbose_eval=0)
        opt_boosts.append(model.best_iteration)
        if eval_metric != "logloss":
            mse_scores.append(score_model_rmse(model, dnowvalid))
            tweedie_scores.append(score_model_tweedie(model, dnowvalid))
        else:
            loglosses.append(score_model_logloss(model, dnowvalid))
    trial.set_user_attr("best_iteration", int(np.mean(opt_boosts)))
    print("Time: {} mean MSE: {:.2f} mean tweedie: {:.5f} mean boost: {}".format(timer.get_elapsed(), np.mean(mse_scores), np.mean(tweedie_scores), int(np.mean(opt_boosts))))
    # Return the mean Tweedie score across all folds
    if eval_metric == "tweedie": return -np.mean(tweedie_scores)
    # Return the mean MSE score across all folds
    if eval_metric == "mse": return np.mean(mse_scores)
    if eval_metric == "logloss": return np.mean(loglosses)

def get_data(goal, preds, file_path_data, file_path_meta, file_path_labels, file_path_icds, file_path_sumstats):
    ### Getting Data
    data = pd.read_csv(file_path_data, sep=",")
    metadata = pd.read_csv(file_path_meta, sep=",")
    labels = pd.read_csv(file_path_labels, sep=",")
    icds = pd.read_csv(file_path_icds, sep=",")
    sumstats = pd.read_csv(file_path_sumstats, sep=",")
    
    sex = metadata[["FINNGENID", "SEX"]]
    sex.SEX = sex.SEX.map({"female": 0, "male": 1})
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
    return(data, X_cols)

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
    parser.add_argument("--lr", type=float, default=0.4)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--run_step0", type=int, default=0)
    parser.add_argument("--time_step1", type=int, default=300)
    parser.add_argument("--metric", type=str, default="mse")

    args = parser.parse_args()

    return(args)

def init_logging(log_dir, log_file_name, date_time):
    logging.basicConfig(filename=log_dir+log_file_name+".log", level=logging.INFO, format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
    logger.info("Time: " + date_time + " Args: --" + ' --'.join(f'{k}={v}' for k, v in vars(args).items()))

def create_xgb_dts(data, X_cols, y_goal):
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
    X_test = valid_data[X_cols]
    y_test = valid_data[y_goal]
    
    X_all = data.drop(columns="SET")[X_cols]
    y_all = data.drop(columns="SET")[y_goal]
    
    ## Scaling
    scaler_base = MinMaxScaler()
    
    transform_cols = []
    for col in X_train.columns:
        if (X_train[col].dtype == "float64" or X_train[col].dtype == "in64") and not X_train[col].isin([0,1,np.nan]).all():
            transform_cols.append(col)
    X_train_scaled = X_train.copy()
    X_train_scaled.loc[:,transform_cols] = scaler_base.fit_transform(X_train_scaled[transform_cols])
    X_valid.loc[:,transform_cols] = scaler_base.transform(X_valid[transform_cols])
    X_test.loc[:,transform_cols] = scaler_base.transform(X_test[transform_cols])
    X_all.loc[:,transform_cols] = scaler_base.transform(X_all[transform_cols])
    # XGBoost datatype
    dtrain = xgb.DMatrix(data=X_train_scaled, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    dtest = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
    dall = xgb.DMatrix(data=X_all, label=y_all, enable_categorical=True)
    
    return(X_train, y_train, y_all, dtrain, dvalid, dtest, dall, scaler_base, transform_cols)
    
def get_base_params(metric):
    if metric == "tweedie":
        base_params = {"objective": "reg:tweedie", "eval_metric": "tweedie-nloglik@1.99"}
    elif metric == "mse":
        base_params = {"objective": "reg:squarederror"}
    elif metric == "logloss":
        base_params = {"objective": "binary:logistic"}
    return(base_params)

def run_speed_test(lr, base_params, dtrain, dvalid, early_stop):
    tic = time.time()
    params = {'tree_method': 'approx', 'learning_rate': lr, 'seed': 1239}
    params.update(base_params)
    model = xgb.train(params=params, dtrain=dtrain,
                      evals=[(dtrain, 'train'), (dvalid, 'valid')],
                      num_boost_round=10000,
                      early_stopping_rounds=early_stop,
                      verbose_eval=1)
        
    print(f'{time.time() - tic:.1f} seconds')
    print(model.best_iteration)
    print(score_model_tweedie(model, dvalid))
    print(score_model_rmse(model, dvalid))
    
if __name__ == "__main__":
    timer = Timer()
    args = get_parser_arguments()
    out_dir = args.res_dir + "xgb_" + args.metric + "_" + args.pred_descriptor + "/"
    log_dir = out_dir + "logs/"
    date = datetime.today().strftime("%Y-%m-%d")
    date_time = datetime.today().strftime("%Y-%m-%d-%H%M")
    file_name = args.lab_name + "_icd-preds_" + date
    file_path_data = args.file_path + args.lab_name + "_" + args.source_file_date + ".csv"
    file_path_meta = args.file_path + args.lab_name + "_" + args.source_file_date + "_meta.csv"
    log_file_name = args.lab_name + "_" + args.pred_descriptor + "_preds_" + date_time
    make_dir(log_dir)
    make_dir(args.res_dir)
    
    init_logging(log_dir, log_file_name, date_time)
    data, X_cols = get_data(args.goal, args.preds, file_path_data, file_path_meta, args.file_path_labels, args.file_path_icds, args.file_path_sumstats)
    X_train, y_train, y_all, dtrain, dvalid, dtest, dall, scaler_base, transform_cols = create_xgb_dts(data, X_cols, args.goal)
    base_params = get_base_params(args.metric)

    if args.run_step0 == 1: run_speed_test(args.lr, base_params, dtrain, dvalid, args.early_stop)
    else:
        timer = Timer()
        
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        tic = time.time()
        while time.time() - tic < args.time_step1:
            study.optimize(lambda trial: optuna_objective(trial, base_params, X_train, y_train, args.metric) , n_trials=1)

        logging.info('Stage 1 ==============================')
        logging.info(f'best score = {study.best_trial.value}')
        logging.info('boosting params ---------------------------')
        logging.info(f'fixed learning rate: {args.lr}')
        logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
        logging.info('best tree params --------------------------')
        for k, v in study.best_trial.params.items():
            logging.info(str(k)+':'+str(v))

        low_learning_rate = 0.01
        timer = Timer()
        params_fin = {}
        params_fin.update(base_params)
        params_fin.update(study.best_trial.params)
        params_fin['learning_rate'] = low_learning_rate
        model_final = xgb.train(params=params_fin, 
                                 dtrain=dtrain, 
                                 num_boost_round=10000,
                                 evals=[(dtrain, 'train'), (dvalid, 'valid')],
                                 early_stopping_rounds=5,
                                 verbose_eval=100)
        logging.info(timer.get_elapsed())
        # Get the best model
        logging.info('Stage 2 ==============================')
        logging.info(f'best score = {score_model_rmse(model_final, dvalid)}')
        logging.info(f'best score = {score_model_tweedie(model_final, dvalid)}')
        
        logging.info('boosting params ---------------------------')
        logging.info(f'fixed learning rate: {params_fin["learning_rate"]}')
        logging.info(f'best boosting round: {model_final.best_iteration}')
        
        keys = list(model_final.get_score(importance_type="gain").keys())
        values = list(model_final.get_score(importance_type="gain").values())
        
        top_gain = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        print(top_gain[0:10])
        logging.info(top_gain[0:10])
        top_gain.to_csv(out_dir + args.lab_name + "_gain_" + args.source_file_date + "_" + date + ".csv")
        
        keys = list(model_final.get_score(importance_type="weight").keys())
        values = list(model_final.get_score(importance_type="weight").values())
        
        top_weight = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by="score", ascending=False)
        print(top_weight[0:10])
        logging.info(top_weight[0:10])
        top_weight.to_csv(out_dir + args.lab_name + "_weight_" + args.source_file_date + "_" + date + ".csv")

        if args.metric != "logloss":
            y_pred = model_final.predict(dall)
            mse = mean_squared_error(y_all, y_pred)
            r2 = r2_score(y_all, y_pred)
            tweedie = score_model_tweedie(model_final, dall)
            print(f"Mean Squared Error: {mse} r2: {r2} tweedie: {tweedie}")
            logging.info(f"Mean Squared Error: {mse} r2: {r2} tweedie: {tweedie}")
    
            out_data = data[["FINNGENID", "EVENT_AGE", "LAST_VAL_DATE"]]
            out_data.loc[:,"VALUE_TRANSFORM"] = y_pred
            out_data = egfr_abnorm(out_data)
            out_data.columns = ["FINNGENID", "EVENT_AGE", "DATE", "VALUE", "ABNORM_CUSTOM"]
        else:
            y_pred = model_final.predict(dall)
            auc = roc_auc_score(y_all, y_pred)
            average_prec = average_precision_score(y_all, y_pred)
            print(f"AUC: {auc}   Average Precision: {average_prec}")
    
            out_data = data[["FINNGENID", "EVENT_AGE", "LAST_VAL_DATE"]]
            out_data.loc[:,"ABNORM_PROBS"] = y_pred
            out_data.loc[:,"ABNORM_PREDS"] = (out_data.ABNORM_PROBS>=0.5).astype("int64")
            out_data.columns = ["FINNGENID", "EVENT_AGE", "DATE", "ABNORM_PROBS", "ABNORM_PREDS"]
        out_data.to_csv(out_dir + args.lab_name + "_preds_" + args.source_file_date + "_" + date + ".csv", index=False)     
        pickle.dump(scaler_base, open(out_dir + args.lab_name + "_scaler_" + args.source_file_date + "_" + date + ".pkl", "wb"))
        pickle.dump(model_final, open(out_dir + args.lab_name + "_model_" + args.source_file_date + "_" + date + ".pkl", "wb"))
