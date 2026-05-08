
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Optuna study                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import optuna
import polars as pl
def create_optuna_study(study_name: str, 
                        file_name: str, 
                        sampler: optuna.samplers,
                        model_fit_date: str,
                        res_dir: str,
                        refit: bool,
                        metric: str) -> optuna.study.Study:
    """Creates an Optuna study object."""
    store_name = "sqlite:///" + res_dir + file_name + "_" + model_fit_date + "_optuna.db"
    if refit: 
        try:
            optuna.delete_study(study_name=study_name, storage=store_name)
        except KeyError:
            print("Record not found, have to create new anyways.")
    if metric=="tweedie": direction="maximize"
    else: direction='minimize'
    study = optuna.create_study(direction=direction, 
                                sampler=sampler, 
                                study_name=study_name, 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                storage=store_name, 
                                load_if_exists=True)
    return(study)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Optuna runs                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import optuna
from typing import Union, Tuple
import xgboost as xgb
import time 
from general_utils import Timer
import logging
import numpy as np
def run_optuna_optim_cv(train: Union[xgb.DMatrix, Tuple[pl.DataFrame, pl.DataFrame]],
                        lab_name: str,
                        refit: bool,
                        time_optim: int,
                        n_trials: int,
                        early_stop: int,
                        n_folds: int,
                        study_name: str,
                        res_dir: str,
                        model_type: str,
                        model_fit_date: str,
                        base_params: dict=None) -> dict:   
    """Runs the first step of the XGBoost optimization, which is to find the best hyperparameters for the model on a high learning rate.
       Uses Optuna to optimize the hyperparameters. The function returns the best hyperparameters found.
       Logs the best hyperparameters found and the best boosting round."""     
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    sampler = optuna.samplers.TPESampler(seed=429)
    study = create_optuna_study(study_name, 
                                lab_name, 
                                sampler, 
                                model_fit_date, 
                                res_dir, 
                                refit,
                                base_params["eval_metric"])
    tic = time.time()
    timer = Timer()
    np.random.seed(9234)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Pick objective                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    optuna_objective = lambda trial: optuna_xgb_cv_objective(trial, 
                                                             base_params, 
                                                             train[0],
                                                             train[1],
                                                             early_stop=early_stop,
                                                             n_folds=n_folds)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Running trials                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if n_trials == -1:
        while time.time() - tic < time_optim:
            study.optimize(optuna_objective, n_trials=1)
    else:
        study.optimize(optuna_objective, n_trials=n_trials)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info('Optuna optimization ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info(f'best score = {study.best_trial.value}')
    logging.info('best tree params --------------------------')
    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
    # if model_type == "xgb":
    #     logging.info('boosting params ---------------------------')
    #     logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
        
    return(study.best_trial.params)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Optuna objectives                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import xgboost as xgb
import optuna
import numpy as np
import gc
from sklearn.model_selection import KFold
def optuna_xgb_cv_objective(trial: optuna.Trial, 
                            base_params: dict, 
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            early_stop: int = 5,
                            n_folds: int = 5) -> float:
    """Objective function for Optuna optimization using cross-validation.
    Returns the mean CV score."""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Suggested hyperparameters                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'min_child_weight': trial.suggest_int("min_child_weight", 5, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 15),
        'gamma': trial.suggest_float('gamma', 0, 15),
    }
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Setting up trial                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params.update(base_params)
    cv_results = []
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Cross-validation                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if base_params["eval_metric"] in ["q10", "q05", "q25", "q50", "q75", "q90", "q95"]:
        del params["eval_metric"]
        cv_results = xgb.cv(params=params,
                            dtrain=xgb.DMatrix(X_train, label=y_train),
                            num_boost_round=200,
                            nfold=n_folds,
                            custom_metric=quantile_eval(base_params["eval_metric"]),
                            stratified=True,
                            early_stopping_rounds=early_stop,
                            seed=42,
                            verbose_eval=False
        )
    else:
        kf = KFold(n_splits=n_folds, 
                   shuffle=True, 
                   random_state=42)
        for tr_idx, va_idx in kf.split(X_train):
            dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
            dval = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx])
            booster = xgb.train(params, 
                                dtrain, 
                                num_boost_round=200,
                                evals=[(dval, "val")], 
                                early_stopping_rounds=early_stop,
                                verbose_eval=False)
            cv_results.append(booster.best_score)
            del dtrain, dval, booster
            gc.collect()
    
    return np.mean(np.asarray(cv_results))