
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
import polars as pl
from typing import Union, Tuple
import xgboost as xgb
import time 
from general_utils import Timer
import numpy as np
def run_optuna_optim(train: Union[xgb.DMatrix,Tuple[pl.DataFrame, pl.DataFrame]], 
                     valid: Union[xgb.DMatrix,  Tuple[pl.DataFrame, pl.DataFrame]],
                     test: Union[xgb.DMatrix, Tuple[pl.DataFrame, pl.DataFrame]], 
                     lab_name: str,
                     refit: bool,
                     time_optim: int,
                     n_trials: int,
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
    if model_type == "elr":
        optuna_objective = lambda trial: optuna_elr_objective(trial,
                                                              train[0],
                                                              train[1],
                                                              valid[0],
                                                              valid[1])
    if model_type == "xgb":
        optuna_objective = lambda trial: optuna_xgb_objective(trial, 
                                                              base_params, 
                                                              train, 
                                                              valid)
    elif model_type == "cat":
        optuna_objective = lambda trial: optuna_cat_objective(trial, 
                                                                base_params, 
                                                                train[0],
                                                                train[1],
                                                                valid[0],
                                                                valid[1])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Running trials                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if n_trials == 1:
        while time.time() - tic < time_optim:
            study.optimize(optuna_objective, n_trials=1)
    else:
        study.optimize(optuna_objective, n_trials=n_trials)

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
                                                          train[1])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Running trials                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if n_trials == 1:
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
    if model_type == "xgb":
        logging.info('boosting params ---------------------------')
        logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
        
    return(study.best_trial.params)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                 Optuna objectives                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
try:
    import catboost as cat
except ImportError:
    pass
import optuna
def optuna_cat_objective(trial: optuna.Trial, 
                         base_params: dict, 
                         X_train,
                         y_train,
                         X_valid,
                         y_valid) -> float:
    """Objective function for the Optuna optimization. Returns the last value of the metric on the validation set."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Suggested hyperparameters                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'min_data_in_leaf': trial.suggest_int("min_data_in_leaf", 20, 100),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1, 10),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'random_strength': trial.suggest_float('random_strength', 1, 10),   
        # < Subsampling
       'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # row sampling
       'rsm': trial.suggest_float('rsm', 0.5, 1.0),              # feature sampling
    }
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Setting up trial                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params.update(base_params)
    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, base_params["eval_metric"])
    evals_result = dict()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Train model                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    model = cat.CatBoostClassifier(**params)
    model.fit(X_train.to_pandas(), y_train.to_pandas(), 
              eval_set=(X_valid.to_pandas(), y_valid.to_pandas()), 
              verbose=0, 
              early_stopping_rounds=5, 
              callbacks=[pruning_callback])
    pruning_callback.check_pruned()
    trial.set_user_attr("best_iteration", int(model.best_iteration_))

    # Get the training metric value at best iteration
    eval_results = model.get_evals_result()
    metric_name = list(eval_results['validation'].keys())[0]
    metric_value = eval_results['validation'][metric_name][model.best_iteration_]


    return metric_value

import xgboost as xgb
import optuna
import numpy as np
def optuna_xgb_cv_objective(trial: optuna.Trial, 
                             base_params: dict, 
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             n_folds: int = 5) -> float:
    """Objective function for Optuna optimization using cross-validation.
    Returns the mean CV score."""
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Suggested hyperparameters                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params = {
        'tree_method': trial.suggest_categorical('tree_method', ["approx", "hist"]),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
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
                            early_stopping_rounds=5,
                            seed=42,
                            verbose_eval=False
        )
    else:
        cv_results = xgb.cv(params=params,
                            dtrain=xgb.DMatrix(X_train, label=y_train),
                            num_boost_round=200,
                            nfold=n_folds,
                            stratified=True,
                            early_stopping_rounds=5,
                            seed=42,
                            verbose_eval=False
        )
        
    
    # Get best iteration and score
    best_iteration = len(cv_results)
    trial.set_user_attr("best_iteration", best_iteration)
    
    # Return mean of test metric at best iteration (last row)
    metric_name = f"test-{base_params['eval_metric']}-mean"
    best_score = cv_results[metric_name].iloc[-1]
    
    return best_score

import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from model_eval_utils import quantile_eval
import xgboost as xgb
import optuna
def optuna_xgb_objective(trial: optuna.Trial, 
                         base_params: dict, 
                         dtrain: xgb.DMatrix, 
                         dvalid: xgb.DMatrix) -> float:
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
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 15),
        'gamma': trial.suggest_float('gamma', 0, 15),
    }

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Setting up trial                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f'valid-{base_params["eval_metric"]}')
    evals_result = dict()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Train model                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if base_params["eval_metric"] in ["q10", "q05", "q25", "q50", "q75", "q90", "q95"]:
        del params["eval_metric"]
        model = xgb.train(params=params, 
                          dtrain=dtrain, 
                          num_boost_round=200, 
                          evals=[(dtrain, "train"), (dvalid, "valid")], 
                          evals_result=evals_result,
                          custom_metric=quantile_eval(base_params["eval_metric"]),
                          early_stopping_rounds=5, 
                          verbose_eval=0,
                          callbacks=[pruning_callback])
    else:
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
    return evals_result["valid"][base_params["eval_metric"]][-1] 