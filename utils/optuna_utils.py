import optuna

def create_optuna_study(study_name: str, 
                        file_name: str, 
                        sampler: optuna.samplers,
                        model_fit_date: str,
                        res_dir: str,
                        refit: bool) -> optuna.study.Study:
    """Creates an Optuna study object."""
    store_name = "sqlite:///" + res_dir + file_name + "_" + model_fit_date + "_optuna.db"
    if refit: 
        try:
            optuna.delete_study(study_name=study_name, storage=store_name)
        except KeyError:
            print("Record not found, have to create new anyways.")
    study = optuna.create_study(direction='minimize', 
                                sampler=sampler, 
                                study_name=study_name, 
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
                                storage=store_name, 
                                load_if_exists=True)
    return(study)


from typing import Union
import xgboost as xgb
import time 
from general_utils import Timer
import logging
import numpy as np
import torch
def run_optuna_optim(train: Union[xgb.DMatrix, torch.utils.data.DataLoader], 
                     valid: Union[xgb.DMatrix,  torch.utils.data.DataLoader],
                     test: Union[xgb.DMatrix,  torch.utils.data.DataLoader], 
                     lab_name: str,
                     refit: bool,
                     time_optim: int,
                     n_trials: int,
                     study_name: str,
                     res_dir: str,
                     model_fit_date: str,
                     base_params: dict) -> dict:   
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
                                refit)
    tic = time.time()
    timer = Timer()
    np.random.seed(9234)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Pick objective                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if base_params["model_type"] == "xgb":
        optuna_objective = lambda trial: optuna_xgb_objective(trial, 
                                                              base_params, 
                                                              train, 
                                                              valid)
    elif base_params["model_type"] == "torch":
        optuna_objective = lambda trial: optuna_torch_objective(trial, 
                                                                base_params, 
                                                                train, 
                                                                valid, 
                                                                test)
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
    logging.info('boosting params ---------------------------')
    logging.info(f'best boosting round: {study.best_trial.user_attrs["best_iteration"]}')
    logging.info('best tree params --------------------------')
    for k, v in study.best_trial.params.items(): logging.info(str(k)+':'+str(v))
        
    return(study.best_trial.params)

import xgboost as xgb
import optuna
def optuna_xgb_objective(trial: optuna.Trial, 
                         base_params: dict, 
                         dtrain: xgb.DMatrix, 
                         dvalid: xgb.DMatrix, 
                         dtest: xgb.DMatrix) -> float:
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

try:
    from termcolor import colored
except:
    def colored(string, color): return(string)
import optuna
import torch
from utils_final import epochs_run
from torch_utils import get_model, get_torch_optimizer
def optuna_torch_objective(trial: optuna.Trial, 
                           base_params: dict,
                           train_mbs, 
                           valid_mbs, 
                           test_mbs):
    """Setup the objective function for Optuna."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Suggested hyperparameters                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    params = {
        'embed_dim_exp': trial.suggest_int('embed_dim_exp', 2, 10, 1),
        'hidden_size_exp': trial.suggest_int('hidden_size_exp', 2, 10, 1),
        'dropout_r': trial.suggest_int("dropout_r", 0, 5, 1),
        'L2': trial.suggest_float('L2', 1e-6, 1e-2, log=True),
        'lr': trial.suggest_int('lr', 1, 11, 2),
        #'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adadelta', 'adagrad', 'adamax', 'asgd', 'rmsprop', 'rprop', 'sgd']),
        'optimizer': trial.suggest_categorical('optimizer', ['adagrad', 'adamax']),

    }
    if base_params["model_name"] == 'RNN': 
        params['n_layers'] = trial.suggest_int('n_layers', 1, 5)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Setting up trial                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # Train model
    print(params)
    ehr_model = get_model(model_name=base_params["model_name"],
                          embed_dim_exp=params["embed_dim_exp"],
                          hidden_size_exp=params["hidden_size_exp"],
                          n_layers=params["n_layers"],
                          dropout_r=params["dropout_r"],
                          cell_type=base_params["cell_type"],
                          bii=base_params["bii"],
                          time=base_params["time"],
                          preTrainEmb=base_params["preTrainEmb"],
                          input_size=base_params["input_size"])   
    optimizer = get_torch_optimizer(ehr_model, 
                                    base_params["eps"],
                                    params["lr"], 
                                    params["L2"],
                                    params["optimizer_name"])
    if torch.cuda.is_available(): ehr_model = ehr_model.cuda() 
    #######Step3. Train, validation and test. default: batch shuffle = true 
    try:
        best_val_loss, best_val_ep, best_model = \
                    epochs_run(base_params["epochs"], 
                                train = train_mbs, 
                                valid = valid_mbs, 
                                test = test_mbs, 
                                model = ehr_model, 
                                optimizer = optimizer,
                                trial=trial,
                                shuffle = True, 
                                model_name=base_params["model_name"], 
                                patience=base_params["patience"])
        trial.set_user_attr("best_iteration", int(best_val_ep))
        if best_val_loss is not None:
            return best_val_loss
        else:
            raise optuna.TrialPruned()
    # Keyboard interrupt to prune
    except KeyboardInterrupt:
        print(colored('-' * 89, 'green'))
        print(colored('Manual pruning','green'))
        raise optuna.TrialPruned()

