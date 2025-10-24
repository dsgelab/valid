import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import Timer
import logging
import numpy as np
import xgboost as xgb
from model_eval_utils import get_train_type

"""Returns the base parameters for the XGBoost model.

    Args:
        metric (str): The evaluation metric.
        lr (float): The learning rate.
        n_classes (int): The number of classes (for multi-class classification).

    Returns:
        dict: A dictionary containing the base parameters for the XGBoost model.
"""
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

def get_cont_goal_col_name(goal: str,
                           col_names: list) -> str:
    """Returns the column name of the continuous goal.
       If first part of goal name in column names, i.e. y_MEDIAN in the case of y_MEDIAN_ABNORM, returns the first part.
       Otherwise returns y_MEAN. If not present, certain plotting will be skipped"""
    # Binary prediction tasks
    goal_split = goal.split("_")
    if goal_split[-1] == "ABNORM" or goal == "y_DIAG":
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
    

"""Fits the final XGB model with the best hyperparameters found in the optimization step.

    Args:
        best_params (dict): The best hyperparameters found in the optimization step.
        X_train (pl.DataFrame): The training features.
        y_train (pl.DataFrame): The training target.
        X_valid (pl.DataFrame): The validation features.
        y_valid (pl.DataFrame): The validation target.
        metric (str): The evaluation metric.
        low_lr (float): The learning rate.  
        early_stop (int): The number of rounds for early stopping.
        n_classes (int): The number of classes (for multi-class classification).

    Returns:
        clf (xgb.XGBClassifier or xgb.XGBRegressor): The fitted XGBoost model.
"""
def xgb_final_fitting(best_params: dict, 
                      X_train: pl.DataFrame, 
                      y_train: pl.DataFrame, 
                      X_valid: pl.DataFrame, 
                      y_valid: pl.DataFrame, 
                      metric: str,
                      low_lr: float,
                      early_stop: int,
                      n_classes: int=2,
                      fit_cv: int=1):
    """Fits the final XGB model with the best hyperparameters found in the optimization step."""

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    timer = Timer()
    params_fin = {}
    params_fin.update(get_xgb_base_params(metric, low_lr, n_classes))
    print(params_fin)

    params_fin.update(best_params)
    print(params_fin)
    np.random.seed(9234)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fitting                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    X = pl.concat([X_train, X_valid])
    y = pl.concat([y_train, y_valid])
    if get_train_type(metric) == "bin" or get_train_type(metric) == "multi":
        # Use Option 4 to find best num_boost_round
        if fit_cv:
            cv_clf = xgb.cv(params=params_fin, 
                            dtrain=xgb.DMatrix(X, y), 
                            early_stopping_rounds=early_stop, 
                            stratified=True, 
                            num_boost_round=10000, 
                            nfold=10,
                            seed=1241,
                            shuffle=True,
                            verbose_eval=100)
            clf = xgb.XGBClassifier(**params_fin, n_estimators=len(cv_clf))
            clf.fit(X, y, verbose=100)
        else:
            clf = xgb.XGBClassifier(**params_fin, early_stopping_rounds=early_stop, n_estimators=10000)
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
    else:
        if metric in ["q10", "q05", "q25", "q50", "q75", "q90", "q95"]: del params_fin["eval_metric"]
        if fit_cv:
            cv_clf = xgb.cv(params=params_fin, 
                            dtrain=xgb.DMatrix(X, y), 
                            early_stopping_rounds=early_stop, 
                            #stratified=True, 
                            num_boost_round=10000, 
                            nfold=10,
                            seed=1241,
                            shuffle=True,
                            verbose_eval=100)
            clf = xgb.XGBRegressor(**params_fin, n_estimators=10000)
            clf.fit(X, y, verbose=100)       
        else:
            clf = xgb.XGBRegressor(**params_fin, early_stopping_rounds=early_stop, n_estimators=10000)
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Final fitting ==============================')
    logging.info('Number of estimators ---------------------------')
    if not fit_cv:
        logging.info(f'best boosting round: {clf.best_iteration}')
    else:
        logging.info(len(cv_clf))
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())
    logging.info('Best params ---------------------------')
    logging.info(f'fixed learning rate: {params_fin["learning_rate"]}')

    logging.info("time taken {timer.get_elapsed()}")

    return(clf)   

from sklearn.linear_model import LogisticRegression
def elr_final_fitting(best_params: dict, 
                      X_train: pl.DataFrame, 
                      y_train: pl.DataFrame,
                      X_valid: pl.DataFrame,
                      y_valid: pl.DataFrame):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    timer = Timer()
    print(best_params)
    np.random.seed(9234)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fitting                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    X = pl.concat([X_train, X_valid])
    y = pl.concat([y_train, y_valid])
    enet = LogisticRegression(penalty="elasticnet", 
                              solver="saga",
                              l1_ratio=best_params["l1_ratio"],
                              C=1/best_params["alpha"],
                              max_iter=5000)
    enet.fit(X.to_numpy(), y.to_numpy().ravel())
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Final fitting ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())

    return(enet)   


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
def lr_fitting(X_train: pl.DataFrame, 
               y_train: pl.DataFrame,
               X_valid: pl.DataFrame,
               y_valid: pl.DataFrame):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Study setup                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    timer = Timer()
    np.random.seed(9234)
    X = pl.concat([X_train, X_valid])
    y = pl.concat([y_train, y_valid])
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fitting                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    lr = LogisticRegression(solver="saga")
    lr.fit(X.to_numpy(), y.to_numpy().ravel())


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Variance explained                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # predicted probabilities
    probs = lr.predict_proba(X.to_numpy())[:, 1]
    # log-likelihood of fitted model
    LL_model = -log_loss(y.to_numpy().ravel(), probs, normalize=False)
    
    # same for null model (model without PGS")
    null_model = LogisticRegression(solver="saga")
    null_model.fit(X.select("EVENT_AGE", "SEX").to_numpy(), y.to_numpy().ravel())
    null_probs = null_model.predict_proba(X.select("EVENT_AGE", "SEX").to_numpy())[:, 1]
    LL_null = -log_loss(y.to_numpy().ravel(), null_probs, normalize=False)

    # McFadden's pseudo-R2
    R2_McF = 1 - (LL_model / LL_null)
    print(f"McFadden pseudo-R2 on training set: {R2_McF:.4f}")

    logging.info("McFadden pseudo-R2 over baseline age+sex on training set:"+str(R2_McF))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Logging info                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    logging.info(timer.get_elapsed())
    # Get the best model
    logging.info('Final fitting ==============================')
    logging.info('Time ---------------------------')
    logging.info(timer.get_elapsed())

    return(lr)   

"""Quantile evaluation function for XGBoost models.
    Args:
        metric (str): The evaluation metric to use.

    Returns:
        Callable: A function that computes the evaluation metric.
"""
from model_eval_utils import get_score_func_based_on_metric
def quantile_eval(metric):
    def eval_metric(x, y):
        loss = get_score_func_based_on_metric(metric)(x, y)
        print(loss)
        return np.mean(loss)
    return eval_metric