import logging
import polars as pl
import xgboost as xgb
import logging
import sys

sys.path.append(("/home/ivm/valid/scripts/utils/"))
from model_eval_utils import get_train_type, get_optim_precision_recall_cutoff
from abnorm_utils import get_abnorm_func_based_on_name
from model_fit_utils import get_cont_goal_col_name

def get_out_data(data: pl.DataFrame, 
                 model_final: xgb.XGBClassifier, 
                 X_all: pl.DataFrame, 
                 y_all: pl.DataFrame, 
                 metric: str,
                 lab_name: str,
                 goal: str,
                 abnorm_extra_choice: str="",
                 optimal_proba_cutoff: float = None,
                 device: str=None) -> pl.DataFrame:
    """Creates the output data with the predictions of the model.
    Args:
        data (pl.DataFrame): The input data.
        model_final (xgb.XGBClassifier): The trained XGBoost model.
        X_all (pl.DataFrame): The input features for all data.
        y_all (pl.DataFrame): The true labels for all data.
        metric (str): The evaluation metric.
        goal (str): The goal column name.

        lab_name (str): The name of the lab. Only needed for the abnormality prediction based on the continuous value.
        abnorm_extra_choice (str): Additional abnormality choice (default is empty string). Only needed for the abnormality prediction based on the continuous value.

       Returns the relevant columns from the input data with the predictions of the model.
       Note that abnormality here is the case/control status.

       If available columns:
            - `FINNGENID`: Individual ID
            - `EVENT_AGE`: Age at event (for controls last lab record and for cases )
            - `LAST_VAL_DATE`: Prediction date (TODO need to check what exact time this is)
            - `SET`: Train, validation or test set
            - `N_PRIOR_ABNORMS`: Number of prior abnormal values
            - `VALUE` (if continuous prediction): Predicted value
            - `ABNORM_PROBS` (if binary prediction): Probability of abnormality
            - `ABNORM_PREDS` (if binary prediction): Predicted abnormality
            - `TRUE_VALUE` (if continuous prediction): True value
            - `TRUE_ABNORM` (if binary prediction): True abnormality
       Otherwise:
               - `FINNGENID`: Individual ID
               - `EVENT_AGE`: Age at event (for controls last lab record and for cases )
               - `SEX`: Sex                - `SET`: Train, validation or test set
               - `SET`: Train, validation or test set.
               - `goal`: True label for the prediction  
               - `ABNORM_PROBS` (if binary prediction): Probability of the positive class (abnormality for lab value prediction)
               - `ABNORM_PREDS` (if binary prediction): Predicted label for the abnormality based on the optimal cut-off
               - `TRUE_ABNORM` (if binary prediction): The goal column.
    """
    
    X_all = df_to_numpy_cuda(X_all, device)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Base data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    try:
        out_data = (data
                    .select("FINNGENID", "SEX", "y_MEAN", "y_MEAN_ABNORM", "y_MIN", "y_MIN_ABNORM", "y_NEXT", "y_NEXT_ABNORM", "EVENT_AGE", "SET", "LAST_VAL_DATE", "ABNORM")
                    .rename({"LAST_VAL_DATE": "DATE", "ABNORM": "N_PRIOR_ABNORMS"})
                )
    except:
        if "SET" not in data.columns:
            data = data.with_columns(pl.lit(3).alias("SET"))
        out_data = (data.select("FINNGENID", "EVENT_AGE",  "SEX", "SET", goal))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Continuous prediction                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
     
    ################# PROBABLY DEPRECATED #####################################
    if get_train_type(metric) == "cont":    
        y_pred = model_final.predict(X_all)
        out_data = out_data.with_columns([
                            pl.Series("ABNORM_PROBS", y_pred),
                            pl.Series("TRUE_VALUE", y_all),
        ])
        # Abnormality prediction based on the continuous value
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "ABNORM_PROBS").rename({"ABNORM_CUSTOM": "ABNORM_PREDS"})
        out_data = get_abnorm_func_based_on_name(lab_name, abnorm_extra_choice)(out_data, "TRUE_VALUE").rename({"ABNORM_CUSTOM": "TRUE_ABNORM"})
        print(out_data["TRUE_ABNORM"].value_counts())
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Binary prediction of "abnormality"                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    else:        
        # # # # # # # # # Continuous column # # # # # # # # # # # # # # # # # # 
        new_goal = get_cont_goal_col_name(goal, data.columns)
        if new_goal is not None:
            y_cont_all = data.select(new_goal)

            out_data = out_data.with_columns([
                pl.Series("TRUE_VALUE", y_cont_all),
            ])
        # # # # # # # # # Binary column # # # # # # # # # # # # # # # # # # 
        out_data = out_data.with_columns([pl.col(goal).alias("TRUE_ABNORM")])

        # # # # # # # # # Abnormality columns # # # # # # # # # # # # # # # # # # #
        if get_train_type(metric) == "bin":  
            y_pred = model_final.predict_proba(X_all)[:,1]
            out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS")])

            # Binary abnormality prediction based on optimal cut-off
            # If optimal_proba_cutoff is not provided, calculate it based on the training data
            # Otherwise use the provided optimal_proba_cutoff (e.g. from training data) for the final prediction data
            if optimal_proba_cutoff is None:
                optimal_proba_cutoff = get_optim_precision_recall_cutoff(out_data)
            logging.info(f"Optimal cut-off for prediction based on PR {optimal_proba_cutoff}")

            out_data = out_data.with_columns(
                (pl.col("ABNORM_PROBS") > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS")
            )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Multi-class prediction                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        elif get_train_type(metric) == "multi":  
            if out_data["TRUE_ABNORM"].unique().to_list() != [0]: # this should be dummy columns like final prediction data

                for pred_val in out_data["TRUE_ABNORM"].unique():
                    y_pred = model_final.predict_proba(X_all)[:,int(pred_val)]
                    out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS_"+str(pred_val))])

                    # Binary abnormality prediction based on optimal cut-off
                    temp_data = out_data.with_columns(pl.when((pl.col.TRUE_ABNORM!=1)&(pl.col.TRUE_ABNORM!=0)).then(pl.lit(0)).otherwise(pl.col.TRUE_ABNORM).alias("TRUE_ABNORM"))
                    temp_data = temp_data.rename({"ABNORM_PROBS_"+str(pred_val): "ABNORM_PROBS"})
                    if optimal_proba_cutoff is None:
                        optimal_proba_cutoff = get_optim_precision_recall_cutoff(temp_data)
                    logging.info(f"Optimal cut-off for {pred_val} prediction based on PR {optimal_proba_cutoff}")

                    out_data = out_data.with_columns((pl.col("ABNORM_PROBS_"+str(pred_val)) > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS_"+str(pred_val)))

                    # rename columns ABNORM_PROBS_1.0 to 1 and 0.0 to 0
                    out_data = out_data.rename({"ABNORM_PROBS_"+str(pred_val): "ABNORM_PROBS_"+str(int(pred_val)), "ABNORM_PREDS_"+str(pred_val): "ABNORM_PREDS_"+str(int(pred_val))})
            else:
                # this is the final prediction, which also means we want to use the training data optimal cut-off
                # in this case there is only one class in the data
                y_pred = model_final.predict_proba(X_all)[:,1]
                out_data = out_data.with_columns([pl.Series(y_pred).alias("ABNORM_PROBS")])
                out_data = out_data.with_columns(
                    (pl.col("ABNORM_PROBS") > optimal_proba_cutoff).cast(pl.Int64).alias("ABNORM_PREDS")
                )
    return(out_data, optimal_proba_cutoff)

import polars as pl
try:
    import cupy as cp
except:
    pass
def df_to_numpy_cuda(data: pl.DataFrame,
                     device):
    """Converts a Polars DataFrame to a NumPy array, and if device is "cuda", also converts it to a CuPy array.

    Args:
        data (pl.DataFrame): The input Polars DataFrame.
        device (str): The device to use ("cpu" or "cuda").

    Returns:
        np.ndarray or cp.ndarray: The converted array, either as a NumPy array (if device is "cpu") or a CuPy array (if device is "cuda").
    """
    no_cuda = True
    if device == "cuda":
        try:
            data = cp.asarray(data.to_numpy() if hasattr(data, "to_numpy") else data)
            no_cuda = False
        except:
            no_cuda = True

    if no_cuda:
        data = data.to_numpy() if hasattr(data, "to_numpy") else data
    return(data)

import polars as pl
import shap
import numpy as np
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from minor_plot_utils import get_plot_names
def get_shap_importances(X_in: pl.DataFrame,
                         explainer: shap.TreeExplainer,
                         lab_name: str,
                         lab_name_two: str = "",
                         translate: bool = True,
                         device: str = None,
                         batch_size: int = 1_000_000) -> tuple[pl.DataFrame, list]:
    """Calculates SHAP importances for the given input data and SHAP explainer.

    The output DataFrame contains the following columns:
        - `labels`: The feature names (translated if translate is True).
        - `orig`: The original feature names.
        - `mean_shap`: The mean absolute SHAP value for each feature.
        - `mean_signed_shap`: The mean signed SHAP value for each feature.
        - `mean_shap_when_1`: The mean SHAP value for samples where the feature is 1 (for binary features) or above the 75th percentile (for continuous features).
        - `mean_shap_when_0`: The mean SHAP value for samples where the feature is 0 (for binary features) or below the 25th percentile (for continuous features).
        - `n_1`: The number of samples where the feature is 1 (for binary features) or above the 75th percentile (for continuous features).
        - `n_0`: The number of samples where the feature is 0 (for binary features) or below the 25th percentile (for continuous features).
    
    Args:
        X_in (pl.DataFrame): The input features for which to calculate SHAP importances.
        explainer (shap.TreeExplainer): The SHAP explainer to use for calculating SHAP values.
        lab_name (str): The name of the lab. Used for translating feature names to more interpretable names.
        lab_name_two (str): An additional lab name for translation purposes. Default is an empty string.
        translate (bool): Whether to translate feature names to more interpretable names. Default is True.
        device (str): The device to use for calculations ("cpu" or "cuda"). Default is None, which means it will use "cpu".
        batch_size (int): The batch size to use for calculating SHAP values. Default is 1,000,000.

    Returns:
        tuple: A tuple containing:
            - pl.DataFrame: A Polars DataFrame with SHAP importances and related statistics for each feature.
            - list: A list of the new feature names after translation (if translate is True) or the original feature names (if translate is False).
    """
    orig_names = X_in.columns
    new_names = get_plot_names(orig_names, lab_name, lab_name_two) if translate else orig_names

    X_np = df_to_numpy_cuda(X_in, device)
    n_rows, n_feat = X_in.shape

    # Precompute quantiles + binary flags once
    is_binary = np.array([
        np.all((col == 0) | (col == 1) | np.isnan(col))
        for col in np.asarray(X_in).T  
    ])
    q25 = np.nanquantile(X_in, 0.25, axis=0)
    q75 = np.nanquantile(X_in, 0.75, axis=0)

    abs_sum = np.zeros(n_feat)
    signed_sum = np.zeros(n_feat)
    shap_sum_hi = np.zeros(n_feat)
    shap_sum_lo = np.zeros(n_feat)
    n_hi = np.zeros(n_feat, dtype=np.int64)
    n_lo = np.zeros(n_feat, dtype=np.int64)

    for i in range(0, n_rows, batch_size):
        X_chunk = X_np[i:i+batch_size]
        if lab_name == "tsh":
            # Multimodal prediction with binary classification for each class, taking high abnormal class SHAP values for importances
            shap_chunk = explainer.shap_values(X_chunk)[:,:,1]
        else:
            shap_chunk = explainer.shap_values(X_chunk)
        try:
            X_chunk_np = X_chunk.get()
        except AttributeError:
            X_chunk_np = X_chunk

        abs_sum += np.abs(shap_chunk).sum(0)
        signed_sum += shap_chunk.sum(0)

        for j in range(n_feat):
            vals = X_chunk_np[:, j]
            if is_binary[j]:
                mask_hi = vals == 1
                mask_lo = vals == 0
            else:
                mask_hi = vals >= q75[j]
                mask_lo = vals <= q25[j]
            shap_sum_hi[j] += shap_chunk[mask_hi, j].sum()
            shap_sum_lo[j] += shap_chunk[mask_lo, j].sum()
            n_hi[j] += mask_hi.sum()
            n_lo[j] += mask_lo.sum()

    with np.errstate(invalid="ignore"):
        mean_when_1 = shap_sum_hi / np.where(n_hi > 0, n_hi, np.nan)
        mean_when_0 = shap_sum_lo / np.where(n_lo > 0, n_lo, np.nan)

    direction_shaps = pl.DataFrame({
        "labels": new_names,
        "orig": orig_names,
        "mean_shap": abs_sum / n_rows,
        "mean_signed_shap": signed_sum / n_rows,
        "mean_shap_when_1": mean_when_1,
        "mean_shap_when_0": mean_when_0,
        "n_1": n_hi,
        "n_0": n_lo,
    }).sort("mean_shap", descending=True)

    return direction_shaps, new_names

import polars as pl
import logging
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from general_utils import make_dir, get_date
def save_importances(top_gain,
                     out_down_path: str,
                     study_name: str,
                     goal,
                     lab_name,
                     subset: str="all",
                     n_features: int=None,
                     cv: bool=False) -> None:
    """Saves the feature importances of the model to a csv file.

    Args:
        top_gain (pl.DataFrame): The dataframe containing the feature importances.
        out_down_path (str): The output directory path.
        subset (str): The subset of the data (default is "all").

    Returns:
        None"""
    crnt_out_down_path = out_down_path+"/"+goal+"/"; make_dir(crnt_out_down_path)
    if n_features is None:
        if cv:
            file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_cv_" + subset + "_" + get_date() + ".csv"
        else:
            file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_" + subset + "_" + get_date() + ".csv"
    else:
        file_path = crnt_out_down_path + lab_name+"_"+study_name+"_"+goal+"_shap_importance_fs"+str(n_features)+"_" + subset + "_" + get_date() + ".csv"
    try:
        logging.info(top_gain.head(10))
        top_gain.write_csv(file_path)
    except:
        top_gain = pl.DataFrame(top_gain).with_columns(pl.col(top_gain.columns[0]).arr.to_struct().alias(top_gain.columns[0])).unnest(top_gain.columns[0])
        logging.info(top_gain.head(10))
        top_gain.write_csv(file_path)


import polars as pl
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Tuple
def create_xgb_dts(data: pl.DataFrame, 
                   X_cols: list, 
                   y_goal: str,
                   train_pct: int = 1,
                   val_data: pl.DataFrame = None,
                   final_fit: bool=False) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, xgb.DMatrix, xgb.DMatrix, StandardScaler]:
    """Creates the XGBoost data matrices 

    Returns the data and the predictors.
    
        Args:
            data (pl.DataFrame): The input data.
            X_cols (list): The list of feature columns.
            y_goal (str): The target column.
            reweight (int): The reweighting factor.

        Returns:
            tuple: A tuple containing the data (polars DataFrame) and the list of predictors
    """
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Replace -1s with 2s                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # For i.e. multi-class like TSH, we want to keep -1 as a separate class, but for binary classification we want to replace it with 2 
    # (so that it is not treated as missing value but as a separate class).
    if -1 in data.get_column(y_goal):
        data = data.with_columns(pl.when(pl.col(y_goal)==-1).then(pl.lit(2)).otherwise(pl.col(y_goal)).alias(y_goal))
    if -1 in val_data.get_column(y_goal):
        val_data = val_data.with_columns(pl.when(pl.col(y_goal)==-1).then(pl.lit(2)).otherwise(pl.col(y_goal)).alias(y_goal))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Fix missing columns                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    # If val data is not exactly the same as data
    if val_data is not None:
        for crnt_col in X_cols:
            if crnt_col not in data.columns:
                val_data = val_data.with_columns(pl.Series(crnt_col, [0]*val_data.height))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Memory saving                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    binary_cols = ([col_name for col_name in X_cols
                       if(set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {0, 1} or
                          set(data.select(pl.col(col_name).drop_nulls().unique()).to_series()) <= {"0", "1"})  
    ])
    numeric_cols = [col_name for col_name in X_cols if col_name not in binary_cols]

    data = data.with_columns([
        pl.col(col).cast(pl.Float64, strict=False) for col, dtype in data.schema.items() if (isinstance(dtype, pl.Int64) or isinstance(dtype, pl.Int32)) and col in numeric_cols
    ])
    data = data.with_columns([
        pl.col(col).fill_nan(None).cast(pl.Int8, strict=False) for col in data.columns if col in binary_cols
    ])
    X_cols = numeric_cols + binary_cols


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Split data                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    # # # # # # # # ALL # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    X_all = data.select(X_cols); y_all = data.select(y_goal)
    if val_data is not None:
        X_val_all = val_data.select(X_cols); y_val_all = val_data.select(y_goal)
    else:
        X_val_all = None; y_val_all = None

    # # # # # # # # TRAIN # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    if not final_fit:
        train_data = data.filter(pl.col("SET")==0).drop("SET")
        finetune_valid_data = data.filter(pl.col("SET")==0.5).drop("SET")
    # Training on validation and test set for final fit. 
    else:
        train_data = data.filter(pl.col("SET").is_in([1,2])).drop("SET")
        data = data.filter(pl.col("SET").is_in([1,2])).with_columns(pl.Series("SET", [0]*data.height))
        finetune_valid_data = pl.DataFrame({col: [] for col in X_cols}); y_finetune_valid = pl.DataFrame({y_goal: []})

    # if training pct<100, take only part of training data
    if train_pct<100:
        train_data = train_data.sample(fraction=train_pct/100, with_replacement=False, seed=42)
        data = data.filter(pl.col("FINNGENID").is_in(train_data["FINNGENID"]).or_(pl.col("SET")!=0))
            
        if finetune_valid_data.height>0:
            finetune_valid_data = finetune_valid_data.sample(fraction=train_pct/100, with_replacement=False, seed=42)
            data = data.filter(pl.col("FINNGENID").is_in(finetune_valid_data["FINNGENID"]).or_(pl.col("SET")!=0.5))
            
    if finetune_valid_data.height>0:
        X_finetune_valid = finetune_valid_data.select(X_cols); y_finetune_valid = finetune_valid_data.select(y_goal)
    else:
        X_finetune_valid = pl.DataFrame({col: [] for col in X_cols}); y_finetune_valid = pl.DataFrame({y_goal: []})

    X_train = train_data.select(X_cols); y_train = train_data.select(y_goal)

    # # # # # # # # VALID & TEST # # # # # # # # # # # # # # # # # # # # # # # # 
    if val_data is None and not final_fit:
        valid_data = data.filter(pl.col("SET")==1).drop("SET")
        test_data = data.filter(pl.col("SET")==2).drop("SET")
            
        X_valid = valid_data.select(X_cols); y_valid = valid_data.select(y_goal)
        X_test = test_data.select(X_cols); y_test = test_data.select(y_goal)
    elif val_data is not None and final_fit:
        valid_data = val_data
        X_valid = valid_data.select(X_cols); y_valid = valid_data.select(y_goal)
        X_test = pl.DataFrame({col: [] for col in X_cols}); y_test = pl.DataFrame({y_goal: []})
    elif val_data is not None and not final_fit:
        valid_data = val_data.filter(pl.col("SET")==1).drop("SET")
        test_data = val_data.filter(pl.col("SET")==2).drop("SET")

        X_valid = valid_data.select(X_cols); y_valid = valid_data.select(y_goal)
        X_test = test_data.select(X_cols); y_test = test_data.select(y_goal)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 XGBoost datatype                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #   
    dtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=True)
    if finetune_valid_data.height>0:
        dfinetunevalid = xgb.DMatrix(data=X_finetune_valid, label=y_finetune_valid, enable_categorical=True)
    else:
        dfinetunevalid = None
    return(data["FINNGENID"], X_train, y_train, X_finetune_valid, y_finetune_valid, X_valid, y_valid, X_test, y_test, X_all, y_all, X_val_all, y_val_all, dtrain, dfinetunevalid, dvalid, data, val_data)

