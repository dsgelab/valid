import sys
sys.path.append("../../utils/")
from general_utils import *

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#              Helpers                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
def fix_tsh(past_preds):
    if "ABNORM_PROBS_1" in past_preds.columns:
        past_preds = past_preds.rename({"ABNORM_PROBS_1": "ABNORM_PROBS"})
        past_preds = past_preds.with_columns(pl.when(pl.col.y_MEAN_ABNORM==1).then(1).otherwise(0).alias("y_MEAN_ABNORM"))
    return(past_preds)

from labeling_utils import get_bbs_indvs
import polars as pl
def add_bb_info(preds, filter_to_bb=True):
    turku_fids = get_bbs_indvs(fg_ver="R13", bbs=["AURIA BIOBANK"]) 
    tampere_fids = get_bbs_indvs(fg_ver="R13", bbs=["TAMPERE BIOBANK"]) 
    preds = preds.with_columns(pl.when(pl.col.FINNGENID.is_in(tampere_fids))
                                         .then(pl.lit("Tampere"))
                                         .when(pl.col.FINNGENID.is_in(turku_fids))
                                         .then(pl.lit("Turku")).otherwise(None).alias("BB"))
    preds = preds.filter(pl.col.SET!=0,~pl.col.BB.is_null())
    return(preds)

import polars as pl
def add_cut_groups(preds, 
                   quart_cuts, 
                   min_prob, 
                   max_prob):
    labels = [str(label) for label in range(1, len(quart_cuts)+2)]
    preds = preds.with_columns(pl.col.ABNORM_PROBS.cut(quart_cuts, labels=labels, left_closed=True).cast(pl.Int32).alias("GROUP"))
    # Transform cuts to abnorm probability 
    quart_cuts_df = pl.DataFrame({"CUTS": [min_prob]+quart_cuts+[max_prob]})
    quart_cuts_df = quart_cuts_df.with_columns((pl.col.CUTS.shift(1)+(pl.col.CUTS-pl.col.CUTS.shift(1))/2).round(3).alias("MID"),
                                                pl.Series("GROUP", ["0"]+labels).cast(pl.Int32),
                                                pl.Series("QUANT", [min_prob]+quart_cuts+[max_prob]).cast(pl.Float32))
    quart_cuts_df = quart_cuts_df.filter(pl.col.GROUP != 0.0)
    preds = preds.join(quart_cuts_df, on="GROUP", how="left")
    return preds, quart_cuts_df

import polars as pl
def get_max_n(future_preds_path, 
              quart_cuts, 
              min_prob,
              max_prob,
              bb_specific=False,
              min_age=30):
    future_preds = pl.read_parquet(future_preds_path).filter(pl.col.EVENT_AGE>=min_age)
    future_preds = add_bb_info(future_preds)
    future_preds = future_preds.filter(pl.col.ABNORM_PROBS<=max_prob)
    
    # Cuts
    ########### ############# ############## ############## ###########
    future_preds, _ =  add_cut_groups(future_preds, quart_cuts, min_prob, max_prob)
        
    # Stats
    group_stats = (future_preds
                     .group_by("GROUP")
                     .agg(pl.len().alias("count").cast(pl.Int32))
                     .sort("GROUP")
                  )
    bb_group_stats = (future_preds
                     .group_by("GROUP", "BB")
                     .agg(pl.len().alias("count"))
                     .sort("GROUP", "BB")
                  )
    
    if not bb_specific: max_n = group_stats.sort("GROUP")["count"]
    else: max_n = bb_group_stats.group_by("GROUP").agg(pl.col.count.min()).sort("GROUP")["count"]
    
    return(max_n)

def check_sample_sizes(quart_cuts,
                       max_n,
                       last_possible_sample_size=0,
                       max_cut=0,
                       vocal=False):
    
    for idx, crnt_cut  in enumerate(quart_cuts):
        if idx >= len(max_n):
            if vocal: print(f"Ran out of groups as {idx} - cut at {quart_cuts[idx]}")
            return(quart_cuts[:idx], last_possible_sample_size, max_cut)
        if max_n[idx] < 10:
            if vocal: print(f"Removing current group {idx+1} - cut at {quart_cuts[idx]} with only {max_n[idx]} possible samples")
            if last_possible_sample_size != max_n[idx]:
                last_possible_sample_size = max_n[idx]
                max_cut = quart_cuts[idx]
            new_quart_cuts = [crnt_cut for crnt_cut in quart_cuts if crnt_cut!=quart_cuts[idx]]

            return(new_quart_cuts, last_possible_sample_size, max_cut)
    return(quart_cuts, last_possible_sample_size, max_cut)
    
def reduce_sample_sizes(sample_sizes, max_n):
    reduce_idxs = list()
    rest_idxs = list()
    for idx, crnt_sample_size  in sample_sizes.items():
        if int(round(max_n[int(idx-1)]/2)*2)<sample_sizes[idx]:
            reduce_idxs.append(idx)
        else:
            rest_idxs.append(idx)
        sample_sizes[idx] = min(sample_sizes[idx], int(round(max_n[int(idx-1)]/2)*2))

    return(sample_sizes, reduce_idxs, rest_idxs)

import polars as pl
def sample_other_select_fids(other_preds,
                             lab_name,
                             all_quart_cuts,
                             all_sample_sizes,
                             min_probs):
    other_select_fids = list()
    for other_lab_name in other_preds:
        if other_lab_name != lab_name:
            if len(other_select_fids) == 0:
                crnt_other_preds =  other_preds[other_lab_name].filter(~pl.col.FINNGENID.is_in(other_select_fids))
            else:
                crnt_other_preds = other_preds[other_lab_name]
            crnt_other_preds, _  = add_cut_groups(crnt_other_preds,
                                                  all_quart_cuts[other_lab_name][:-1], 
                                                  min_probs[other_lab_name],
                                                  all_quart_cuts[other_lab_name][-1])
            crnt_select_sample_sizes = (
                        pl.DataFrame({"GROUP": [y for x in all_sample_sizes[other_lab_name].keys() for y in (x,)*2], 
                                      "BB": ["Tampere", "Turku"]*len(all_sample_sizes[other_lab_name]),
                                      "N": [y/2 for x in all_sample_sizes[other_lab_name].values() for y in (x,)*2],
                                     }
                ))
            other_group_selects = (
                            crnt_other_preds
                            .filter(pl.col.ABNORM_PROBS>min_probs[other_lab_name])
                            .join(crnt_select_sample_sizes, on=["GROUP", "BB"])
                            .group_by('GROUP', "BB")
                            .map_groups(lambda x: x.sample(n=min(x["N"][0], x.height)))
                    )
            [other_select_fids.append(crnt_fid) for crnt_fid in crnt_other_preds["FINNGENID"] if crnt_fid in other_group_selects["FINNGENID"]]
    return(other_select_fids)

import polars as pl
def create_bb_sample_size_df(sample_sizes,
                             max_n):
    if max_n is not None:
        return (pl.DataFrame({"GROUP": [y for x in sample_sizes.keys() for y in (x,)*2], 
                          "BB": ["Tampere", "Turku"]*len(sample_sizes),
                          "N": [y/2 for x in sample_sizes.values() for y in (x,)*2],
                          "MAX_N": [y for x in max_n for y in (x,)*2]
                         }))
    else:
        return (pl.DataFrame({"GROUP": [y for x in sample_sizes.keys() for y in (x,)*2], 
                          "BB": ["Tampere", "Turku"]*len(sample_sizes),
                          "N": [y/2 for x in sample_sizes.values() for y in (x,)*2]
                         }))
        
from collections import defaultdict
def collapse_sample_sizes(collapsed_quart_cuts,
                          quart_cuts,
                          sample_sizes,
                          min_prob):
    collapsed_sample_sizes = defaultdict(int)
    for idx, crnt_quart_cut in enumerate(collapsed_quart_cuts):
        for idx_two in range(0, len(quart_cuts)):
            crnt_lower = min_prob
            if idx == 0:
                crnt_lower = min_prob
            else:
                crnt_lower = collapsed_quart_cuts[idx-1]
                
            if (quart_cuts[idx_two] > crnt_lower) and (quart_cuts[idx_two] <= crnt_quart_cut):
                collapsed_sample_sizes[idx+1] += sample_sizes[idx_two+1]
                
    return(collapsed_sample_sizes)
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#              Main                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, log_loss
import matplotlib.pyplot as plt
from collections import defaultdict
def draw_now(past_preds_paths,
             future_preds_paths,
             all_quart_cuts,
             all_sample_sizes, 
             min_probs,
             part_pct,
             lab_name,
             n_boots,
             min_age=30,
             save_files=False,
             lab_colors = {"HbA1c": "#29557b", "eGFR": "#d5694f", "TSH": "#b9ada8", "LDL": "#eab034"},
             metric_functions = {"AUC": roc_auc_score, "AvgPrec": average_precision_score, "Brier": brier_score_loss, "logloss": log_loss},
             base_path="/home/ivm/valid/data/processed_data/kanta_R14/model_evals/selections/"+get_date()+"/"):

    make_dir(base_path)
    past_preds = fix_tsh(pl.read_parquet(past_preds_paths[lab_name])).filter(pl.col.EVENT_AGE>=min_age)
    past_preds = add_bb_info(past_preds)
    
    ########### ############# ############## ############## ###########
    quart_cuts = all_quart_cuts[lab_name]
    max_prob = quart_cuts[-1]
    min_prob = min_probs[lab_name]
    sample_sizes = all_sample_sizes[lab_name]

    ########### ############# ############## ############## ###########
    past_preds = past_preds.filter(pl.col.ABNORM_PROBS<=max_prob)
    quart_cuts = quart_cuts[:-1]

    ########### ############# ############## ############## ###########
    other_preds = dict()
    for other_lab_name in past_preds_paths:
        if other_lab_name != lab_name:
            crnt_other_preds =  fix_tsh(pl.read_parquet(past_preds_paths[other_lab_name])).filter(pl.col.EVENT_AGE>=min_age)
            crnt_other_preds = add_bb_info(crnt_other_preds)
            other_preds[other_lab_name] = crnt_other_preds

    # Cuts
    ########### ############# ############## ############## ###########
    past_preds, quart_cuts_df = add_cut_groups(past_preds,
                                               quart_cuts, 
                                               min_prob,
                                               max_prob)

    # Max possible
    ########### ############# ############## ############## ###########
    max_n = get_max_n(future_preds_paths[lab_name], 
                      quart_cuts, 
                      min_prob,
                      max_prob,
                      bb_specific=True)

    # Group selection
    ########### ############# ############## ############## ###########
    pcts = {round(crnt_mid, 3): list() for crnt_mid in quart_cuts_df["MID"]}
    all_value_stats = {round(crnt_mid, 3): {"median": list(), "q1": list(), "q3": list(), "whis_low": list(), "whis_high": list()} for crnt_mid in quart_cuts_df["MID"]}
    all_age_stats = {round(crnt_mid, 3): {"median": list(), "q1": list(), "q3": list(), "whis_low": list(), "whis_high": list()} for crnt_mid in quart_cuts_df["MID"]}
    n_abnorms = {round(crnt_mid, 3): list() for crnt_mid in quart_cuts_df["MID"]}
    ns ={round(crnt_mid, 3): list() for crnt_mid in quart_cuts_df["MID"]}
    sub_metrics = defaultdict(list)
    sub_part_metrics = defaultdict(list)
    sub_group_metrics = defaultdict(list)
    
    for n_boot in range(n_boots):
        select_sample_sizes = create_bb_sample_size_df(sample_sizes, max_n)
        # Do selections
        ########### ############# ############## ############## ###########
        group_selects = (
                past_preds
              #  .filter(pl.col.ABNORM_PROBS)
                .join(select_sample_sizes, on=["GROUP", "BB"])
                .group_by('GROUP', "BB")
                .map_groups(lambda x: x.sample(n=min(x["N"][0], x.height)))
        )

        select_preds = past_preds.filter(pl.col.FINNGENID.is_in(group_selects["FINNGENID"]) )
        # Adding participation rate 
        select_sub_preds = select_preds.filter(pl.col.FINNGENID.is_in(select_preds.sample(n=select_preds.height*0.25, shuffle=True)["FINNGENID"]))

        # Stats
        ########### ############# ############## ############## ###########
        crnt_sub_stats = (select_sub_preds
                          .group_by("GROUP", "CUTS", "MID")
                          .agg(pl.len().alias("N"),
                              (pl.col.y_MEAN_ABNORM).sum().alias("N_ABNORM"))
                          .with_columns((pl.col.N_ABNORM/pl.col.N).alias("PCT")).sort(pl.col.GROUP)
                     )   
        crnt_stats = (select_preds
                          .group_by("GROUP", "CUTS", "MID")
                          .agg(pl.len().alias("N"),
                              (pl.col.y_MEAN_ABNORM).sum().alias("N_ABNORM"))
                          .with_columns((pl.col.N_ABNORM/pl.col.N).alias("PCT")).sort(pl.col.GROUP)
                     )  
        
        # Gathering info
        ########### ############# ############## ############## ###########
        for crnt_mid in crnt_stats["MID"]:
            ns[round(crnt_mid,3)].append(select_preds.filter(pl.col.MID==crnt_mid).height)
            if select_sub_preds.filter(pl.col.MID==crnt_mid).height>0:
                pcts[round(crnt_mid,3)].append(crnt_sub_stats.filter(pl.col.MID==crnt_mid).get_column("PCT")[0])
                n_abnorms[round(crnt_mid,3)].append(crnt_sub_stats.filter(pl.col.MID==crnt_mid).get_column("N_ABNORM")[0])     
                
                all_value_stats[round(crnt_mid,3)]["median"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["y_MEAN"].median())
                all_value_stats[round(crnt_mid,3)]["q1"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["y_MEAN"].quantile(0.25))
                all_value_stats[round(crnt_mid,3)]["q3"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["y_MEAN"].quantile(0.75))
                all_value_stats[round(crnt_mid,3)]["whis_low"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["y_MEAN"].quantile(0.025))
                all_value_stats[round(crnt_mid,3)]["whis_high"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["y_MEAN"].quantile(0.975))

                all_age_stats[round(crnt_mid,3)]["median"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["EVENT_AGE"].median())
                all_age_stats[round(crnt_mid,3)]["q1"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["EVENT_AGE"].quantile(0.25))
                all_age_stats[round(crnt_mid,3)]["q3"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["EVENT_AGE"].quantile(0.75))
                all_age_stats[round(crnt_mid,3)]["whis_low"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["EVENT_AGE"].quantile(0.025))
                all_age_stats[round(crnt_mid,3)]["whis_high"].append(select_sub_preds.filter(pl.col.MID==crnt_mid)["EVENT_AGE"].quantile(0.975))
            else:
                pcts[round(crnt_mid,3)].append(0)
                n_abnorms[round(crnt_mid,3)].append(0)        
                
        # Calculating metrics
        ########### ############# ############## ############## ###########
        for crnt_metric in metric_functions:
            sub_metrics[crnt_metric].append(metric_functions[crnt_metric](select_preds["y_MEAN_ABNORM"], select_preds["ABNORM_PROBS"]))
            sub_part_metrics[crnt_metric].append(metric_functions[crnt_metric](select_sub_preds["y_MEAN_ABNORM"], select_sub_preds["ABNORM_PROBS"]))
            sub_group_metrics[crnt_metric].append(metric_functions[crnt_metric](select_sub_preds["y_MEAN_ABNORM"], select_sub_preds["MID"]))

    for crnt_mid in crnt_stats["MID"]:
        all_value_stats[round(crnt_mid,3)]["median"] = np.asarray(all_value_stats[round(crnt_mid,3)]["median"]).mean()
        all_value_stats[round(crnt_mid,3)]["q1"] = np.asarray(all_value_stats[round(crnt_mid,3)]["q1"]).mean()
        all_value_stats[round(crnt_mid,3)]["q3"] = np.asarray(all_value_stats[round(crnt_mid,3)]["q3"]).mean()
        all_value_stats[round(crnt_mid,3)]["whis_low"] = np.asarray(all_value_stats[round(crnt_mid,3)]["whis_low"]).mean()
        all_value_stats[round(crnt_mid,3)]["whis_high"] = np.asarray(all_value_stats[round(crnt_mid,3)]["whis_high"]).mean()

        all_age_stats[round(crnt_mid,3)]["median"] = np.asarray(all_age_stats[round(crnt_mid,3)]["median"]).mean()
        all_age_stats[round(crnt_mid,3)]["q1"] = np.asarray(all_age_stats[round(crnt_mid,3)]["q1"]).mean()
        all_age_stats[round(crnt_mid,3)]["q3"] = np.asarray(all_age_stats[round(crnt_mid,3)]["q3"]).mean()
        all_age_stats[round(crnt_mid,3)]["whis_low"] = np.asarray(all_age_stats[round(crnt_mid,3)]["whis_low"]).mean()
        all_age_stats[round(crnt_mid,3)]["whis_high"] = np.asarray(all_age_stats[round(crnt_mid,3)]["whis_high"]).mean()

    # AUCs
    ########### ############# ############## ############## ###########   
    metric_df = pl.DataFrame({ "TYPE": ["All", "Binned", "Sampled", "Sampled&Participation", "Sampled&Participation&Binned"],
                               "PARTICIPATION": [part_pct]*5})
    for crnt_metric in metric_functions:
        true_metric = metric_functions[crnt_metric](past_preds["y_MEAN_ABNORM"], past_preds["ABNORM_PROBS"])
        group_metric = metric_functions[crnt_metric](past_preds["y_MEAN_ABNORM"], past_preds["MID"])
        
        print(f"True {crnt_metric}:                               {true_metric:.3f}")
        print(f"Group {crnt_metric}:                              {group_metric:.3f}")
        
        print("")
        sub_metrics[crnt_metric] = np.sort(np.asarray(sub_metrics[crnt_metric]))
        sub_part_metrics[crnt_metric] = np.sort(np.asarray(sub_part_metrics[crnt_metric]))
        sub_group_metrics[crnt_metric] = np.sort(np.asarray(sub_group_metrics[crnt_metric]))
        
        sub_boot_stats = [sub_metrics[crnt_metric].mean(),  sub_metrics[crnt_metric][int(0.025*len(sub_metrics[crnt_metric]))], sub_metrics[crnt_metric][int(0.975*len(sub_metrics[crnt_metric]))]]
        sub_part_boot_stats = [sub_part_metrics[crnt_metric].mean(),  sub_part_metrics[crnt_metric][int(0.025*len(sub_part_metrics[crnt_metric]))], sub_part_metrics[crnt_metric][int(0.975*len(sub_part_metrics[crnt_metric]))]]
        sub_part_group_boot_stats = [sub_group_metrics[crnt_metric].mean(),  sub_group_metrics[crnt_metric][int(0.025*len(sub_group_metrics[crnt_metric]))], sub_group_metrics[crnt_metric][int(0.975*len(sub_group_metrics[crnt_metric]))]]
    
        print(f"Subsample {crnt_metric}:                          {sub_boot_stats[0]:.3f} 95% CI: {sub_boot_stats[1]:.3f}-{sub_boot_stats[2]:.3f}")
        print(f"Subsample&Participation {crnt_metric}:            {sub_part_boot_stats[0]:.3f} 95% CI: {sub_part_boot_stats[1]:.3f}-{sub_part_boot_stats[2]:.3f}")
        print(f"Subsample&Participation&Group {crnt_metric}:      {sub_part_group_boot_stats[0]:.3f} 95% CI: {sub_part_group_boot_stats[1]:.3f}-{sub_part_group_boot_stats[2]:.3f}")
        
        print("")
        metric_df = metric_df.with_columns(pl.Series(f"{crnt_metric}_MEAN", [true_metric, group_metric, sub_boot_stats[0], sub_part_boot_stats[0], sub_part_group_boot_stats[0]]),
                                           pl.Series(f"{crnt_metric}_CIneg", [None, None, sub_boot_stats[1], sub_part_boot_stats[1], sub_part_group_boot_stats[1]]),
                                           pl.Series(f"{crnt_metric}_CIpos", [None, None, sub_boot_stats[2], sub_part_boot_stats[2], sub_part_group_boot_stats[2]]),
                                          )

    # Plotting
    ########### ############# ############## ############## ###########
    plt_data = pl.DataFrame({"GROUP": sample_sizes.keys(),
                             "CUT": all_quart_cuts[lab_name],
                             "MID": pcts.keys(),
                             "MIN_SELECTED_N": np.asarray([np.asarray(ns[crnt_mid]).min() for crnt_mid in ns.keys()]),
                             "MEAN_SELECTED_N": np.asarray([np.asarray(ns[crnt_mid]).mean() for crnt_mid in ns.keys()]),
                             "ORIGINAL_PLANNED_N": sample_sizes.values(),
                             "MAX_POSSIBLE_N": max_n*2,
                             "MEAN_PCT_ABNORMAL": np.asarray([np.asarray(pcts[crnt_mid]).mean() for crnt_mid in pcts.keys()]),
                             "CIneg_PCT_ABNORMAL": np.asarray([np.sort(np.asarray(pcts[crnt_mid]))[int(0.025*len(np.sort(np.asarray(pcts[crnt_mid]))))] for crnt_mid in pcts.keys()]),
                             "CIpos_PCT_ABNORMAL": np.asarray([np.sort(np.asarray(pcts[crnt_mid]))[int(0.975*len(np.sort(np.asarray(pcts[crnt_mid]))))] for crnt_mid in pcts.keys()]),
                             
                             "MEAN_MEDIAN_VAL": [all_value_stats[crnt_mid]["median"] for crnt_mid in all_value_stats],
                             "MEAN_Q1_VAL":[all_value_stats[crnt_mid]["q1"] for crnt_mid in all_value_stats],
                             "MEAN_Q3_VAL": [all_value_stats[crnt_mid]["q3"] for crnt_mid in all_value_stats],
                             "MEAN_WHIS_LOW_VAL": [all_value_stats[crnt_mid]["whis_low"] for crnt_mid in all_value_stats],
                             "MEAN_WHIS_HIGH_VAL": [all_value_stats[crnt_mid]["whis_high"] for crnt_mid in all_value_stats],

                             "MEAN_MEDIAN_AGE": [all_age_stats[crnt_mid]["median"] for crnt_mid in all_age_stats],
                             "MEAN_Q1_AGE":[all_age_stats[crnt_mid]["q1"] for crnt_mid in all_age_stats],
                             "MEAN_Q3_AGE": [all_age_stats[crnt_mid]["q3"] for crnt_mid in all_age_stats],
                             "MEAN_WHIS_LOW_AGE": [all_age_stats[crnt_mid]["whis_low"] for crnt_mid in all_age_stats],
                             "MEAN_WHIS_HIGH_AGE": [all_age_stats[crnt_mid]["whis_high"] for crnt_mid in all_age_stats]
    }).with_columns((pl.col.MEAN_PCT_ABNORMAL-pl.col.CIneg_PCT_ABNORMAL).alias("CIneg_PCT_ABNORMAL"), 
                    pl.max_horizontal((pl.col.CIpos_PCT_ABNORMAL-pl.col.MEAN_PCT_ABNORMAL),0).alias("CIpos_PCT_ABNORMAL"),
                    (pl.col("MEAN_SELECTED_N")/pl.col("MAX_POSSIBLE_N")).alias("PCT_OF_MAX_POSSIBLE_N_SELECTED")
                   )
    
    fig, axes = plt.subplots(5, 1, figsize=(17, 20))
    df_new = plot_distribution(axes[0], past_preds, quart_cuts, lab_colors[lab_name], lab_name, max_prob)
    plot_calibration(axes[1], plt_data, max_prob, lab_name, lab_colors[lab_name])
    plot_manual_boxplots(axes[2], all_age_stats, max_prob, lab_colors[lab_name], "Baseline age")
    plot_manual_boxplots(axes[3], all_value_stats, max_prob, lab_colors[lab_name], "Mean value")
    plot_barplot(axes[4], plt_data, max_prob, lab_colors[lab_name])
    display(plt_data.filter(pl.col("PCT_OF_MAX_POSSIBLE_N_SELECTED")>0.5))

    plt.tight_layout()


    # Saving
    ########### ############# ############## ############## ###########
    if save_files:
        df_new.write_csv(base_path+lab_name+"_probs_distribution_part"+str(int(round(part_pct*100)))+"pct_"+get_date()+".csv")
        plt_data.write_csv(base_path+lab_name+"_selection_stats_part"+str(int(round(part_pct*100)))+"pct_"+get_date()+".csv")
        metric_df.write_csv(base_path+lab_name+"_metric_stats_part"+str(int(round(part_pct*100)))+"pct_"+get_date()+".csv")
        plt.savefig(base_path+lab_name+"_overview_part"+str(int(round(part_pct*100)))+"pct_"+get_date()+".pdf")
    plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#              Plotting                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
import numpy as np
from major_plot_utils import round_column_min5
import seaborn as sns
def plot_distribution(ax,
                      preds,
                      quart_cuts,
                      lab_color,
                     lab_name,
                     max_prob):
    newcol_x, min_val_x, max_val_x = round_column_min5(preds["ABNORM_PROBS"]*100)
    newcol_x = newcol_x/100
    
    df_new = ((newcol_x).value_counts())
    sns.lineplot(df_new, x="ABNORM_PROBS", y="count",  ax=ax, color=lab_color)
    ax.fill_between(df_new.sort(pl.col.ABNORM_PROBS)["ABNORM_PROBS"], df_new.sort(pl.col.ABNORM_PROBS)["count"], alpha=0.4, color=lab_color)
    for idx, threshold in enumerate(quart_cuts):
        ax.axvline(threshold, linestyle='--', linewidth=1.5, label=f'{threshold:.1f}')
    ax.set_xlim(0,max_prob)
    ax.set_xlabel('Predicted proportion')
    ax.set_ylabel('Number of individuals (>4)')
    ax.set_title(lab_name)
    sns.despine()
    return(df_new)

def plot_calibration(ax,
                     plt_data,
                     max_prob,
                     lab_name,color):
    
    x = plt_data["MID"].to_numpy()
    y = plt_data["MEAN_PCT_ABNORMAL"].to_numpy()
    ci_neg = plt_data["CIneg_PCT_ABNORMAL"].to_numpy()
    ci_pos = plt_data["CIpos_PCT_ABNORMAL"].to_numpy()
    
    # Perfect calibration diagonal
    ax.plot([0, max_prob], 
            [0, max_prob], 
            linestyle="--", 
            color="gray", 
            linewidth=1,
            alpha=0.5, 
            label="Perfect calibration")
    
    # Plot each bin
    for idx in range(len(x)):
        ax.errorbar(
            np.asarray(x[idx]), np.asarray(y[idx]), 
            yerr=np.asarray([max(ci_neg[idx],0.01),ci_pos[idx]]).reshape((2,1)),
            fmt="o",
            color=color,
            elinewidth=2,
            capsize=5,
            capthick=1.5,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )
    
    # Axes
    ax.set_xlim(0, max_prob)
    ax.set_ylim(0, max_prob)
    ax.set_xlabel("Predicted proportion", fontsize=11)
    ax.set_ylabel("Observed proportion", fontsize=11)
    ax.set_title(f"{lab_name} calibration curve · {len(x)-1}-bin design", fontsize=12, fontweight="500")

import matplotlib.patches as mpatches
def plot_manual_boxplots(ax,
                         all_value_stats,
                         max_prob,
                         color,
                         name,
                         box_width=0.005):
    for x, stats in all_value_stats.items():
        # IQR box
        box = mpatches.FancyBboxPatch((x - (box_width/2), stats["q1"]), box_width, stats["q3"] - stats["q1"],
                                       boxstyle="square,pad=0", linewidth=1.5,
                                       edgecolor="black", facecolor=color, alpha=0.6)
        ax.add_patch(box)
        # Median line
        ax.plot([x - (box_width/2), x + (box_width/2)], [stats["median"], stats["median"]], color="black", lw=1.5)
        # Whiskers
        ax.plot([x, x], [stats["whis_low"], stats["q1"]], color="black", lw=1)
        ax.plot([x, x], [stats["q3"], stats["whis_high"]], color="black", lw=1)
        ax.plot([x - (box_width/4), x + (box_width/4)], [stats["whis_low"], stats["whis_low"]], color="black", lw=1)
        ax.plot([x -(box_width/4), x + (box_width/4)], [stats["whis_high"], stats["whis_high"]], color="black", lw=1)
        
    ax.set_xticks(list(all_value_stats.keys()))
    ax.set_xlim(0,max_prob)
    ax.set_ylabel(name)

def plot_barplot(ax,
                plt_data,
                max_prob,
                color):
    ax.bar(plt_data["MID"], plt_data["MEAN_SELECTED_N"], width=0.005, label="Selected", color=color)
    ax.set_xlim(0,max_prob)