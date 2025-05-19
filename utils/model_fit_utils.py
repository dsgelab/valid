import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from plot_utils import create_report_plots
from general_utils import get_date

def save_all_report_plots(out_data: pl.DataFrame,
                          out_plot_path: str,
                          out_down_path: str,
                          train_importances: pl.DataFrame=None,
                          valid_importances: pl.DataFrame=None,
                          test_importances: pl.DataFrame=None) -> None:
    fig = create_report_plots(out_data.filter(pl.col("SET") == 0).select("TRUE_ABNORM"), 
                              out_data.filter(pl.col("SET") == 0).select("ABNORM_PROBS"),
                              out_data.filter(pl.col("SET") == 0).select("ABNORM_PREDS"),
                              importances=train_importances)
    fig.savefig(out_plot_path + "train_report_" + get_date() + ".png")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 1).select("TRUE_ABNORM"), 
                              out_data.filter(pl.col("SET") == 1).select("ABNORM_PROBS"),
                              out_data.filter(pl.col("SET") == 1).select("ABNORM_PREDS"),
                              importances=valid_importances)
    fig.savefig(out_plot_path + "val_report_" + get_date() + ".png")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 2).select("TRUE_ABNORM"),
                                    out_data.filter(pl.col("SET") == 2).select("ABNORM_PROBS"),
                                    out_data.filter(pl.col("SET") == 2).select("ABNORM_PREDS"),
                                    importances=test_importances)
    fig.savefig(out_plot_path + "test_report_" + get_date() + ".png")

    fig = create_report_plots(out_data.filter(pl.col("SET") == 0).select("TRUE_ABNORM"),
                                    out_data.filter(pl.col("SET") == 0).select("ABNORM_PROBS"),
                                    out_data.filter(pl.col("SET") == 0).select("ABNORM_PREDS"),
                                    importances=train_importances,
                                    fg_down=True)
    fig.savefig(out_down_path + "train_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "train_report_" + get_date() + ".pdf")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 1).select("TRUE_ABNORM"), 
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PREDS"),
                                  importances=valid_importances,
                                  fg_down=True)
    fig.savefig(out_down_path + "val_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "val_report_" + get_date() + ".pdf")   

    fig = create_report_plots(out_data.filter(pl.col("SET") == 2).select("TRUE_ABNORM"),
                              out_data.filter(pl.col("SET") == 2).select("ABNORM_PROBS"),
                              out_data.filter(pl.col("SET") == 2).select("ABNORM_PREDS"),
                              importances=test_importances,
                              fg_down=True)
    fig.savefig(out_down_path + "test_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "test_report_" + get_date() + ".pdf")

