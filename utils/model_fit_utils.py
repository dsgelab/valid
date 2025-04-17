import polars as pl
from valid.utils.general_utils import get_date
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from valid.utils.general_utils import create_report_plots

def save_all_report_plots(out_data: pl.DataFrame,
                          out_plot_path: str,
                          out_down_path: str,
                          importances: pl.DataFrame=None) -> None:
    fig = create_report_plots(out_data.filter(pl.col("SET") == 0).select("TRUE_ABNORM"), 
                                  out_data.filter(pl.col("SET") == 0).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 0).select("ABNORM_PREDS"),
                                  importances=importances)
    fig.savefig(out_plot_path + "train_report_" + get_date() + ".png")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 1).select("TRUE_ABNORM"), 
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PREDS"),
                                  importances=importances)
    fig.savefig(out_plot_path + "val_report_" + get_date() + ".png")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 2).select("TRUE_ABNORM"),
                                    out_data.filter(pl.col("SET") == 2).select("ABNORM_PROBS"),
                                    out_data.filter(pl.col("SET") == 2).select("ABNORM_PREDS"),
                                    importances=importances)
    fig.savefig(out_plot_path + "test_report_" + get_date() + ".png")

    fig = create_report_plots(out_data.filter(pl.col("SET") == 0).select("TRUE_ABNORM"),
                                    out_data.filter(pl.col("SET") == 0).select("ABNORM_PROBS"),
                                    out_data.filter(pl.col("SET") == 0).select("ABNORM_PREDS"),
                                    importances=importances,
                                    fg_down=True)
    fig.savefig(out_down_path + "train_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "train_report_" + get_date() + ".pdf")
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 1).select("TRUE_ABNORM"), 
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 1).select("ABNORM_PREDS"),
                                  importances=importances,
                                  fg_down=True)
    fig.savefig(out_down_path + "val_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "val_report_" + get_date() + ".pdf")   
        
    fig = create_report_plots(out_data.filter(pl.col("SET") == 2).select("TRUE_ABNORM"),
                                  out_data.filter(pl.col("SET") == 2).select("ABNORM_PROBS"),
                                  out_data.filter(pl.col("SET") == 2).select("ABNORM_PREDS"),
                                  importances=importances,
                                  fg_down=True)
    fig.savefig(out_down_path + "test_report_" + get_date() + ".png")
    fig.savefig(out_down_path + "test_report_" + get_date() + ".pdf")

