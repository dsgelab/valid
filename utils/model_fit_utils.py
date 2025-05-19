import polars as pl
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
from plot_utils import create_report_plots
from general_utils import get_date

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