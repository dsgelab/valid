
# Custom utils
import sys
sys.path.append(("/home/ivm/valid/scripts/utils/"))
# Standard stuff
import polars as pl

def get_abnorm_start_dates(data: pl.DataFrame) -> pl.DataFrame:
    """
    Get the start dates of abnormality sequences.
    """

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Preparing data                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    #data = data.with_columns(pl.when(pl.col.ABNORM_CUSTOM==0).then(0).otherwise(1).alias("ABNORM_BIN"))
    data = (data
            .filter((pl.len()>1).over("FINNGENID"))
            .filter(pl.col.ABNORM_CUSTOM.is_not_null())
            .sort(["FINNGENID", "DATE"], descending=False)
            .select("FINNGENID", "DATE", "ABNORM_CUSTOM")
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Shifting data                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
            .with_columns(pl.col.ABNORM_CUSTOM.shift(1).over("FINNGENID").alias("PREV_ABNORM"))
    )
    data = (data
            .with_columns(pl.when(pl.col.PREV_ABNORM.is_null())
                          .then(-1)
                          .otherwise(pl.col.PREV_ABNORM)
                          .alias("PREV_ABNORM"),
            )
    )
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Find Start                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    # Marks start of sequence where the prior abnormality is different from the current one
    # -1 is used to mark the start of the first sequence
    # Otherwise it will bee some shift from 0 to 1 or 1 to 0 etc.
    data = (data
            .with_columns(pl.when(pl.col.ABNORM_CUSTOM!=pl.col.PREV_ABNORM)
                            .then(pl.lit("START"))
                            .otherwise(pl.lit("CONTINUE"))
                            .alias("START")
            )
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #                 Add start dates                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
    data = (data
            .with_columns(pl.when(pl.col.START=="START").then(pl.col.DATE).otherwise(None).alias("START_DATE"))
    )
    # Recursively fill the start dates forward until all are filled
    # This is done to ensure that all start dates are filled in the same sequence
    data = data.with_columns(
                pl.col("START_DATE").fill_null(strategy="forward").over("FINNGENID")
    )
    # Add time difference between the start of the sequence and the current date
    data = data.with_columns((pl.col.DATE-pl.col.START_DATE).dt.total_days().alias("DIFF"))
    if data.schema["DATE"] == pl.Datetime: data=data.with_columns(pl.col.DATE.dt.date().alias("DATE"))
    if data.schema["START_DATE"] == pl.Datetime: data=data.with_columns(pl.col.START_DATE.dt.date().alias("START_DATE"))

    return(data)

def get_data_diags(data: pl.DataFrame, 
                   diff_days: int) -> pl.DataFrame:
    data = (data
                      # Candiates of abnormality sequences, x days apart
                      .filter((pl.col("ABNORM_CUSTOM") >= 1)&(pl.col("DIFF") >= diff_days))
                      .sort(["START_DATE", "DIFF"], descending=False)
                      # Longest difference from the start of this abnormality
                      .filter((pl.col.DIFF==pl.col.DIFF.first()).over(["FINNGENID", "START_DATE"]))
                      # First abnormality sequence that satisfies the criteria
                      .filter((pl.col.START_DATE==pl.col.START_DATE.first()).over("FINNGENID"))
                      .with_columns(pl.col("DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_DIAG_DATE"),
                                    pl.col("START_DATE").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("DATA_FIRST_DIAG_ABNORM_DATE"))
    )
    return(data)


import datetime 
def relabel_data_diag(data: pl.DataFrame,
                      start_pred_date: datetime,
                      shorten_pred_period: int,
                      diff_days: int) -> pl.DataFrame:
    # have to re-label data for after the start of the prediction period
    after_end_data = data.filter((pl.col.DATE>=start_pred_date)&(pl.col.ABNORM_CUSTOM != 0.5))
    after_end_data = get_abnorm_start_dates(after_end_data)
    after_end_data = get_data_diags(after_end_data, diff_days)
    if shorten_pred_period:
        # Join after_end_data back to data on FINNGENID (and START_DATE if needed)
        data = data.join(
                after_end_data.select(
                "FINNGENID", "DATA_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"
                ),
                on="FINNGENID",
                how="left"
        ).with_columns(
                pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE>=start_pred_date)
                .then(pl.col.DATA_DIAG_DATE_right)
                .otherwise(pl.col.DATA_DIAG_DATE)
                .alias("DATA_DIAG_DATE"),
                pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE>=start_pred_date)
                .then(pl.col.DATA_FIRST_DIAG_ABNORM_DATE_right)
                .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                .alias("DATA_FIRST_DIAG_ABNORM_DATE")
        ).drop(["DATA_DIAG_DATE_right", "DATA_FIRST_DIAG_ABNORM_DATE_right"])
    else:
        # Join after_end_data back to data on FINNGENID (and START_DATE if needed)
        data = data.join(
                after_end_data.select(
                "FINNGENID", "DATA_DIAG_DATE", "DATA_FIRST_DIAG_ABNORM_DATE"
                ),
                on="FINNGENID",
                how="left"
        ).with_columns(
                pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE<start_pred_date)
                .then(pl.col.DATA_DIAG_DATE_right)
                .otherwise(pl.col.DATA_DIAG_DATE)
                .alias("DATA_DIAG_DATE"),
                pl.when(pl.col.DATA_FIRST_DIAG_ABNORM_DATE<start_pred_date)
                .then(pl.col.DATA_FIRST_DIAG_ABNORM_DATE_right)
                .otherwise(pl.col.DATA_FIRST_DIAG_ABNORM_DATE)
                .alias("DATA_FIRST_DIAG_ABNORM_DATE")
        ).drop(["DATA_DIAG_DATE_right", "DATA_FIRST_DIAG_ABNORM_DATE_right"])
    data = (data
                 .with_columns(pl.min_horizontal(
                                  pl.col("DATA_DIAG_DATE"), 
                                  pl.col("FIRST_ICD_DIAG_DATE"), 
                                  pl.col("FIRST_MED_DIAG_DATE")
                               ).alias("FIRST_DIAG_DATE")
                  )
    )
    return(data)
