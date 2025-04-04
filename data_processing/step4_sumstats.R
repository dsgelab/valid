.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

#install.packages("readr")
#install.packages("dplyr")
#install.packages("optparse")
#install.packages("e1071")
#install.packages("lubridate")
library(plyr)
library(readr)
library(dplyr)
library(optparse)
library(e1071)
library(lubridate)
library(arrow)

options(width=250)

args_list <- list(
  make_option(c("-r", "--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step5_data/1_year_buffer/",
              help="Path to results directory."),
  make_option(c("-f", "--file_path"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step4_labels/",
              help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end"),
  make_option(c("-d", "--file_name"), action="store", type="character", default="2025-01-31",help="Date of original processed file."),
  make_option(c("--file_path_labels"), action="store", type="character", default="",help="If label file different. [default: '']"),
  make_option(c("--custom_file_name"), action="store", type="character", default="",help="If label file different. [default: '']")
  
)
parser <- OptionParser(option_list=args_list)
args <- parse_args(parser, positional_arguments=0)$options
print(args)

date = Sys.Date()
file_path_data <- paste0(args$file_path, args$file_name, ".parquet")

data <- tibble::as_tibble(arrow::read_parquet(file_path_data))
if(args$file_path_labels != "") {
  labels <-  tibble::as_tibble(arrow::read_parquet(args$file_path_labels))
  out_file_path <- paste0(args$res_dir, args$custom_file_name, "_sumstats.parquet")
} else {
  if(args$custom_file_name == ""){
    out_file_path <- paste0(args$res_dir, args$file_name, "_sumstats.parquet")
  } else {
    out_file_path <- paste0(args$res_dir, args$custom_file_name, "_sumstats.parquet")
  }
  labels <- tibble::as_tibble(arrow::read_parquet(paste0(args$file_path, args$file_name, "_labels.parquet")))
}

data <- left_join(labels %>% select(FINNGENID, START_DATE, SET), data) %>% arrange(FINNGENID) 
print(data)

mean_val <- mean(data %>% dplyr::filter(SET == 0) %>% pull(VALUE))
set.seed(10234)

sumstats <- data  %>% group_by(FINNGENID) %>% dplyr::filter(DATE < START_DATE, !is.na(VALUE)) %>% dplyr::arrange(desc(DATE)) %>%
  dplyr::reframe(MIN=min(VALUE), 
                 MAX=max(VALUE), 
                 SD=sd(VALUE), 
                 MEAN=mean(VALUE), 
                 MEDIAN=median(VALUE), 
                 SUM=sum(VALUE), 
                 KURT=e1071::kurtosis(VALUE), 
                 SKEW=e1071::skewness(VALUE), 
                 ABS_ENERG=sum(VALUE**2),
                 SUM_ABS_CHANGE=sum(abs(VALUE-lead(VALUE)), na.rm=TRUE),
                 MEAN_ABS_CHANGE=mean(abs(VALUE-lead(VALUE)), na.rm=TRUE),
                 MAX_ABS_CHANGE=max(abs(lead(VALUE)-VALUE), na.rm=TRUE),
                 MAX_CHANGE=max(lead(VALUE)-VALUE, na.rm=TRUE),
                 MEAN_CHANGE=mean(VALUE-lag(VALUE), na.rm=TRUE),
                 UNIQUE_VALS=length(unique(round(VALUE))),
                 SEQ_LEN=length(VALUE),
                 QUANT_25=quantile(VALUE, probs=c(0.25)),
                 QUANT_75=quantile(VALUE, probs=c(0.75)),
                 IDX_QUANT_100=VALUE[quantile(1:n(), probs=c(0))],
                 IDX_QUANT_50=VALUE[quantile(1:n(), probs=c(0.5))],
                 IDX_QUANT_0=VALUE[quantile(1:n(), probs=c(1))],
                 ABNORM=sum(ABNORM_CUSTOM),
                 MIN_LOC=lubridate::time_length(DATE[which.min(VALUE)]%--%START_DATE, "days"), 
                 MAX_LOC=lubridate::time_length(DATE[which.max(VALUE)]%--%START_DATE, "days"),
                 FIRST_LAST=lubridate::time_length(min(DATE)%--%max(DATE), "days"),
                 LAST_VAL_DATE=max(DATE),
                 SET=SET) %>% distinct() %>% ungroup() 
sumstats %>%mutate(FIRST_LAST=round(FIRST_LAST/365.25)) %>%pull(FIRST_LAST) %>% round() %>% table()
sumstats <- dplyr::mutate(sumstats, MAX_CHANGE=ifelse(is.infinite(MAX_CHANGE), NA, MAX_CHANGE), MAX_ABS_CHANGE=ifelse(is.infinite(MAX_ABS_CHANGE), NA, MAX_ABS_CHANGE))
sumstats <- dplyr::select(sumstats, -UNIQUE_VALS)
sumstats$LAST_VAL_DATE <- as.Date(sumstats$LAST_VAL_DATE)

## Adding sumstats for missing values
# Missing data-imputation
if(args$file_path_labels == "")  {
  missing_data <- dplyr::group_by(data, FINNGENID) %>% dplyr::filter(n()==1, is.na(VALUE)) %>% ungroup()
  train_sumstats <- dplyr::filter(sumstats,SET==0)
  
  missing_data_sumstats <- missing_data %>% dplyr::mutate(MIN=mean(train_sumstats$MIN, na.rm=TRUE), 
                                                          MAX=mean(train_sumstats$MAX, na.rm=TRUE), 
                                                          SD=NA, 
                                                          MEAN=mean(train_sumstats$MEAN, na.rm=TRUE), 
                                                          MEDIAN=mean(train_sumstats$MEDIAN, na.rm=TRUE),
                                                          SUM=mean(train_sumstats$SUM, na.rm=TRUE), 
                                                          KURT=NA, 
                                                          SKEW=NA, 
                                                          ABS_ENERG=mean(train_sumstats$ABS_ENERG, na.rm=TRUE), 
                                                          SUM_ABS_CHANGE=0,
                                                          MEAN_ABS_CHANGE=NA,
                                                          MAX_CHANGE=NA,
                                                          MAX_ABS_CHANGE=NA,
                                                          MEAN_CHANGE=NA, 
                                                          QUANT_25=mean(train_sumstats$QUANT_25, na.rm=TRUE), 
                                                          QUANT_75=mean(train_sumstats$QUANT_75, na.rm=TRUE),
                                                          IDX_QUANT_0=mean(train_sumstats$IDX_QUANT_0, na.rm=TRUE), 
                                                          IDX_QUANT_50=mean(train_sumstats$IDX_QUANT_50, na.rm=TRUE),
                                                          IDX_QUANT_100=mean(train_sumstats$IDX_QUANT_100, na.rm=TRUE), 
                                                          MIN_LOC=mean(train_sumstats$MIN_LOC, na.rm=TRUE),
                                                          MAX_LOC=mean(train_sumstats$MAX_LOC, na.rm=TRUE),
                                                          ABNORM=0,
                                                          SEQ_LEN=0,
                                                          FIRST_LAST=0.0,
                                                          LAST_VAL_DATE=NA)
  missing_data_sumstats <- missing_data_sumstats[colnames(sumstats)]
  print(missing_data_sumstats)
  sumstats <- rbind(sumstats, missing_data_sumstats)
}

dir.create(args$res_dir, showWarnings=FALSE)
arrow::write_parquet(sumstats, out_file_path)