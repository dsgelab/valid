.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

library(readr)
library(dplyr)
library(optparse)
library(lubridate)

args_list <- list(
  make_option(c("-r", "--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step5_data/1_year_buffer/",
              help="Path to results directory."),
  make_option(c("-f", "--file_path_icds"), action="store", type="character", default="/home/ivm/valid/data/extra_data/data/processed_data/step1/icds_r12_2024-10-18_min1pct_sum_onttop_2025-02-10.csv",
              help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end"),
  make_option(c("-s", "--file_path"), type="character", action="store", default="krea",help="Readable name of the measurement value."),
  make_option(c("-d", "--file_name"), action="store", type="character", default="2024-10-18",help="Date of original processed file."),
  make_option(c("-h", "--current"), action="store", type="integer", default=0,help="Whether to filter for current and not historical data."),
  make_option(c("-o", "--out_file_name"), action="store", type="character", default="",help="idnetifier for out file."),
  make_option(c("-b", "--bin_count"), action="store", type="integer", default=0,help="whether to count number of occurance or stay binary observed/not observed.")
)

parser <- OptionParser(option_list=args_list, add_help_option=FALSE)
args <- parse_args(parser, positional_arguments=0)$options
print(args)

date = Sys.Date()

icd_data <- readr::read_delim(args$file_path_icds)

file_path_labels <- paste0(args$file_path, args$file_name, "_labels.csv")
end_dates <- readr::read_delim(file_path_labels)

if(args$current) {
  file_path_data <- paste0(args$file_path, args$file_name, ".csv")
  start_dates <- readr::read_delim(file_path_data) 
  start_dates <- dplyr::group_by(start_dates, FINNGENID) %>% dplyr::arrange(DATE) %>% slice(1L) %>% dplyr::rename(DATA_START_DATE=DATE) %>% dplyr::select(FINNGENID, DATA_START_DATE) %>% dplyr::ungroup()
  print(start_dates)
  start_dates <- dplyr::mutate(start_dates, DATA_START_DATE=DATA_START_DATE%m+%months(-12))
  icd_data <- dplyr::left_join(icd_data, start_dates, by="FINNGENID")
  icd_data <- dplyr::filter(icd_data, DATE>=DATA_START_DATE) %>% dplyr::select(-DATA_START_DATE)
  print(icd_data)
}
# Data in at least 1% of individuals
end_dates <- end_dates %>% dplyr::select(FINNGENID, START_DATE)
icd_data <- dplyr::left_join(icd_data, end_dates, by="FINNGENID")
icd_data <- dplyr::mutate(icd_data, START_DATE=as.Date(START_DATE))
icd_data <- dplyr::filter(icd_data, DATE < START_DATE)

# Count
icd_wider <- icd_data %>% dplyr::group_by(FINNGENID, ICD_THREE) %>% dplyr::reframe(N_DIAG=n()) %>% dplyr::ungroup()
if(args$bin_count == 1) {
  icd_wider <- icd_wider %>% dplyr::mutate(N_DIAG=ifelse(N_DIAG>0, 1, 0))
}
icd_wider <- icd_wider %>% tidyr::pivot_wider(names_from=ICD_THREE, values_from=N_DIAG, values_fill=0)

## Adding info on when was last recorded ICD-code
last_date <- dplyr::select(icd_data, FINNGENID, DATE) %>% distinct() %>% dplyr::group_by(FINNGENID) %>% dplyr::arrange(FINNGENID, desc(DATE))
last_date <- last_date %>% slice_min(order_by=desc(DATE))
icd_wider <- dplyr::left_join(icd_wider, last_date)
icd_wider <- dplyr::rename(icd_wider, LAST_ICD_DATE=DATE)
print(icd_wider)
dir.create(args$res_dir, showWarnings=FALSE, recursive = TRUE)

out_file_path <- ""
if(args$out_file_name != "") {
  out_file_path <- paste0(args$res_dir, "icds_", args$out_file_name, "_", args$file_name, ".csv")
} else {
  out_file_path <- paste0(args$res_dir, "icds_", args$file_name, ".csv")
}
readr::write_delim(icd_wider, out_file_path, delim=",")

