.libPaths("/home/ivm/R/x86_64-pc-linux-gnu-library/4.4")

library(optparse)
library(readr)
library(dplyr)
library(lubridate)


args_list <- list(
  make_option(c("-r", "--res_dir"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step3_meta/",help="Path to results directory."),
  make_option(c("-f", "--file_path"), action="store", type="character", default="/home/ivm/valid/data/processed_data/step1_clean/", help="Path to data. Needs to contain both data and metadata (same name with _name.csv) at the end"),
  make_option(c("-n", "--file_name"), type="character", action="store", default="krea",help="Readable name of the measurement value."),
  make_option(c("-d", "--diags_file_path"), type="character", action="store", default="krea",help="Readable name of the measurement value.")
  make_option(c("-m", "--min_diff"), type="character", action="store", default="krea",help="Minimum num days for a data-based diagnosis."),
  make_option(c("-m", "--min_diff"), type="character", action="store", default="krea",help="Minimum num days for a data-based diagnosis."),

)

parser <- OptionParser(option_list=args_list)
args <- parse_args(parser, positional_arguments=0)$options
print(args)
get_abnorm_start_dates <- function(data) {
  data <- data %>% 
    # No missing abnormality
    dplyr::filter(!is.na(ABNORM_CUSTOM)) %>% 
    dplyr::arrange(FINNGENID, DATE) %>% 
    dplyr::select(FINNGENID, DATE, VALUE, ABNORM_CUSTOM) %>%
    dplyr::mutate(DATE=as.Date(DATE)) %>%
    dplyr::group_by(FINNGENID) %>% 
    # Cannot define length for single value individuals
    dplyr::filter(n() > 1) %>%
    dplyr::mutate(PREV_ABNORM=lag(ABNORM_CUSTOM),
                  # Very start of the period has an extra value that it is not the same as ABNORM
                  PREV_ABNORM=ifelse(is.na(PREV_ABNORM), -1, PREV_ABNORM),
                  START=ifelse(ABNORM_CUSTOM!=PREV_ABNORM, "START", NA),
                  START_DATE=ifelse(!is.na(START), DATE, NA)) 
  # Start filling start date column for first rows for block
  data$START_DATE <- as.Date(NA)
  data$START_DATE[!is.na(data$START)] <- as.Date(data$DATE[!is.na(data$START)])
  # Fill consequent rows of block with start date from before recursive until all filled
  while(sum(is.na(data %>% pull(START_DATE))) > 0) {
    data$START_DATE[is.na(data$START_DATE)] <- as.Date(lag(data$START_DATE)[is.na(data$START_DATE)])
  }
  
  return(data)
}

get_block_lengths <- function(data) {
  block_lengths <- dplyr::group_by(data, FINNGENID, ABNORM_CUSTOM, START_DATE) %>%
    # Cannot define period when less than one measurement available
    dplyr::filter(n() >= 2) %>%
    # Length of measurements between first and last of that abnormality
    dplyr::reframe(N_MEASURE=n(), DIFF=time_length(min(DATE)%--%max(DATE), "days")) %>%
    dplyr::distinct()
}

get_all_time_diffs <- function(data) {
  data %>% dplyr::group_by(FINNGENID, ABNORM_CUSTOM, START_DATE) %>%
    dplyr::filter(n() >= 2) %>%
    dplyr::reframe(DATE, N_MEASURE=n(), DIFF=time_length(START_DATE%--%DATE, "days")) %>%
    dplyr::distinct() 
}

data <- readr::read_delim(paste0(args$file_path, args$file_name, ".csv"))  
prep_data <- get_abnorm_start_dates(data) 
#abnorm_lens <- get_block_lengths(prep_data)
all_times <- get_all_time_diffs(prep_data)

############# Data-based diagnoses
# First period of abnormality
all_diags <- all_times %>% dplyr::filter(ABNORM_CUSTOM > 0, DIFF >= args$min_diff) %>% dplyr::group_by(FINNGENID, START_DATE) %>% dplyr::arrange(DIFF) %>% slice(1L) 
# First moment for diagnosis
all_diags <- all_diags %>% dplyr::ungroup() %>% dplyr::group_by(FINNGENID) %>% dplyr::arrange(DATE) %>% slice(1L) %>% dplyr::ungroup()
# renaming
all_diags <- all_diags %>% dplyr::ungroup() %>% select(FINNGENID, DATE, START_DATE)  %>% dplyr::rename(DATA_DIAG_DATE=DATE, DATA_FIRST_DIAG_ABNORM_DATE=START_DATE) %>% dplyr::mutate(DATA_DIAG_DATE=as.Date(DATA_DIAG_DATE), DATA_FIRST_DIAG_ABNORM_DATE=as.Date(DATA_FIRST_DIAG_ABNORM_DATE))

############# First abnormality and some stats
first_abnorm <- dplyr::filter(all_times, ABNORM_CUSTOM>0) %>% group_by(FINNGENID) %>% arrange(desc(DATE)) %>% slice(1L) %>% dplyr::select(FINNGENID, START_DATE, N_MEASURE, DIFF)
first_abnorm <- dplyr::rename(first_abnorm, FIRST_ABNORM_DATE=START_DATE, FIRST_ABNORM_N=N_MEASURE, FIRST_ABNORM_DIFF=DIFF)
# missing individuals with only a single abnormality
single_abnorms <- dplyr::filter(data, ABNORM_CUSTOM>0) %>% dplyr::arrange(DATE) %>% dplyr::group_by(FINNGENID) %>% slice(1L) %>% dplyr::select(FINNGENID, DATE)
first_abnorm <- dplyr::left_join(first_abnorm, single_abnorms, by="FINNGENID")
# right now this means we prefer abnormalities with at least 2 measurements but if not available we take the first alone one
first_abnorm$FIRST_ABNORM_DATE[is.na(first_abnorm$FIRST_ABNORM_DATE)] <- first_abnorm$FIRST_ABNORM_DATE[is.na(first_abnorm$DATE)] 
first_abnorm <- dplyr::select(first_abnorm, -DATE)

all_meta <- dplyr::full_join(all_diags, first_abnorm, by="FINNGENID")
############# ICD-based diagnoses
icd_data <- readr::read_delim(paste0(args$diags_file_path, "_diags.csv"), delim=",") 
icd_data <- dplyr::group_by(icd_data, FINNGENID) %>% dplyr::arrange(APPROX_EVENT_DAY) %>% slice(1L)
all_meta <- dplyr::full_join(all_meta, icd_data %>% rename(FIRST_ICD_DIAG_DATE=APPROX_EVENT_DAY) %>% dplyr::select(FINNGENID, FIRST_ICD_DIAG_DATE), by="FINNGENID")

############# Medication-based diagnoses
med_data <- readr::read_delim(paste0(args$diags_file_path, "_meds.csv"), delim=",")
med_data <- dplyr::group_by(med_data, FINNGENID) %>% dplyr::arrange(APPROX_EVENT_DAY) %>% slice(1L)
first_measure <- data %>% group_by(FINNGENID) %>% arrange(DATE) %>% slice(1L) %>% dplyr::rename(FIRST_MEASURE_DATE=DATE)
## Filtering for after first measurement minus 6 months
med_data <- dplyr::left_join(med_data, first_measure, by="FINNGENID") %>% dplyr::filter(APPROX_EVENT_DAY>=FIRST_MEASURE_DATE%m+%months(-6))
all_meta <- dplyr::full_join(all_meta, med_data %>% rename(FIRST_MED_DATE=APPROX_EVENT_DAY) %>% dplyr::select(FINNGENID, FIRST_MED_DATE), by="FINNGENID")
all_meta <- dplyr::mutate(all_meta, FIRST_DIAG_DATE=apply(all_meta %>% select(DATA_DIAG_DATE, FIRST_ICD_DIAG_DATE, FIRST_MED_DATE), 1, FUN=function(df){min(df, na.rm=TRUE)}))

############# age exclusion
age_excl <- data %>% group_by(FINNGENID) %>% arrange(desc(EVENT_AGE)) %>% slice(1L) 
age_excl <- age_excl %>% filter(EVENT_AGE < 30 | EVENT_AGE > 70) %>% dplyr::mutate(AGE_EXCL=1) %>% dplyr::select(FINNGENID, AGE_EXCL)
age_excl

all_meta <- dplyr::full_join(all_meta, age_excl %>% dplyr::select(FINNGENID, AGE_EXCL), by="FINNGENID")
all_meta <- dplyr::left_join(data %>% select(FINNGENID) %>% distinct(), all_meta, by="FINNGENID")
all_meta$AGE_EXCL[is.na(all_meta$AGE_EXCL)] <- 0
print(all_meta)

############# Writing
readr::write_delim(all_times, paste0(args$res_dir, args$file_name, "_alltimes.csv"), delim=",")
readr::write_delim(all_meta, paste0(args$res_dir, args$file_name,  "_meta.csv"), delim=",")
readr::write_delim(data %>% filter(!(FINNGENID %in% age_excl$FINNGENID)) %>% dplyr::mutate(DATE=as.Date(DATE)), paste0(args$res_dir, args$file_name, ".csv"), delim=",")