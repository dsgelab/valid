library(readr)
library(dplyr)

# Data in at least 1% of individuals
icd_data <- readr::read_delim("/home/ivm/valid/icd_data/data/processed_data/step0/ICD_long_r12_2024-10-18.csv")
#N_total <- icd_data %>% pull(FINNGENID) %>% unique() %>% length()
N_total <- 520105 # from handbook R12 people with detailed longitduinal data
icd_data <- icd_data %>% dplyr::mutate(ICD_THREE=substr(ICD_CODE, 1, 3))
stats <- icd_data %>% group_by(ICD_THREE) %>% reframe(N_ENTRY=n(), N_INDV=length(unique(FINNGENID)), N_PERCENT=N_INDV/N_total) %>% arrange(desc(N_INDV))
icd_data <- dplyr::filter(icd_data, ICD_THREE %in% (stats %>% dplyr::filter(N_PERCENT >= 0.01) %>% dplyr::pull(ICD_THREE)))

# Data before end of observation period
end_dates <- readr::read_delim("/home/ivm/valid/data/processed_data/step4_labels/krea_labels_2024-10-18_1-year.csv")
end_dates <- end_dates %>% dplyr::select(FINNGENID, START_DATE)
icd_data <- dplyr::left_join(icd_data, end_dates, by="FINNGENID")
icd_data <- dplyr::mutate(icd_data, START_DATE=as.Date(START_DATE))
icd_data <- dplyr::filter(icd_data, DATE < START_DATE)

# Make wide count
icd_wider <- icd_data %>% dplyr::group_by(FINNGENID, ICD_THREE) %>% dplyr::reframe(N_DIAG=n()) %>% tidyr::pivot_wider(names_from=ICD_THREE, values_from=N_DIAG, values_fill=0)
last_date <- dplyr::select(icd_data, FINNGENID, DATE) %>% distinct() %>% dplyr::group_by(FINNGENID) %>% dplyr::arrange(FINNGENID, desc(DATE))
last_date <- last_date %>% slice_min(order_by=desc(DATE))
icd_wider <- dplyr::left_join(icd_wider, last_date)
icd_wider <- dplyr::rename(icd_wider, LAST_ICD_DATE=DATE)

icd_wider
readr::write_delim(icd_wider, "/home/ivm/valid/data/processed_data/step5_predict/r12_2024-10-18_min1pct_2024-01-27_krea_2024-10-18", delim=",")

