if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, caTools, tidyverse, data.table, lubridate, tictoc, DescTools, lightgbm)
set.seed(84)               
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "../../input/train.csv"
test_path  <- "../../input/test.csv"

tr_col_names <- c("ip", "app", "device", "os", "channel", "click_time", "attributed_time", 
                  "is_attributed")
#truncated hours: 12pm-14pm, 17pm-19pm, 21pm-23pm (UTC + 8)
#including actual hours
test_hours_group1 <- c("4","5","6","6")
test_hours_group2 <- c("8","9","10","11")
test_hours_group3 <- c("12","13","14","15")
test_m_freq_hours <- c("4","5","9","10","13","14")
test_l_freq_hours <- c("6","7","8","11","12","15")


train_chunker <- function(chunk, skiprows, nrows) {
    cat("Piping train: ", chunk, "\n")
    df_name <- fread(train_path, skip=skiprows, nrows=nrows, colClasses=list(numeric=1:5),
                     showProgress = FALSE, col.names = tr_col_names) %>%
      select(-c(attributed_time)) %>%
      mutate(wday = Weekday(click_time), 
             hour = hour(click_time),
             test_hours_set1 =  ifelse(hour %in% test_hours_group1, 1, 
                                ifelse(hour %in% test_hours_group2, 2, 
                                ifelse(hour %in% test_hours_group3, 3, 4))),
             test_hours_set2 =  ifelse(hour %in% test_m_freq_hours, 1, 
                                ifelse(hour %in% test_l_freq_hours, 2, 3)))%>%
      select(-c(click_time)) %>%
      add_count(ip, test_hours_set1) %>% rename("nip_test_hours_set1" = n) %>%
      add_count(ip, wday, test_hours_set2) %>% rename("nip_day_test_hours_set2" = n) %>%
      select(-c(test_hours_set1, test_hours_set2)) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip, hour)) # drop direct dependency on time (experimental)
      return(df_name)
  }


#*****************************************************************
#Process chunk size of 75 million rows from training data

total_rows <- 184903890
chunk_rows <- 75000000
skip1_rows <- total_rows - chunk_rows 

tic("Total processing time for train data --->")
train <- train_chunker("Chunk of 75 million rows", skip1_rows, chunk_rows)
dim(train)
print(object.size(train), units = "auto")
toc()

invisible(gc())
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
print("free up memory by converting to lgb.Datast before reding test data")
#time based split
# tr_index <- nrow(train)
# dtrain <- train %>% head(0.95 * tr_index)
# valid  <- train %>% tail(0.05 * tr_index)

#shuffled split
sample = sample.split(train$is_attributed, SplitRatio = 0.95)
dtrain = subset(train, sample == TRUE)
valid  = subset(train, sample == FALSE)

print("Table of class unbalance")
table(dtrain$is_attributed)

rm(train)
invisible(gc())

cat("train size : ", dim(dtrain), "\n")
cat("valid size : ", dim(valid), "\n")

# categorical_features = c("app", "device", "os", "channel", "hour")
categorical_features = c("app", "device", "os", "channel")

dtrain = lgb.Dataset(data = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), 
                     label = dtrain$is_attributed, categorical_feature = categorical_features)
dvalid = lgb.Dataset(data = as.matrix(valid[, colnames(valid) != "is_attributed"]), 
                     label = valid$is_attributed, categorical_feature = categorical_features)

rm(valid)
invisible(gc())

cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
print("Piping test data:")

tic("Total processing time for test data --->")
test <- fread(test_path, colClasses=list(numeric=2:6), showProgress = FALSE)
sub  <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())

test <- test %>%
      mutate(wday = Weekday(click_time), 
             hour = hour(click_time),
             test_hours_set1 =  ifelse(hour %in% test_hours_group1, 1, 
                                ifelse(hour %in% test_hours_group2, 2, 
                                ifelse(hour %in% test_hours_group3, 3, 4))),
             test_hours_set2 =  ifelse(hour %in% test_m_freq_hours, 1, 
                                ifelse(hour %in% test_l_freq_hours, 2, 3)))%>%
      select(-c(click_time)) %>%
      add_count(ip, test_hours_set1) %>% rename("nip_test_hours_set1" = n) %>%
      add_count(ip, wday, test_hours_set2) %>% rename("nip_day_test_hours_set2" = n) %>%
      select(-c(test_hours_set1, test_hours_set2)) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip, hour)) # drop direct dependency on time (experimental)

print(object.size(test), units = "auto")
# test[num_vars] <- lapply(test[num_vars], normalize)
cat("test  size : ", dim(test), "\n")
toc()

dtest <- as.matrix(test[, colnames(test)])
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
#Modelling

print("Modelling")
params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.1, 
              num_leaves= 7,
              max_depth= 4,
              min_child_samples= 100,
              max_bin= 100,
              subsample= 0.7,
              subsample_freq= 1,
              colsample_bytree= 0.7,
              min_child_weight= 0,
              min_split_gain= 0,
              scale_pos_weight=99.7,
              reset_data = TRUE
              )

tic("Total time for model training --->")
model <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                   nrounds = 1000, verbose= 1, early_stopping_rounds = 20, eval_freq = 100)

rm(dtrain, dvalid)
invisible(gc())
toc()
cat("-----------------------------------", "\n")
cat("Validation AUC @ best iter: ", max(unlist(model$record_evals[["validation"]][["auc"]][["eval"]])))
cat("-----------------------------------", "\n")

#*****************************************************************
#Predictions

print("Predictions")
preds <- predict(model, data = dtest, n = model$best_iter)
preds <- as.data.frame(preds)
sub$is_attributed = preds
sub$is_attributed = round(sub$is_attributed,4)
fwrite(sub, "sub_lightgbm_R_75m_th_grp.csv")
head(sub,10)

print("Feature importance")
kable(lgb.importance(model, percentage = TRUE))

print("finished...")