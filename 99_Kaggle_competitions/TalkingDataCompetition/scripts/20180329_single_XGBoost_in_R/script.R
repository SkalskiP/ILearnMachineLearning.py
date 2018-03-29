if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, pryr, caTools, tidyverse, data.table, lubridate, tictoc, DescTools, xgboost)
set.seed(84)               
options(scipen = 9999, warn = -1, digits= 4)

train_path <- "../../input/train.csv"
test_path  <- "../../input/test.csv"

tr_col_names <- c("ip", "app", "device", "os", "channel", "click_time", "attributed_time", 
                  "is_attributed")
most_freq_hours_in_test_data <- c("4","5","9","10","13","14")
least_freq_hours_in_test_data <- c("6","11","15")

cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
#Feature engineering

train_chunker <- function(chunk, skiprows, nrows) {
    cat("Piping train: ", chunk, "\n")
    df_name <- fread(train_path, skip=skiprows, nrows=nrows, colClasses=list(numeric=1:5),
                     showProgress = FALSE, col.names = tr_col_names) %>%
      select(-c(attributed_time)) %>%
      mutate(wday = Weekday(click_time), 
         hour = hour(click_time),
         in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                          ifelse(hour %in% least_freq_hours_in_test_data, 2, 3))) %>%
      select(-c(click_time)) %>%
      add_count(ip, wday, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
      select(-c(in_test_hh)) %>%
      add_count(ip, wday, hour) %>% rename("nip_day_hh" = n) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip))
      return(df_name)
  }

#*****************************************************************
#Process chunk size of 55 million rows from training data

total_rows <- 184903890
chunk_rows <- 50000000
skip1_rows <- total_rows - chunk_rows 

tic("Total processing time for train data --->")
train <- train_chunker("Chunk of 50 million rows", skip1_rows, chunk_rows)
dim(train)
print("train file size: ")
print(object.size(train), units = "auto")
toc()
invisible(gc())
cat("memory in use"); mem_used()

#*****************************************************************
# Reduce size before reading test data
print("shuffled split")

#time based split
# tr_index <- nrow(train)
# dtrain <- train %>% head(0.95 * tr_index)
# valid  <- train %>% tail(0.05 * tr_index)

#shuffled split
sample = sample.split(train$is_attributed, SplitRatio = .9)
dtrain = subset(train, sample == TRUE)
valid  = subset(train, sample == FALSE)

rm(train, sample)
invisible(gc())

print("Table of class unbalance")
table(dtrain$is_attributed)
rm(train)
invisible(gc())

cat("memory in use"); mem_used()
cat("--------------------------------", "\n")
cat("train dim : ", dim(dtrain), "\n")
cat("valid dim : ", dim(valid), "\n")

print("Converting to xgb.DMatrix")
dtrain <- xgb.DMatrix(as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), 
                      label = dtrain$is_attributed)
cat("memory in use"); mem_used()                      
dvalid <- xgb.DMatrix(as.matrix(valid[, colnames(valid) != "is_attributed"]), 
                      label = valid$is_attributed)
rm(valid)
invisible(gc())
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
# Separate processing to avoid hitting memory limit of ~ 17 GB
print("Piping test data:")

tic("Total processing time for test data --->")
test <- fread(test_path, colClasses=list(numeric=2:6), showProgress = FALSE)
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())

test <- test %>%
      mutate(wday = Weekday(click_time), 
         hour = hour(click_time),
         in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                          ifelse(hour %in% least_freq_hours_in_test_data, 2, 3))) %>%
      select(-c(click_time)) %>%
      add_count(ip, wday, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
      select(-c(in_test_hh)) %>%
      add_count(ip, wday, hour) %>% rename("nip_day_hh" = n) %>%
      select(-c(wday)) %>%
      add_count(ip, hour, os) %>% rename("nip_hh_os" = n) %>%
      add_count(ip, hour, app) %>% rename("nip_hh_app" = n) %>%
      add_count(ip, hour, device) %>% rename("nip_hh_dev" = n) %>%
      select(-c(ip))
dim(test)
print("test file size: ")
print(object.size(test), units = "auto")
toc()
invisible(gc())
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

dtest  <- xgb.DMatrix(as.matrix(test[, colnames(test)]))
rm(test)
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")

#*****************************************************************
# Set params and run XGBoost model

params <- list( objective 	= "binary:logistic", 
				grow_policy = "lossguide",
				tree_method = "hist",
				eval_metric = "auc", 
				max_leaves 	= 7, 
				max_delta_step = 7,
				scale_pos_weight = 9.7,
				eta = 0.1, 
				max_depth = 4, 
				subsample = 0.9, 
				min_child_weight = 0,
				colsample_bytree = 0.7, 
				random_state = 84
				)
				
tic("Total time for model training --->")
xgb.model <- xgb.train(data = dtrain, params = params, maximize= TRUE,
				       silent = 1, watchlist = list(valid = dvalid), nthread = 4, 
                       nrounds = 1100, print_every_n = 50, early_stopping_rounds = 50)
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")
rm(dtrain, dvalid)
invisible(gc())
toc()

cat("Best score: ", xgb.model$best_score, "\n") 
cat("Best iteration: ", xgb.model$best_ntreelimit, "\n") 
cat("memory in use"); mem_used()
cat("--------------------------------", "\n")
                            
#*****************************************************************
# Predictions

print("Predictions")
preds <- predict(xgb.model, newdata = dtest, ntreelimit = xgb.model$best_ntreelimit)
rm(dtest)
invisible(gc())

preds <- as.data.frame(preds)
sub$is_attributed = round(preds, 5)
fwrite(sub, "../../sub_xgb_hist_R_50m.csv")
head(sub,10)
cat("--------------------------------", "\n")

#*****************************************************************
# Performance evaluation

print("Feature importance")
kable(xgb.importance(model = xgb.model))

print("finished")