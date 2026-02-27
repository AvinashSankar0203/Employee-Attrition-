#!/usr/bin/env Rscript

# ============================================================
# Employee Attrition Prediction (R)
# Models: Logistic Regression, Decision Tree, Random Forest, SVM
# Metrics: Accuracy, Precision, Recall, F1, ROC/AUC
# ============================================================

suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(ggplot2)
  library(randomForest)
  library(e1071)
  library(pROC)
  library(readr)
})

# ---------------------------
# Simple CLI args parser
# ---------------------------
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  if (!(flag %in% args)) return(default)
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) return(default)
  return(args[i + 1])
}

DATA_PATH <- get_arg("--data", default = "")
SPLIT     <- as.numeric(get_arg("--split", default = "0.7"))
SEED      <- as.integer(get_arg("--seed", default = "42"))
OUT_DIR   <- get_arg("--out", default = "outputs")

if (DATA_PATH == "" || !file.exists(DATA_PATH)) {
  cat("ERROR: Provide a valid dataset path.\n",
      "Example:\n",
      "  Rscript attrition_model.R --data data/WA_Fn-UseC_-HR-Employee-Attrition.csv\n",
      sep = "")
  quit(status = 1)
}

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
set.seed(SEED)

# ---------------------------
# Load data
# ---------------------------
df <- read_csv(DATA_PATH, show_col_types = FALSE)

if (!("Attrition" %in% names(df))) {
  stop("Target column 'Attrition' not found. Please use the IBM HR Attrition dataset (Attrition = Yes/No).")
}

# ---------------------------
# Basic preprocessing
# ---------------------------

# Drop commonly non-informative columns (safe to ignore if not present)
drop_cols <- c(
  "EmployeeCount", "Over18", "StandardHours",
  "EmployeeNumber" # identifier
)
drop_cols <- drop_cols[drop_cols %in% names(df)]
if (length(drop_cols) > 0) {
  df <- df %>% select(-all_of(drop_cols))
}

# Remove rows with missing values (simple baseline)
df <- na.omit(df)

# Ensure target is factor with positive class = "Yes"
df$Attrition <- factor(df$Attrition, levels = c("No", "Yes"))

# Convert character columns to factors
char_cols <- names(df)[sapply(df, is.character)]
for (cc in char_cols) df[[cc]] <- as.factor(df[[cc]])

# Identify numeric predictors to scale (exclude target)
num_cols <- setdiff(names(df)[sapply(df, is.numeric)], "Attrition")
preproc <- preProcess(df[, num_cols, drop = FALSE], method = c("center", "scale"))
df_scaled <- df
df_scaled[, num_cols] <- predict(preproc, df[, num_cols, drop = FALSE])

# ---------------------------
# Train/test split
# ---------------------------
idx <- createDataPartition(df_scaled$Attrition, p = SPLIT, list = FALSE)
train <- df_scaled[idx, ]
test  <- df_scaled[-idx, ]

# Helper: compute metrics
metrics_from_cm <- function(cm) {
  # Positive class is "Yes"
  byc <- cm$byClass
  acc <- as.numeric(cm$overall["Accuracy"])
  prec <- as.numeric(byc["Precision"])
  rec  <- as.numeric(byc["Recall"])
  f1   <- as.numeric(byc["F1"])
  tibble(Accuracy = acc, Precision = prec, Recall = rec, F1 = f1)
}

# Helper: compute ROC/AUC given probs for positive class
roc_auc <- function(actual, prob_yes) {
  r <- roc(response = actual, predictor = prob_yes, levels = c("No","Yes"), direction = "<", quiet = TRUE)
  list(roc = r, auc = as.numeric(auc(r)))
}

results <- list()
roc_list <- list()

# ---------------------------
# 1) Logistic Regression
# ---------------------------
logit_model <- glm(Attrition ~ ., data = train, family = binomial())

logit_prob <- predict(logit_model, newdata = test, type = "response")
logit_pred <- factor(ifelse(logit_prob >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_logit <- confusionMatrix(logit_pred, test$Attrition, positive = "Yes")

results[["LogisticRegression"]] <- metrics_from_cm(cm_logit) %>% mutate(Model = "LogisticRegression",
                                                                       AUC = roc_auc(test$Attrition, logit_prob)$auc)
roc_list[["LogisticRegression"]] <- roc_auc(test$Attrition, logit_prob)$roc

# ---------------------------
# 2) Decision Tree
# ---------------------------
tree_model <- rpart::rpart(Attrition ~ ., data = train, method = "class")
tree_prob <- predict(tree_model, newdata = test, type = "prob")[, "Yes"]
tree_pred <- factor(ifelse(tree_prob >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_tree <- confusionMatrix(tree_pred, test$Attrition, positive = "Yes")

results[["DecisionTree"]] <- metrics_from_cm(cm_tree) %>% mutate(Model = "DecisionTree",
                                                                 AUC = roc_auc(test$Attrition, tree_prob)$auc)
roc_list[["DecisionTree"]] <- roc_auc(test$Attrition, tree_prob)$roc

# ---------------------------
# 3) Random Forest
# ---------------------------
rf_model <- randomForest::randomForest(Attrition ~ ., data = train, ntree = 500)
rf_prob <- predict(rf_model, newdata = test, type = "prob")[, "Yes"]
rf_pred <- factor(ifelse(rf_prob >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_rf <- confusionMatrix(rf_pred, test$Attrition, positive = "Yes")

results[["RandomForest"]] <- metrics_from_cm(cm_rf) %>% mutate(Model = "RandomForest",
                                                               AUC = roc_auc(test$Attrition, rf_prob)$auc)
roc_list[["RandomForest"]] <- roc_auc(test$Attrition, rf_prob)$roc

# ---------------------------
# 4) SVM (linear kernel)
# ---------------------------
# Use caret for probability extraction
ctrl <- trainControl(method = "none", classProbs = TRUE)
svm_model <- caret::train(
  Attrition ~ .,
  data = train,
  method = "svmLinear",
  trControl = ctrl
)

svm_prob <- predict(svm_model, newdata = test, type = "prob")[, "Yes"]
svm_pred <- factor(ifelse(svm_prob >= 0.5, "Yes", "No"), levels = c("No","Yes"))
cm_svm <- confusionMatrix(svm_pred, test$Attrition, positive = "Yes")

results[["SVMLinear"]] <- metrics_from_cm(cm_svm) %>% mutate(Model = "SVMLinear",
                                                             AUC = roc_auc(test$Attrition, svm_prob)$auc)
roc_list[["SVMLinear"]] <- roc_auc(test$Attrition, svm_prob)$roc

# ---------------------------
# Summarize metrics
# ---------------------------
metrics_df <- bind_rows(results) %>%
  select(Model, Accuracy, Precision, Recall, F1, AUC) %>%
  arrange(desc(F1))

print(metrics_df)

write.csv(metrics_df, file.path(OUT_DIR, "metrics.csv"), row.names = FALSE)

best_model <- metrics_df$Model[1]
cat("\nBest model by F1:", best_model, "\n")

# ---------------------------
# Plot ROC curves
# ---------------------------
roc_df <- bind_rows(lapply(names(roc_list), function(name) {
  r <- roc_list[[name]]
  tibble(
    Model = name,
    FPR = 1 - r$specificities,
    TPR = r$sensitivities
  )
}))

p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "ROC Curves (Employee Attrition)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

ggsave(filename = file.path(OUT_DIR, "roc_curves.png"), plot = p, width = 8, height = 6, dpi = 200)

cat("\nSaved outputs to:", OUT_DIR, "\n")
