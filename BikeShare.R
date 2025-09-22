library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
library(glmnet)

############################################################################

# Reading in the Data
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")

############################################################################

# Setting as Factors
train_data$weather <- factor(train_data$weather)
train_data$workingday <- factor(train_data$workingday)
train_data$holiday <- factor(train_data$holiday)
train_data$season <- factor(train_data$season)

############################################################################

# Data Cleaning
train_data <- train_data |>
  select(-casual) |>
  select(-registered)

############################################################################

# EDA
plot_intro(train_data)
plot_correlation(train_data)
plot_bar(train_data)
plot_histogram(train_data)
plot_missing(train_data)
ggpairs(train_data)
plot1 <- ggplot(data = train_data,aes(x = weather))+
  geom_bar()
plot1
plot2 <- ggplot(data = train_data, aes(x = temp, y = count)) +
  geom_point()
plot2
plot3 <- ggplot(data = train_data, aes(x = humidity)) +
  geom_bar()
plot3
plot4 <- ggplot(data = train_data, aes(x = windspeed)) +
  geom_histogram()
plot4
(plot1 + plot2) / (plot3 + plot4)

############################################################################

# Recipe           
my_recipe <- recipe(log_count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = as.factor(weather),
              season = as.factor(season),
              holiday = as.factor(holiday),
              workingday = as.factor(workingday)) %>%
  step_date(datetime, features = "dow") %>%
  step_time(datetime, features = c("hour")) %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=train)
test <- bake(prepped_recipe, new_data = test_data)

############################################################################

# Penalized Regression Model and Workflow
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% 
  set_engine("glmnet") 
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) 
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))


collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()
bestTune <- CV_results %>%
  select_best(metric="rmse")


final_wf <-preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
final_wf %>%
  predict(new_data = test_data)

# Penalized Regression
preg_model <- linear_reg(penalty=.01, mixture=0) %>%
  set_engine("glmnet")
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=train)
bike_predictions <- predict(preg_wf, new_data=test_data)
bike_predictions

############################################################################

# Regression Trees Workflow
tree_mod <- decision_tree(tree_depth = tune(),
                          cost_complexity = tune(),
                          min_n=tune()) %>%
  set_engine("rpart") %>% 
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod)
grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5)
folds <- vfold_cv(train, v = 5, repeats=1)


CV_results <- tree_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

bestTune <- CV_results %>%
  select_best(metric="rmse")

final_wf <-tree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
bike_predictions <- final_wf %>%
  predict(new_data = test_data)


############################################################################

kaggle_submission <- bike_predictions %>%
  bind_cols(., test_data) %>%
  mutate(count = pmax(0, expm1(.pred))) %>%  
  select(datetime, count) %>%
  mutate(datetime = as.character(format(datetime))) 
vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")
