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
train <- train_data |>
  select(-casual, - registered) |>
  mutate(log_count = log(count + 1)) |>
  select(-count)

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
  step_time(datetime, features = "hour") %>%
  step_date(datetime, features = c("month", "year")) %>%
  step_mutate(hour = as.factor(datetime_hour)) %>%
  step_rm(datetime, datetime_hour) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact(terms = ~ starts_with("hour_"):starts_with("workingday_")) %>%
  step_interact(terms = ~ datetime_year:starts_with("workingday_")) %>%
  step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=train)
test <- bake(prepped_recipe, new_data = test_data)

###########################################################################

# Baked for Data Robot

train_baked <- bake(prepped_recipe, new_data = train_data)
test_baked <- bake(prepped_recipe, new_data = test_data)
vroom_write(train_baked, "data_robot_train_data.csv", delim = ",")
vroom_write(test_baked, "data_robot_test_data.csv", delim = ",")

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

# Random Forest Workflow
install.packages("rpart")
install.packages("ranger")

forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

grid_of_tuning_params <- grid_regular(mtry(range = c(1, 10)),
                                      min_n(range = c(2,20)),
                                      levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

# Set up grid of tuning values

CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

# Set up K-fold CV
bestTune <- CV_results %>%
  select_best(metric="rmse")

# Finalize workflow and predict
final_wf <-forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
bike_predictions <- final_wf %>%
  predict(new_data = test_data)

############################################################################

# Boost Model
library(bonsai)
library(lightgbm)
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

grid_of_tuning_params <- grid_regular(tree_depth(),
                                      trees(),
                                      learn_rate(),
                                      levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

# Set up grid of tuning values

CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

# Set up K-fold CV
bestTune <- CV_results %>%
  select_best(metric="rmse")

# Finalize workflow and predict
final_wf <-boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
bike_predictions <- final_wf %>%
  predict(new_data = test_data)

############################################################################

# Bart Model
bart_model <- bart(trees=tune()) %>% 
  set_engine("dbarts") %>% 
  set_mode("regression")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

grid_of_tuning_params <- grid_regular(trees(),
                                      levels = 5)

folds <- vfold_cv(train, v = 5, repeats=1)

# Set up grid of tuning values

CV_results <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

# Set up K-fold CV
bestTune <- CV_results %>%
  select_best(metric="rmse")

# Finalize workflow and predict
final_wf <-bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)
bike_predictions <- final_wf %>%
  predict(new_data = test_data)

############################################################################

#Stacking Model with H2O.ai
library(agua)
h2o::h2o.init()

auto_model <- auto_ml() %>%
  set_engine("h2o", max_models = 5) %>%
  set_mode("regression")

automl_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(auto_model) %>%
  fit(data = train)

bike_predictions <- automl_wf %>%
  predict(new_data = test_data)

############################################################################

kaggle_submission <- bike_predictions %>%
  bind_cols(., test_data) %>%
  mutate(count = pmax(0, expm1(.pred))) %>%  
  select(datetime, count) %>%
  mutate(datetime = as.character(format(datetime))) 
vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")
