library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
library(glmnet)
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")

# EDA
train_data$weather <- factor(train_data$weather)
train_data$workingday <- factor(train_data$workingday)
train_data$holiday <- factor(train_data$holiday)
train_data$season <- factor(train_data$season)

train_data <- train_data |>
  select(-casual) |>
  select(-registered)
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


# Data Cleaning Section

train <- train_data |>
  select(-casual, - registered) |>
  mutate(log_count = log(count + 1)) |>
  select(-count)

# Recipe Creation           
my_recipe <- recipe(log_count~., data=train) %>% 
step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
step_mutate(weather = factor(weather, levels=c(1,2,3), labels=c("clear", "cloudy", "severe"))) %>%
step_mutate(season = factor(season, levels=c(1,2,3,4), labels=c("spring", "summer", "fall", "winter"))) %>%
step_mutate(holiday = factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
step_mutate(workingday = factor(workingday, levels=c(0,1), labels=c("no", "yes"))) %>%
step_date(datetime, features="dow") %>% 
step_time(datetime, features=c("hour", "minute")) %>%
step_dummy(all_nominal_predictors()) %>% 
step_normalize(all_numeric_predictors())
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=train)
test <- bake(prepped_recipe, new_data = test_data)

# Penalized Regression, keep penalty really small
preg_model <- linear_reg(penalty=1, mixture=.5) %>% 
set_engine("glmnet") 
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=train)
bike_predictions <- predict(preg_wf, new_data=test_data)
bike_predictions

# Linear Regression
my_linear_model <- linear_reg() %>% 
set_engine("lm") %>% 
set_mode("regression") %>% 
fit(formula=log_count~., data=train)
bike_predictions <- predict(my_linear_model,
                            new_data=test_data) 
bike_predictions 

# Kaggle Submission
kaggle_submission <- bike_predictions %>%
bind_cols(., test) %>%
mutate(count = pmax(0, expm1(.pred))) %>%  
select(datetime, count) %>%                
mutate(datetime = as.character(format(datetime))) 
vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")

head(test, n = 5)


# Step 1: Prepare Training Data
train <- train_data |>
  select(-casual, -registered) |>
  mutate(log_count = log(count + 1)) |>
  select(-count)

# Step 2: Create Recipe
my_recipe <- recipe(log_count ~ ., data = train) %>%
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
  step_mutate(weather = factor(weather, levels = c(1,2,3), labels = c("clear", "cloudy", "severe"))) %>%
  step_mutate(season = factor(season, levels = c(1,2,3,4), labels = c("spring", "summer", "fall", "winter"))) %>%
  step_mutate(holiday = factor(holiday, levels = c(0,1), labels = c("no", "yes"))) %>%
  step_mutate(workingday = factor(workingday, levels = c(0,1), labels = c("no", "yes"))) %>%
  step_date(datetime, features = "dow") %>%
  step_time(datetime, features = c("hour", "minute")) %>%
  step_rm(datetime) %>%                    # <-- Fix 1: Remove non-numeric datetime
  step_zv() %>%                            # <-- Fix 2: Remove zero-variance columns
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Step 3: Preprocess and bake manually (optional, but not necessary for workflow)
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data = train)
test <- bake(prepped_recipe, new_data = test_data)

# Step 4: Define Model
preg_model <- linear_reg(penalty = 1, mixture = .5) %>%
  set_engine("glmnet")

# Step 5: Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data = train)

# Step 6: Predict
bike_predictions <- predict(preg_wf, new_data = test_data)

# Step 7: (Optional) Back-transform log predictions to original scale
bike_predictions <- bike_predictions %>%
  mutate(count = exp(.pred) - 1)


