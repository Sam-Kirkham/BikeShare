library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")

# Data Cleaning Section

train <- train_data |>
  select(-casual, - registered) |>
  mutate(log_count = log(count + 1)) |>
  select(-count)
           
my_recipe <- recipe(log_count~., data=train) %>% 
step_mutate(weather = ifelse(weather == 4, 3, weather)) %>%
step_mutate(weather = factor(weather, levels=c(1,2,3), labels=c("clear", "cloudy", "severe"))) %>%
step_mutate(season = factor(season, levels=c(1,2,3,4), labels=c("spring", "summer", "fall", "winter"))) %>%
step_mutate(holiday = factor(holiday, levels=c(0,1), labels=c("no", "yes"))) %>%
step_date(datetime, features="dow") %>% 
step_time(datetime, features=c("hour", "minute"))
prepped_recipe <- prep(my_recipe)
bake(prepped_recipe, new_data=train)



# train_data$weather <- factor(train_data$weather)
# train_data$workingday <- factor(train_data$workingday)
# train_data$holiday <- factor(train_data$holiday)
# train_data$season <- factor(train_data$season)

# train_data <- train_data |>
#   select(-casual) |>
#   select(-registered)
# plot_intro(train_data)
# plot_correlation(train_data)
# plot_bar(train_data)
# plot_histogram(train_data)
# plot_missing(train_data)
# ggpairs(train_data)
# plot1 <- ggplot(data = train_data,aes(x = weather))+
#   geom_bar()
# plot1
# plot2 <- ggplot(data = train_data, aes(x = temp, y = count)) + 
#   geom_point()
# plot2
# plot3 <- ggplot(data = train_data, aes(x = humidity)) + 
#   geom_bar()
# plot3
# plot4 <- ggplot(data = train_data, aes(x = windspeed)) + 
#   geom_histogram()
# plot4
# (plot1 + plot2) / (plot3 + plot4)


my_linear_model <- linear_reg() %>% 
set_engine("lm") %>% 
set_mode("regression") %>% 
fit(formula=log_count~., data=train)
bike_predictions <- predict(my_linear_model,
                            new_data=test_data) 
bike_predictions 


kaggle_submission <- bike_predictions %>%
bind_cols(., test_data) %>%
mutate(count = pmax(0, expm1(.pred))) %>%  
select(datetime, count) %>%                
mutate(datetime = as.character(format(datetime))) 
vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")
