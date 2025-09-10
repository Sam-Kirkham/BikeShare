library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")

# train_data$weather <- factor(train_data$weather)
# train_data$workingday <- factor(train_data$workingday)
# train_data$holiday <- factor(train_data$holiday)
# train_data$season <- factor(train_data$season)

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


## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() %>% #Type of model
set_engine("lm") %>% # Engine = What R function to use
set_mode("regression") %>% # Regression just means quantitative response
fit(formula=count~., data=train_data)
## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=test_data) # Use fit to predict
bike_predictions ## Look at the output12


kaggle_submission <- bike_predictions %>%
bind_cols(., test_data) %>% #Bind predictions with test data
select(datetime, .pred) %>% #Just keep datetime and prediction variables
rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write out the file
vroom_write(x=kaggle_submission, file="./LinearPreds.csv", delim=",")
