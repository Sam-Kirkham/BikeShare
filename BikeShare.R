library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)
library(ggplot2)
library(GGally)
library(patchwork)
test_data <- vroom("test.csv")
train_data <- vroom("train.csv")
train_data$weather <- as.factor(train_data$weather)
train_data$season <- as.factor(train_data$season)
train_data$holiday <- as.factor(train_data$holiday)
train_data$workingday <- as.factor(train_data$workingday)
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
