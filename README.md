# Web Traffic Forecasting

## Description
* Kaggle's Problem
* As time series problem that predicting daily PV for each page in Google
* This competition focuses on the problem of forecasting the future values of multiple time series, as it has always been one of the most challenging problems in the field

## Algorithm
* 1) Detect CPD(Change point detection) and Frequency(w/ statistitics)
* 2) Redefine train data period with CPD, Fit ARIMA model w/ Frequency
* 3) To validation of Model's performance, fit train data without last 7-days, testing last 7-days
* 4) If SMAPE is large in last 7-days, Fit Deep learning model(Informer, SCINet, NLinear) 

## Reference
* https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/overview
