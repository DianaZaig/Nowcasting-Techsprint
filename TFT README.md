Introduction
The main characteristics of TFT that make it interesting for nowcasting or forecasting purposes are:

multi-horizon forecasting: the ability to output, at each point in time 
, a sequence of forecasts for 
quantile prediction: each forecast is accompanied by a quantile band that communicates the amount of uncertainty around a prediction
flexible use of different types of inputs: static inputs (akin to fixed effects), historical input and known future input (eg, important holidays, years that are known to have major sports events such as Olympic games, etc)
interpretability: the model learns to select variables from the space of all input variables to retain only those that are globally meaningful, to assign attention to different parts of the time series, and to identify events of significance
Main innovations
The present model includes the following innovations:

Multi-frequency input

Context enhancement from lagged target: the last known values of the target variable are embedded (bag of observations), and this embedding is used similar to the static context enhancement as a starting point for the cell value in the decoder LSTM.
