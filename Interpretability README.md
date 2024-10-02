We demonstrate interpretability two ways: (1) examining the importance of each input variable in prediction, (2) identifying any regimes or events that lead to significant changes in temporal dynamics. 

**1. Analyzing variable importance**
We aggregate selection weights for each variable across our entire test set, recording the 10th, 50th and 90th percentiles of each sampling distribution. We present the results for its variable importance analysis.

**2. Identifying regimes & significant events**
Identifying sudden changes in temporal patterns can also be very useful, as temporary shifts can occur due to the presence of significant regimes or events. 
.
**Conclusions**
We introduce TFT, a novel attention-based deep learning model for interpretable high-performance multi-horizon forecasting. To handle static covariates, a priori known inputs, and observed inputs effectively across a wide range of multi-horizon forecasting datasets, TFT uses specialized components. Specifically, these include: (1) sequence-to-sequence and attention-based temporal processing components that capture time-varying relationships at different timescales, (2) static covariate encoders that allow the network to condition temporal forecasts on static metadata, (3) gating components that enable skipping over unnecessary parts of the network, (4) variable selection to pick relevant input features at each time step, and (5) quantile predictions to obtain output intervals across all prediction horizons. 
