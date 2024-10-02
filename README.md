#  Nowcasting-Techsprint

## MF-TFT

The model used to nowcast bank liquidity metrics is an adapted version of the Temporal Fusion Transformer (Lim et al, 2021). These adaptations aim at facilitating the use of data with multiple frequency (eg, weekly, monthly) and to minimise the size of the model so that it is suitable for smaller scale data as typical in economic or financial settings.

### Overall architecture

The MF-TFT (multiple frequency temporal fusion transformer) builds on excellent ideas from the original TFT architecture, such as quantile predictions and a combination of time series encoding and a transformer architecture, with simplifications made to allow the model to run more efficiently. The end result is the architecture described below.

Some notation:

- $d_{\text{model}}$ is the dimensionality of the model. It is shared across all latent layers, ie embeddings, recurrent network memories and attention values.

- $l(f)$ is the number of lags considered for each frequency of time series data. For example, one year worth of lags would mean broadly $l(d) = 365$, $l(m) = 12$, etc. 

- $\tau_{\text{max}}$ is the maximum number of nowcasting steps. It is calculated from the number of time steps available at the highest frequency after the last data point of the dependent variable. For example, if the dependent variable is quarterly, and the highest frequency data is daily, then $\tau_{\text{max}} = 90$. 

1. **Variable encoding**: every input variable is encoded in $\mathbb{R}^{d_{\text{model}}}$

2. **Temporal encoding**: the encoded variables are then consolidated in a time series to make sense of the temporal dimension of information, with one temporal encoding for each frequency

3. **Frequency translation**: temporal encodings of lower frequency are translated - using an architecture similar to autoencoders - into a higher frequency representation

4. **Dynamic averaging of temporal encodings**: the temporal encodings, now all represented at the highest frequency, are weighted according to a learned parameter

5. **Transformer**: a transformer layer combines the weighted temporal encoding of historical data together with future data (ie, data from the highest frequency, measured *after* the last known data point of the dependent variable

6. **Quantile output**: the steps of the output from the transformer layer that correspond to the future data are then used to create one output for each quantile

### Dates

Unlike most traditional machine learning methods that involve time series (eg, texts, videos or audio), in economics and finance *dates* are very important. 

Dates have different frequencies: daily, weekly, monthly, etc. (In fact, there is much more complexity - eg, business days, etc but this is abstracted for now.)

In addition, the different date subdivisions have unequal lengths, for example. And many economic data depend on these fixed periods, 

MF-TFT accommodates this by filtering data according to dates, not to number of time steps.

### Usage 

```
model = TFT()
model.fit(
    x={"D": df_daily, "ME": df_monthly, "QE": df_quarterly},
    y={"ME": df_dependent}
)
```

