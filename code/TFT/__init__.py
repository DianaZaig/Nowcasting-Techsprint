import os
os.environ["KERAS_BACKEND"] = "tensorflow" # or "torch", "jax" according to user preference
import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin

class TFT(BaseEstimator, RegressorMixin):
    freq_rank = {
        "YE": 0,   # Yearly
        "QE": 1,   # Quarterly
        "ME": 2,   # Monthly
        "W": 3,   # Weekly
        "D": 4   # Daily
    }

    freq_len = { # measured in units of the highest frequency, which are days
        "YE": 365,
        "QE": 91,
        "ME": 30,
        "W": 7,
        "D": 1
    }

    @classmethod
    def _higher_freq(cls, X_keys, y_key):
        y_freq_rank = cls.freq_rank[list(y_key)[0]]
        return [k for k in list(X_keys) if cls.freq_rank[k] > y_freq_rank]

    @classmethod
    def _leq_freq(cls, X_keys, y_key):
        # returns the frequencies in `X` that are lower or equal to `y`
        y_freq_rank = cls.freq_rank[list(y_key)[0]]
        return [k for k in list(X_keys) + list(y_key) if cls.freq_rank[k] <= y_freq_rank]    

    @classmethod
    def _highest_freq(cls, X_keys, y_key):
        # Convert dict or dict_keys type y_key to string if necessary
        y_freq_rank = cls.freq_rank[list(y_key)[0]]

        # Filter keys that are strictly higher than y_key in frequency
        higher_freqs = [k for k in list(X_keys) if cls.freq_rank[k] > y_freq_rank]
        
        # Return the highest frequency (i.e., max by rank) if any are found
        if higher_freqs:
            return max(higher_freqs, key=lambda k: cls.freq_rank[k])
        else:
            return y_freq

    @classmethod
    def _count_nowcasting_days(cls, highest_freq, y_freq):
        return int(cls.freq_len[y_freq]/cls.freq_len[highest_freq])

    def __init__(
        self,
        name="TFT-MF",
        d_model:int=8,
        n_head:int=1,
        quantiles:list=[0.1, 0.5, 0.9],
        lags:dict={
            "YE": 1,   # Yearly
            "QE": 4,   # Quarterly
            "ME": 12,   # Monthly
            "W": 52,   # Weekly
            "D": 365   # Daily
        }, # need to contain at least the keys in X
        tscv=TimeSeriesSplit(n_splits=5),
        compile_args={
            "optimizer": keras.optimizers.Adam(learning_rate=1e-4, clipnorm=0.01)    
        },
        fit_args = {
            "epochs": 2, 
            "batch_size":10, 
            "shuffle": True,
            "callbacks": [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    min_delta=1e-4
                )
            ]},
        cache_data=True,
        cache_model=False,
        verbose=True
    ):
        self.name = name
        self.d_model = d_model
        self.n_head = n_head
        self.quantiles = quantiles
        self.lags = lags
        self.tscv = tscv
        self.compile_args = compile_args
        self.fit_args = fit_args
        self.cache_data=cache_data
        self.cache_model=cache_model
        self.verbose = verbose

    @keras.saving.register_keras_serializable()
    def _build_model(self):
        hist_inputs = {k: keras.layers.Input(shape=(None, self.n_features_in_[k]), name=f"HistInput__freq_{k}") for k in self.n_features_in_.keys()}
        encoded_cont_inputs = {
            k: keras.layers.TimeDistributed(
                keras.layers.Dense(self.d_model),
                name=f"encoding__freq_{k}"
            )(v)
            for k, v in hist_inputs.items()
        }

        higher_freqs = [k for k in list(hist_inputs.keys()) if self.freq_rank[k] > self.freq_rank[self.y_freq_]]
        fut_inputs = {k: keras.layers.Input(shape=(None, self.n_features_in_[self.highest_freq_X_]), name=f"FutInput__freq_{k}") for k in higher_freqs}

        #<temp LSTM>
        hist_LSTMs = {}
        hist_states_h = {}
        hist_states_c = {}
        for k, v in encoded_cont_inputs.items():
            lstm = keras.layers.Masking(mask_value=0.0)(v)
            lstm, state_h, state_c = keras.layers.LSTM(units=self.d_model, return_sequences=True, return_state=True, name=f"LSTMencoder__freq_{k}")(lstm)
            hist_LSTMs[k] = lstm
            hist_states_h[k] = state_h
            hist_states_c[k] = state_c

        # combine nonlinearly the states (at the end of historical series) from the different frequencies
        state_h = keras.ops.concatenate([v for v in hist_states_h.values()], axis=-1) if len(hist_states_h.keys()) > 1 else list(hist_states_h.values())[0]
        state_h = keras.layers.Dense(units=self.d_model, activation="sigmoid")(state_h)
        
        state_c = keras.ops.concatenate([v for v in hist_states_c.values()], axis=-1) if len(hist_states_c.keys()) > 1 else list(hist_states_c.values())[0]
        state_c = keras.layers.Dense(units=self.d_model, activation="sigmoid")(state_c)

        # combine the frequency time series by encoding the lower frequency LSTM memories into a higher frequency-consistent time series

        num_timesteps_transformer = self.lags[self.highest_freq_X_] + self.nowcasting_steps_ # in other words, historical time steps + future time steps, both at the highest frequency
        
        # the process to translate information from one frequency to another is only necessary if there is more than one frequency in the data
        if len(hist_states_h.keys()) > 1:
            latent = {
                k: keras.Sequential([
                    keras.layers.RepeatVector(self.lags[self.highest_freq_X_]),
                    keras.layers.LSTM(units=self.d_model, return_sequences=True),
                    keras.layers.TimeDistributed(keras.layers.Dense(units=self.d_model))
                ], name=f"translate__freq_{k}_to_{self.highest_freq_X_}")(v)
                for k, v in hist_states_h.items()
                if k != self.highest_freq_X_
            }
            # now an LSTM decoder for each frequency and then a time distributed dense layer, finaly combining the outpout with the originally high frequency before passing to the transformers
            all_latent = [v for v in latent.values()] + [hist_LSTMs[self.highest_freq_X_]]
            hist_LSTM = keras.ops.concatenate(all_latent, axis=-1)
            hist_LSTM = keras.layers.TimeDistributed(keras.layers.Dense(units=self.d_model, use_bias=False))(hist_LSTM)
        else:
            hist_LSTM = hist_LSTMs[self.highest_freq_X_]

        # for the future data, we use only the highest frequency time series, which is the same as the data being outputted


        future_dec = {
            k: keras.layers.LSTM(units=self.d_model, return_sequences=True, return_state=False, name=f"LSTMDecoder__freq_{k}")(v, initial_state=[state_h, state_c])
            for k, v in fut_inputs.items()
        }
        future_dec = keras.ops.concatenate([v for v in future_dec.values()], axis=-1)
        future_dec = keras.layers.TimeDistributed(keras.layers.Dense(units=self.d_model, use_bias="False"), name="SummariseFutureFeatures")(future_dec)

        #</temp LSTM>

        # concatenate along the time dimension
        if self.verbose: print(f"Shape of historical data: {hist_LSTM.shape}. Shape of future data: {future_dec.shape}")
        transformer_layer = keras.ops.concatenate([hist_LSTM, future_dec], axis=1) 
        # transformer_layer # (sample, time (t-self.lags[highest_freq_X]; t+self.nowcasting_steps), embe dim)
        
        num_quantiles = len(self.quantiles)
        
        
        # if self.use_static_context: pass

        # Output layers:
        # In order to enforce monotoncity of the quantiles forecast only the lowest quantile from a base forecast layer, and use output_len - 1 additional layers with ReLU activation to produce the difference between the current quantile and the previous one

        output_lowest_quant = keras.layers.TimeDistributed(
            keras.layers.Dense(1),
            name=f"output_q{str(self.quantiles[0])}"
        )(transformer_layer[Ellipsis, self.num_encoder_steps:, :])

        output_quant_deltas = [
            keras.layers.TimeDistributed(
                keras.layers.Dense(1, activation="relu", use_bias=False),
                name=f"output_delta_to_q{str(self.quantiles[i+1])}"
            )(transformer_layer[Ellipsis, self.num_encoder_steps:, :])
            for i in range(num_quantiles - 1)
        ]
        # now it is time to sum up the lowest quantile with the quantile deltas
        # output_lowest_quant_expanded = keras.ops.expand_dims(output_lowest_quant, axis=-1)
        quantile_outputs = [output_lowest_quant]
        for delta in output_quant_deltas:
            # Expand delta along the quantile axis and sum with the expanded lowest quantile
            #delta_expanded = keras.ops.expand_dims(delta, axis=-1)  # Shape: (samples, time, nowcasting_steps, 1)
            quantile_outputs.append(output_lowest_quant + delta)
        outputs = keras.ops.concatenate(quantile_outputs, axis=-1)
        model = keras.Model(inputs=[hist_inputs, fut_inputs], outputs=outputs)
        model.compile(**self.compile_args, loss=self.quantile_loss)
        return model

    def quantile_loss(self, y_true, y_pred):
        # Assuming quantiles is a numpy array of shape (q,)
        # Extend the shape of y_true to (b, t, 1) to align with y_pred's shape (b, t, q)
        y_true_extended = keras.ops.expand_dims(y_true, axis=-1)

        # Compute the difference in a broadcasted manner
        pred_diff = y_true_extended - y_pred

        # Calculate the quantile loss using broadcasting
        # No need for a loop; numpy will broadcast quantiles across the last dimension
        
        q = keras.ops.array(self.quantiles)
        q_loss = keras.ops.maximum(q * pred_diff, (q - 1) * pred_diff)
        
        # Average over the time axis
        q_loss = keras.ops.mean(q_loss, axis=-2)

        # Sum over the quantile axis to get the final loss
        final_loss = keras.ops.sum(q_loss, axis=-1)

        return final_loss

    def _add_last_avail_date(self, y_index):
        last_avail_dates = pd.DataFrame(index=y_index)
        def add_lastdate(df, offset_func, newcol=None):
            new_df = pd.concat([
                df,
                pd.DataFrame(
                    df.index - offset_func, 
                    index=df.index, 
                )], axis=1
            )
            new_df.columns = [c for c in new_df.columns[:-1]] + [f"LastAvail{newcol}"]
            return new_df
        if "D" in self.data_freqs_:
            last_avail_dates = add_lastdate(last_avail_dates, pd.offsets.DateOffset(days=1), newcol="D")
        if "W" in self.data_freqs_:
            last_avail_dates = add_lastdate(last_avail_dates, pd.offsets.DateOffset(weeks=1), newcol="W")
        if "ME" in self.data_freqs_:
            last_avail_dates = add_lastdate(last_avail_dates, pd.offsets.MonthEnd(1), newcol="ME")
        if "QE" in self.data_freqs_:
            last_avail_dates = add_lastdate(last_avail_dates, pd.offsets.QuarterEnd(1), newcol="QE")
        if "YE" in self.data_freqs_:
            last_avail_dates = add_lastdate(last_avail_dates, pd.offsets.YearEnd(1), newcol="YE")
        self.last_avail_dates_ = last_avail_dates

    def _prepare_pred_data(self, X, y):
        pred_data = {"X_hist": {}, "X_fut": {}, "y": y}
        for Xk, Xv in X.items():
            pred_data["X_hist"][Xk] = []
            for y_date in y.index:
                date_lim = self.last_avail_dates_[f"LastAvail{Xk}"][y_date]
                try:
                    to_pad = Xv[:date_lim].values
                except KeyError:
                    to_pad = np.zeros((1, 1))
                pred_data["X_hist"][Xk].append(to_pad)
            pred_data["X_hist"][Xk] = keras.utils.pad_sequences(pred_data["X_hist"][Xk], maxlen=self.lags[Xk], dtype=np.float32)

            if self.freq_rank[Xk] > self.freq_rank[self.y_freq_]:
                pred_data["X_fut"][Xk] = []    
                for y_date in y.index:
                    date_lim = self.last_avail_dates_[f"LastAvail{Xk}"][y_date]
                    date_lim_ydata = self.last_avail_dates_[f"LastAvail{self.y_freq_}"][y_date]
                    if date_lim_ydata < y_date:
                        try:
                            to_pad = Xv[date_lim_ydata:y_date][1:].values
                        except KeyError:
                            to_pad = np.zeros((1, 1))
                        pred_data["X_fut"][Xk].append(to_pad)
                pred_data["X_fut"][Xk] = keras.utils.pad_sequences(pred_data["X_fut"][Xk], padding="post", maxlen=self.nowcasting_steps_, dtype=np.float32)
        return pred_data

    def _organise_data(self, X, y):
        fit_data = {}
        for fold in self.split_idx_.keys():
            fit_data[fold] = {}
            for chunk in ["train", "valid"]:
                fit_data[fold][chunk] = {"X_hist": {}, "X_fut": {}, "y": y.iloc[self.split_idx_[fold][chunk]]}
                for Xk, Xv in X.items():
                    fit_data[fold][chunk]["X_hist"][Xk] = []
                    for y_date in y.index[self.split_idx_[fold][chunk]]:
                        date_lim = self.last_avail_dates_[f"LastAvail{Xk}"][y_date]
                        try:
                            to_pad = Xv[:date_lim].values
                        except KeyError:
                            to_pad = np.zeros((1, 1))
                        fit_data[fold][chunk]["X_hist"][Xk].append(to_pad)
                    fit_data[fold][chunk]["X_hist"][Xk] = keras.utils.pad_sequences(fit_data[fold][chunk]["X_hist"][Xk], maxlen=self.lags[Xk], dtype=np.float32)

                    if self.freq_rank[Xk] > self.freq_rank[self.y_freq_]:
                        fit_data[fold][chunk]["X_fut"][Xk] = []    
                        for y_date in y.index[self.split_idx_[fold][chunk]]:
                            date_lim = self.last_avail_dates_[f"LastAvail{Xk}"][y_date]
                            date_lim_ydata = self.last_avail_dates_[f"LastAvail{self.y_freq_}"][y_date]
                            if date_lim_ydata < y_date:
                                try:
                                    to_pad = Xv[date_lim_ydata:y_date][1:].values
                                except KeyError:
                                    to_pad = np.zeros((1, 1))
                                fit_data[fold][chunk]["X_fut"][Xk].append(to_pad)
                        fit_data[fold][chunk]["X_fut"][Xk] = keras.utils.pad_sequences(fit_data[fold][chunk]["X_fut"][Xk], padding="post", maxlen=self.nowcasting_steps_, dtype=np.float32)

        return fit_data

    def fit(self, X:dict, y:dict):
        # DOC: both X and y need to be dicts where the keys are frequencies, and the elements are pandas dataframes with dates as indices
        
        if self.verbose: print("Processing `X` and `y` data.")

        if self.verbose: print("  working out the frequencies...")
        
        # find all data with higher frequencies then `y`
        self.y_freq_ = list(y.keys())[0]
        self.highest_freq_X_ = self._highest_freq(X.keys(), y.keys())
         
        # now we resample `y` such that we are repeating it at the highest possible frequency; each such repetition will become the basis of the nowcast
        # in other words, the final transformer layer should 
        y_df = y[self.y_freq_]
        resampled_y_df = y_df if list(y.keys())[0] == self.highest_freq_X_ else y_df.resample(self.highest_freq_X_).bfill()
        self.resampled_y_dates = resampled_y_df.index
        self.num_encoder_steps = self.lags[self.highest_freq_X_]
        self.nowcasting_steps_ = self._count_nowcasting_days(self.highest_freq_X_, list(y.keys())[0])
        
        # to simplify the model, only include the static context if there is variation
        entities = resampled_y_df.columns
        self.use_static_context = True if len(entities) > 1 else False

        if self.verbose: print("  calculating ...")
        # this step finds, for each nowcasting time period (remember: measured at the highest possible frequency), the last available date for the lower frequencies
        #leq_freq_X = self._leq_freq(X.keys(), y.keys()) # for ME in all_freqs...
        self.data_freqs_ = list(X.keys()) + list(y.keys())
        self._add_last_avail_date(y_index=resampled_y_df.index)

        # now the `resampled_y_df` dates are split into different validation folds
        self.split_idx_ = {f"fold_{i}": {"train": train, "valid": test} for i, (train, test) in enumerate(self.tscv.split(resampled_y_df.index))}

        # the lagged `y` data is also used as explanatory variable, so it needs to be concatenated to that, according to the corresponding frequency
        for yk, yv in y.items():
            if yk in X:
                X[yk] = pd.concat([X[yk], yv], axis=1)
            else:
                X[yk] = yv

        # now use these last available dates to loop over the resampled_y data and for each resampled_y date, keep only those Xs that belong to the last available date and up to a lag for each frequency, padding when necessary
        
        if self.cache_data:
            filename = "TFT.pkl"
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    fit_data = pickle.load(f)
            else:
                fit_data = self._organise_data(X=X, y=resampled_y_df)
                with open(filename, "wb") as f:
                    pickle.dump(fit_data, file=f)
        else:
            fit_data = self._organise_data(X=X, y=resampled_y_df)
    
        if self.verbose: print("Processing `X` and `y` data: concluded")

        # TODO: construct tscv
        fold = "fold_0"
        # find how many continuous variables there are for each frequency
        self.n_features_in_ = {k: v.shape[-1] for k, v in fit_data[fold]["train"]["X_hist"].items()}
        # TODO: find all unique frequencies and calculate their time features
        
        def _build_and_fit(fit_data):
            self.model = self._build_model()
            self.history_ = self.model.fit(
                    x=[fit_data[fold]["train"]["X_hist"], fit_data[fold]["train"]["X_fut"]],
                    y=fit_data[fold]["train"]["y"],
                    validation_data=([fit_data[fold]["valid"]["X_hist"], fit_data[fold]["valid"]["X_fut"]], fit_data[fold]["valid"]["y"]),
                    **self.fit_args
                )

        if self.cache_model:
            model_filename = "mf_tft.keras"
            if os.path.isfile(model_filename):
                self.model = keras.saving.load_model(model_filename)
            else:
                _build_and_fit(fit_data=fit_data)
                self.model.save(model_filename)
        else:
            _build_and_fit(fit_data=fit_data)
            
    def predict(self, X, y):
        # unlike typical models, in `predict` we need the `y` data as well because the lags of this variable are incorporated in the covariate space
        y_df = y[self.y_freq_]
        resampled_y_df = y_df if list(y.keys())[0] == self.highest_freq_X_ else y_df.resample(self.highest_freq_X_).bfill()
        pred_data = self._prepare_pred_data(X=X, y=resampled_y_df)
        for k, v in pred_data["X_hist"].items():
            print(k, v.shape)
        for k, v in pred_data["X_fut"].items():
            print(k, v.shape)
        pred = self.model.predict([pred_data["X_hist"], pred_data["X_fut"]])
        return pred

    def fit_predict(self, X, y):
        self.fit(X=X, y=y)
        return self.predict(X=X)

    def document(self):
        # TODO: implement automatic documentation using gingado's forecastmodel card
        pass
    
    def summary(self):
        if not hasattr(self, "model"):
            self.model = self._build_model()
        self.model.summary()

    def plot_model(self, show_shapes=True, show_layer_names=True, show_layer_activations=True):
        return keras.utils.plot_model(self.model, show_shapes=show_shapes, show_layer_names=show_layer_names, show_layer_activations=show_layer_activations)

    def plot_loss(self):
        pd.DataFrame(self.history_.history).plot()
