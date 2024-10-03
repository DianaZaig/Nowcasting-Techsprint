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

### Entity encoding

In economics and finance practice, often the only dimension that is "static" in panel data settings is the *entity* identification (country, bank, firm, household, etc). 

Based on this, the MF-TFT simplifies the original TFT architecture by avoiding static variable selection altogether, since it assumes there is only one such static variable.

This also enables a simplified API for the user. There is no need to prepare and pass to `fit` a separate set of static data with their vocabulary sizes (a technicality needed to embed categorical variables). As long as the `y` variable is a pandas DataFrame, its columns will be considered as the entity dimension.

This simplification does not mean the model provides less insights. In addition to helping select the input (time-varying) covariates for each entity's case, MF-TFT stores the entity embeddings. 

These embeddings can serve as input in analyses that are interesting in their own right. For example, the user can compare the embeddings to see which entities tend to react similarly to the same variables. For financial supervision, this can be an useful input for benchmarking.

### Usage 

Creating and fitting the model:

```
model = TFT()
model.fit(
    x={"D": df_daily, "ME": df_monthly, "QE": df_quarterly},
    y={"ME": df_dependent}
)
```

Using a fitted model to predict (note that `y` also needs to be passed due to how the model combines the data internally):

```
pred = model.predict(
    x={"D": df_daily, "ME": df_monthly, "QE": df_quarterly},
    y={"ME": df_dependent}
)

date_idx = tft.pred_dates[list(tft.pred_dates.keys())[-1]]
df_pred = pd.DataFrame(pred[-1,:,:], index=date_idx)
df_pred.columns = ["Q10", "Q50", "Q90"]
df_pred.plot()
```

Extracting variable weights:

```
tft.var_weights_
```

Comparing entities:

```
tft.entities
```

### Data

The data sources are considered to have potential informational value regarding liquidity risk for individual banks. In this step of the TechSprint, we considered the following potential data sources:

- Regulatory data (e.g., LCR) 
- Financial data from the credit institution (e.g., P&L) 
- Market data (e.g., CDS) 
- Macro data (e.g., GDP) 
- Social media data (e.g., Tweets)

Based on expert judgment, all of these data sources could potentially be helpful for the task of nowcasting liquidity risk metrics. 

To quickly develop the model and have a working prototype by the end of the event, the final data selection was based on existing research (https://www.sciencedirect.com/science/article/pii/S2666827023000646?via%3Dihub#sec2) on the use of Machine Learning (ML) to predict liquidity risk. The target variable in this setting was:

MLA = [Total liquid assets / Demand or liquid liabilities] × 100%

The input variables are split by the frequency, the *monthly* data is as follows:

1 F001_ASSET_CASH 2 F002_ASSET_BAL_BOT 3 F003_ASSET_BAL_BOT_SMR 4 F004_ASSET_BAL_BOT_CURRENT_ACCOUNT 5 F005_ASSET_BAL_BOT_OTHERS 6 F006_ASSET_BAL_OTHER_BANKS 7 F007_ASSET_BAL_OTHER_BANKS_TZ 8 F008_ASSET_BAL_OTHER_BANKS_ABROAD 9 F009_ASSET_BAL_OTHER_BANKS_ABROAD_FIXED_INV 10 F010_ASSET_BAL_OTHER_BANKS_ABROAD_OTHER_ACC 11 F011_ASSET_BAL_OTHER_BANKS_PROBABLE_LOSS 12 F012_ASSET_CLEARING_ITEMS 13 F013_ASSET_INV_DEBT_SECURITIES 14 F014_ASSET_INV_DEBT_SECURITIES_TBILLS 15 F015_ASSET_INV_DEBT_SECURITIES_OTHER_GOV_SECURITIES 16 F016_ASSET_INV_DEBT_SECURITIES_PRIVATE_SECURITIES 17 F017_ASSET_INV_DEBT_SECURITIES_OTHERS 18 F018_ASSET_INV_DEBT_SECURITIES_PROBABLE_LOSS 19 F019_ASSET_INTERBANK_LOANS_RECEIVABLE_NET 20 F020_ASSET_INTERBANK_CALL_LOANS_TZ 21 F021_ASSET_INTERBANK_OTHER_LOANS_TZ 22 F022_ASSET_INTERBANK_ABROAD 23 F023_ASSET_INTERBANK_PROBABLE_LOSS 24 F024_ASSET_LOANS_AND_ADVANCES_NET 25 F025_ASSET_LOANS_AND_ADVANCES 26 F026_ASSET_OD 27 F027_ASSET_RESTRUCTURED_LOANS 28 F028_ASSET_ACCRUED_INTERESTS 29 F029_ASSET_LOANS_AND_ADVANCES_PROBABLE_LOSS 30 F030_ASSET_INTEREST_SUSPENSE 31 F031_ASSET_COMM_N_OTHER_BILLS_PURCH_O_DISC 32 F032_ASSET_EXPORT_BILLS 33 F033_ASSET_IMPORT_BILLS 34 F034_ASSET_DOMESTIC_BILLS 35 F035_ASSET_CLEAN_BILLS_PURCHASED 36 F036_ASSET_DOM_ACCEPTANCES_DISCOUNTED 37 F037_ASSET_DOM_ACCEPTANCES_DISCOUNTED_OWN 38 F038_ASSET_DOM_ACCEPTANCES_DISCOUNTED_OTHERS 39 F039_ASSET_BILLS_PROBABLE_LOSS 40 F040_ASSET_CUSTOMER_LIAB_FOR_ACCEPTANCE 41 F041_ASSET_UNDERWRITTING_ACC 42 F042_ASSET_UNDERWRITTING_SECURITIES_PURCHASED 43 F043_ASSET_UNDERWRITTING_SALES_RECEIVABLES 44 F044_ASSET_UNDERWRITTING_PROBABLE_LOSS 45 F045_ASSET_EQUITY_INVESTMENT 46 F046_ASSET_EQUITY_INVESTMENT_SUBSDIARIES 47 F047_ASSET_EQUITY_INVESTMENT_OTHERS 48 F048_ASSET_EQUITY_INVESTMENT_PROBABLE_LOSS 49 F049_ASSET_CLAIM_ON_TREASURY 50 F050_ASSET_ON_USE_PREMISES_FURNITURE_EQUIPMENTS 51 F051_ASSET_ON_USE_FACILITIES 52 F052_ASSET_ON_USE_STAFF_HOUSING 53 F053_ASSET_ON_USE_ACC_DEPRECIATION 54 F054_ASSET_PROP_OWNED 55 F055_ASSET_PROP_OWNED_TO_3YRS 56 F056_ASSET_PROP_OWNED_ABOVE_3YRS 57 F057_ASSET_PROP_OWNED_PROBABLE_LOSSES 58 F058_ASSET_INTER_BRANCH_FLOAT 59 F059_ASSET_INTER_BRANCH_FLOAT_TO_30DAYS 60 F060_ASSET_INTER_BRANCH_FLOAT_31_60DAYS 61 F061_ASSET_INTER_BRANCH_FLOAT_61_90DAYS 62 F062_ASSET_INTER_BRANCH_FLOAT_91_18ODAYS 63 F063_ASSET_INTER_BRANCH_FLOAT_ABOVE_180DAYS 64 F064_ASSET_INTER_BRANCH_FLOAT_PROBABLE_LOSS 65 F065_ASSET_OTHERS 66 F066_ASSET_OTHERS_GOLD 67 F067_ASSET_OTHERS_STAMP_ACC 68 F068_ASSET_OTHERS_RETURNED_CHEQUES 69 F069_ASSET_OTHERS_OTHER_ACCR_INTEREST 70 F070_ASSET_OTHERS_SUNDRY_DEBTORS 71 F071_ASSET_OTHERS_PREPAID_EXP 72 F072_ASSET_OTHERS_DEFERRED_CHARGES 73 F073_ASSET_OTHERS_SHORTAGES_FRAUD 74 F074_ASSET_OTHERS_INTANGIBLE 75 F075_ASSET_OTHERS_MISCELLANEOUS 76 F076_ASSET_OTHERS_PROBABLE_LOSS 77 F077_ASSETS_TOTAL 78 F078_LIAB_DEPOSITS_OTHER_THAN_BANKS 79 F079_LIAB_DEPOSITS_OTHER_THAN_BANKS_CURRENT 80 F080_LIAB_DEPOSITS_OTHER_THAN_BANKS_SAVINGS 81 F081_LIAB_DEPOSITS_OTHER_THAN_BANKS_TIME_DEPOSITS 82 F082_LIAB_DEPOSITS_OTHER_THAN_BANKS_MATURED_TERM 83 F083_LIAB_DEPOSITS_OTHER_THAN_BANKS_DORMANT 84 F084_LIAB_SPECIAL_DEPOSIT 85 F085_LIAB_SPECIAL_DEPOSIT_SPECIAL_FINANCING 86 F086_LIAB_SPECIAL_DEPOSIT_CUSTOMER_MARGIN 87 F087_LIAB_SPECIAL_DEPOSIT_OTHERS 88 F088_LIAB_DEPOSIT_FROM_OTHER_BANKS 89 F089_LIAB_DEPOSIT_FROM_OTHER_BANKS_TZ 90 F090_LIAB_DEPOSIT_FROM_OTHER_BANKS_ABROAD 91 F091_LIAB_BANKERS_CHEQUES_AND_DRAFTS 92 F092_LIAB_PAYMENT_ORDERS 93 F093_LIAB_BORROWINGS 94 F094_LIAB_BORROWINGS_BOT 95 F095_LIAB_BORROWINGS_BOT_DEDISCOUNT 96 F096_LIAB_BORROWINGS_BOT_BORROWINGS 97 F097_LIAB_BORROWINGS_FR_OTHER_BANKS_AND_FI_TZ 98 F098_LIAB_BORROWINGS_OTHERS_TZ 99 F099_LIAB_BORROWINGS_BANKS_ABROAD 100 F100_LIAB_BORROWINGS_OTHERS_ABROAD 101 F101_LIAB_SUBORDINATED_DEBT 102 F102_LIAB_ACCRUED_TAX_UNPAID_EXPENSES 103 F103_LIAB_ACCRUED_TAX_PAYABLE 104 F104_LIAB_ACCRUED_OTHER_TAXES 105 F105_LIAB_ACCRUED_INTEREST_PAYABLE_DEPOSITS 106 F106_LIAB_ACCRUED_INTEREST_PAYABLE_BORROWINGS 107 F107_LIAB_ACCRUED_INTEREST_PAYABLE_OTHERS 108 F108_LIAB_ACCRUED_DEPOSIT_INSURANCE_PREMIUM 109 F109_LIAB_ACCRUED_OTHER_EXPENSES 110 F110_LIAB_UNEARNED_INCOME_AND_OTHER_CREDITS 111 F111_LIAB_OUTSTANDING_ACCEPTANCES 112 F112_LIAB_INTERBRANCH_FLOAT_ITEMS 113 F113_LIAB_INTER_BRANCH_FLOAT_TO_30DAYS 114 F114_LIAB_INTER_BRANCH_FLOAT_31_60DAYS 115 F115_LIAB_INTER_BRANCH_FLOAT_61_90DAYS 116 F116_LIAB_INTER_BRANCH_FLOAT_91_18ODAYS 117 F117_LIAB_INTER_BRANCH_FLOAT_ABOVE_180DAYS 118 F118_LIAB_OTHER 119 F119_LIAB_OTHER_ACC_PAYABLE 120 F120_LIAB_OTHER_DIVIDENT_PAYABLE 121 F121_LIAB_OTHER_SUBSCRIPTIONS_PAYABLE 122 F122_LIAB_OTHER_WITHHOLDING_TAX_PAYABE 123 F123_LIAB_OTHER_MISCELLANEOUS 124 F124_LIAB_OTHER_SUNDRY 125 F125_LIAB_TOTAL 126 F126_CAPITAL_TOTAL 127 F127_CAPITAL_PAID_UP_SC 128 F128_CAPITAL_PAID_UP_ORDINARY_SC 129 F129_CAPITAL_PAID_UP_IRREDEEMABLE_NONCUM_PREF_SC 130 F130_CAPITAL_PAID_UP_SC_OTHER_PREF 131 F131_CAPITAL_OTHER 132 F132_CAPITAL_OTHER_SH_PR 133 F133_CAPITAL_OTHER_GRANTS 134 F134_CAPITAL_OTHER_GEN_RESERVES 135 F135_CAPITAL_OTHER_RE 136 F136_CAPITAL_OTHER_PL 137 F137_CAPITAL_OTHER_FA_REVALUATION 138 F138_CAPITAL_OTHER_RESERVES 139 F139_CAPITAL_LIABILITIES_AND_CAPITAL_SUM 140 F140_OTHERS_OUTSTANDING_LC 141 F141_OTHERS_OUTSTANDING_LC_SIGHT_IMPORT_LC 142 F142_OTHERS_OUTSTANDING_LC_USANCE 143 F143_OTHERS_OUTSTANDING_LC_DEFERRED 144 F144_OTHERS_OUTSTANDING_LC_DOMESTIC 145 F145_OTHERS_OUTSTANDING_LC_STAND_BY 146 F146_OTHERS_OUTSTANDING_LC_OTHER 147 F147_OTHERS_EXP_LETTERS 148 F148_OTHERS_OUTSTANDING_GUARANTEES 149 F149_OTHERS_INWARD_BILLS 150 F150_OTHERS_OUTWARD_BILLS 151 F151_OTHERS_FWD_EXCHANGE_BOUGHT 152 F152_OTHERS_FWD_EXCHANGE_SOLD 153 F153_OTHERS_TRUST_AND_FIDUCIARY_ACC 154 F154_OTHERS_ITEMS_FOR_SAFEKEEPING 155 F155_OTHERS_UNDERWRITING_ACC_UNSOLD 156 F156_OTHERS_LATE_DEPOSIT_PAYMENTS_RCVD 157 F157_OTHERS_TRAVELLERS_CHQ_UNSOLD 158 F158_OTHERS_SECURITIES_SOLD_UNDER_REPO 159 F159_OTHERS_SECURITIES_PURCHASED_USER_REPO 160 F160_OTHERS_UNDRAWN_BAL 161 F161_OTHERS_UNDRAWN_BAL_UNEXPIRED_OD 162 F162_OTHERS_UNDRAWN_BAL_TERM_LOANS_AND_ADVANCES 163 F163_OTHERS_OTHER 164 F164_OTHERS_TOTAL_CONTINGENT_ACCOUNTS 165 F165_OTHERS_TOTAL_RWA 166 F166_OTHERS_PRE_OPERATING_EXPENSES 167 GDP 168 INF 169 IBCM 170 LR 171 DR 172 MLA 173 EWE_InterestIncome 174 EWE_InterestExpense 175 EWE_NetInterestIncome 176 EWE_NonInterestIncome 177 EWE_BadDebtsWrittenOff 178 EWE_ProvBadDebts 179 EWE_NonInterestIncome.1 180 EWE_ExtraOrdinaryItems 181 EWE_PBT 182 EWE_Tax 183 EWE_PAT 184 EWE_AnnualizedPBT 185 EWE_AvgInterestBearingAsset 186 EWE_AnnualisedNetIncome 187 EWE_ReturnOnAvgAssets 188 EWE_NetInterestIncomeToAvgInterestBearingAssets 189 EWE_NonInterestIncomeToAvgInterestBearingAsset 190 EWE_EarningsRating 191 EWL_LIQASSET_CASH 192 EWL_LIQASSET_BOTBAL_SMR 193 EWL_LIQASSET_BOTBAL_CURR 194 EWL_LIQASSET_BOTBAL_OTHER 195 EWL_LIQASSET_DUE_FR_BANKS_DOMESTIC 196 EWL_LIQASSET_DUE_FR_BANKS_FOREIGN 197 EWL_LIQASSET_DUE_FR_BANKS_FOREIGN.1 198 EWL_LIQASSET_TBILLS 199 EWL_LIQASSET_TOTAL 200 EWL_LIQLIAB_CURR_ACC 201 EWL_LIQLIAB_SAVINGS 202 EWL_LIQLIAB_TIMEDEPOSIT 203 EWL_LIQLIAB_SPECIALDEPOSIT 204 EWL_LIQLIAB_BANK_DEPOSITS 205 EWL_LIQLIAB_CHQ_DRAFTS 206 EWL_LIQLIAB_BORROWINGS_FROM_BOT 207 EWL_LIQLIAB_BORROWINGS_FROM_TZ_BANKS 208 EWL_LIQLIAB_OTHER_BORROWING_TZ 209 EWL_LIQLIAB_BORROWING_ABROAD 210 EWL_LIQLIAB_ACCR_EXPENSES 211 EWL_LIQLIAB_PAYABLES 212 EWL_LIQLIAB_DIVIDENDS 213 EWL_LIQLIAB_SUBSCRIPTIONS 214 EWL_LIQLIAB_WHT 215 EWL_LIQLIAB_SUNDRY_CR 216 EWL_LIQLIAB_TOTAL 217 EWL_PUPLICATE_EWL_DEPOSITS_CURRENT 218 EWL_02. Savings Deposits 219 EWL_03. Time Deposits 220 EWL_04. Special Deposits 221 EWL_05. Deposits from other Banks in Tanzania 222 EWL_06. Deposits from other Banks Abroad 223 EWL_07. Matured Term Deposits 224 EWL_08. Dormant Accounts 225 EWL_09. TOTAL DEPOSITS 226 EWL_10. Core Deposits 227 EWL_Core Deposits to Total Funding 228 EWL_Liquid Assets to Demand Liabilities 229 EWL_Gross Loans to Total Deposits 230 EWL_LIQUIDITY RATING 231 EWAQ_Capital 232 EWAQ_NPL 233 EWAQ_GrossLoans 234 EWAQ_LargeExposures 235 EWAQ_NPLsNetOfProvisions 236 EWAQ_NPLsNetOfProvisions2CoreCapital 237 EWAQ_NPLs2GrossLoans 238 EWAQ_AssetsQualityRating 239 EWAQ_Loans 240 MLA_CLASS

The *weekly* data is as follwos:

1 INSTITUTIONCODE 2 REPORTINGDATE 3 DESC_NO 4 ASSET_AMOUNT 5 LIABILITY_AMT

### Interpretability

Interpretability of the model can be shown in two ways: (1) examining the importance of each input variable in prediction, (2) identifying any regimes or events that lead to significant changes in temporal dynamics.

1. **Analyzing variable importance**

We aggregate selection weights for each variable across our entire test set, recording the 10th, 50th and 90th percentiles of each sampling distribution. We present the results for its variable importance analysis.

2. **Identifying regimes & significant events**

Identifying sudden changes in temporal patterns can also be very useful, as temporary shifts can occur due to the presence of significant regimes or events. . 

### Conclusions 

We introduce TFT, a novel attention-based deep learning model for interpretable high-performance multi-horizon forecasting. To handle static covariates, a priori known inputs, and observed inputs effectively across a wide range of multi-horizon forecasting datasets, TFT uses specialized components. Specifically, these include: (1) sequence-to-sequence and attention-based temporal processing components that capture time-varying relationships at different timescales, (2) static covariate encoders that allow the network to condition temporal forecasts on static metadata, (3) gating components that enable skipping over unnecessary parts of the network, (4) variable selection to pick relevant input features at each time step, and (5) quantile predictions to obtain output intervals across all prediction horizons.

### References

1. Temporal Fusion Transformer (TFT; Lim et al, 2021) https://www.sciencedirect.com/science/article/pii/S0169207021000637
2. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting - https://www.youtube.com/watch?v=M7O4VqRf8s4
3. Interactive Python Notebook as a practical intro to the Lim et al TFT model:  https://github.com/dkgaraujo/TemporalFusionTransformers/blob/main/temporal_fusion_transformers.ipynb
4. Using machine learning for detecting liquidity risk in banks – ScienceDirect https://www.sciencedirect.com/science/article/pii/S2666827023000646?via%3Dihub#sec6
5. Using Machine Learning to Detect Liquidity Risk In Commercial Banks | Code Ocean https://codeocean.com/capsule/5775871/tree/v1
6. Gringado machine learning library for economics and finance https://bis-med-it.github.io/gingado

