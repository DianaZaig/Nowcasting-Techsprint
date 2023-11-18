#!/usr/bin/env python
# coding: utf-8

# ### Program 1 of 3
# ### Purpose: Data Preparations to obtain a weekly dataset for factor analysis.
# ### Inputs: collected data
# ### Outputs: final weekly dataset for factors analysis ( data\\final\\proposed_dataset_21082023.csv)
# ### Developer: Rweyemamu Barongo rbarongo@gmail.com, ribarongo@bot.go.tz, ribarongo@udsm.ac.tz
# 

# #### Install libraries

# In[12]:


#!pip install --upgrade xlrd


# In[13]:


#!pip install xlwt


# In[14]:


#!pip install IO


# In[17]:


#get_ipython().system('pip install pandas')


# In[18]:


#get_ipython().system('pip install matplotlib')


# #### Import libraries

# In[19]:

#preliminary libraries
import preparation
from preparation import install_missing_packages
install_missing_packages()
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
from io import StringIO    
from scipy.optimize import curve_fit
import warnings 
import config_unix_filesystem as config
from config_unix_filesystem import check_model_results_file
from config_unix_filesystem import check_if_exist_or_create_folders
from config_unix_filesystem import check_data_files_I


# #### inspect configurations and source data

# In[20]:


check_data_files_I()
check_if_exist_or_create_folders()
check_model_results_file()


# #### Load data from files

# **Import liquidity secondary data from source**

# 

# In[21]:


# Load weekly data from source to pandas dataframe 
df_new=pd.read_csv(config.weeklyDataWbook)
unnamedCols = [col for col in df_new.columns if 'Unnamed' in col]
df_new.drop(columns=unnamedCols, inplace=True)
# df1 = pd.read_excel(config.file1, sheet_name= [0, 1, 2, 3, 4, 5, 6])
# df2 = pd.read_excel(config.file2, sheet_name= [0, 1, 2, 3, 4, 5, 6])


# In[22]:


# #join weekly liquidity data pieces
# df_new = pd.concat([df1[0], df1[1],df1[2],df1[3],df1[4],df1[5],df1[6],df2[0], df2[1],df2[2],df2[3],df2[4],df2[5],df2[6]], axis=0)


# In[23]:


#view shape of dataframe
df_new.shape    # number of weekly submission rows before sampling commercial banks


# In[24]:


#view columns in dataframe
df_new.columns


# In[25]:


#df_new.to_csv(config.weeklyDataWbook)


# In[26]:


#view data 
df_new.head(3)


# In[27]:


# Explore shape
#df1[0].shape


# #### Sampling  (commercial banks existed during review period)
# #### Data was anonymised upon collection by changing bank codes from AXXX into BXXXX to conceal banks identity
# #### Banks outside the sample were all given a bank code = 'Unsampled'

# In[28]:


df_new['INSTITUTIONCODE'].unique()


# In[29]:


# filter one bank and attempt to pivot
mask = (df_new['INSTITUTIONCODE']=='B5114')&(df_new['REPORTINGDATE']=='2010-01-15')
df_new[mask].to_csv(config.weekly_dataset)


# In[30]:


# view sorted values by REPORTINGDATE and INSTITUTIONCODE
df_new.sort_values(by=['REPORTINGDATE','INSTITUTIONCODE'])
#df.sort_values(by=['col1', 'col2'])


# #### Drop Rows With INSTITUTIONCODE='Unsampled' to Remain With Targeted Sample

# In[31]:


df_new = df_new.dropna(subset=['INSTITUTIONCODE'])
df_new = df_new[df_new['INSTITUTIONCODE'] != 'Unsampled']
#'Unsampled' is an institutioncode assigned to all institutions out of sample


# In[32]:


# Export dataframe to csv format and save in disk, for analysis of intermediate outputs
df_new.to_csv(config.file3)


# In[33]:


#import exported file 
dfX = pd.read_csv(config.file3)   #one dataframe with all liquidity data
unnamedCols2 = [col for col in dfX.columns if 'Unnamed' in str(col)]
dfX.drop(columns=unnamedCols2, inplace=True)


# In[34]:


#view data 
dfX.head(5)


# In[35]:


#view columns
dfX.columns


# #### Derive Amount Variable fom Asset and Liability Variables

# In[36]:


#computer AMOUNT variable from ASSET_AMOUNT and LIABILITY_AMOUNT
mask1 = (dfX['DESC_NO'] <= 12) 
dfX['AMOUNT']=np.where(mask1,dfX['LIABILITY_AMT'],dfX['ASSET_AMOUNT'])


# In[37]:


#dfX.head()
mask = (dfX['INSTITUTIONCODE']=='B5412')&(dfX['REPORTINGDATE']=='2010-01-15')
dfX[mask].sort_values(by=['DESC_NO'])


# #### Data Transformation: Pivot Data into a Time-Series Dataset Format

# In[38]:


dfX_01 = dfX.pivot_table(index=['INSTITUTIONCODE','REPORTINGDATE'], columns=['DESC_NO'], values='AMOUNT')
dfX_01 


# #### Data Annotation (Weekly Liquidity Dataset)

# In[39]:


#df_annot[['SN','LABEL_NAME']]
dfX_01.rename(columns={1: "01_CURR_ACC", 
                    2: "02_TIME_DEPOSIT",
                    3: "03_SAVINGS",
                    4: "04_OTHER_DEPOSITS",
                    5: "05_BANKS_DEPOSITS",
                    6: "06_BORROWING_FROM_PUBLIC",
                    7: "07_INTERBANKS_LOAN_PAYABLE",
                    8: "08_CHEQUES_ISSUED",
                    9: "09_PAY_ORDERS",
                    10:"10_FOREIGN_DEPOSITS_AND_BORROWINGS",
                    11:"11_OFF_BALSHEET_COMMITMENTS",
                    12:"12_OTHER_LIABILITIES",
                    13:"13_CASH",
                    14:"14_CURRENT_ACC",
                    15:"15_SMR_ACC",
                    16:"16_FOREIGN CURRENCY",
                    17:"17_OTHER DEPOSITS",
                    18:"18_BANKS_TZ",
                    19:"19_BANKS_ABROAD",
                    20:"20_CHEQUES_ITEMS_FOR_CLEARING",
                    21:"21_INTERBANK_LOANS",
                    22:"22_TREASURY_BILLS",
                    23:"23_OTHER_GOV_SECURITIES",
                    24:"24_FOREIGN_CURRENCY",
                    25:"25_COMMERCIAL_BILLS",
                    26:"26_PROMISSORY_NOTES"
                    },inplace=True)


# In[40]:


#1.5  dataset with definitive column names
dfX_02 = dfX_01.copy()
dfX_02 = dfX_02.reset_index()
dfX_02


# In[41]:


#import config
print(config.file7)


# In[42]:


#backup of data in files form intermediate data analysis  
dfX_02.to_csv(config.file5)
dfX_03 = pd.read_csv(config.file5)
unnamedCols2 = [col for col in dfX_03.columns if 'Unnamed' in str(col)]
dfX_03.drop(columns=unnamedCols2, inplace=True)


# In[43]:


dfX_03  #.shape  #(31589, 28)


# #### Compute Derived Variables

# In[44]:


#1.6 transformed dataset to include computed variables with XX prefix
#transform
def totalDeposits(f01, f02, f03, f04):
  return f01 + f02 + f03 + f04
def totalLiquidLiab(f01,f05, f06, f07, f08, f09, f10, f11, f12):
  return f01 + f05 + f06 + f07 + f08 + f09 + f10 + f11 + f12
def botBalance(f14, f15,f16,f17):
  return f14 + f15 + f16 + f17
def balOtherBanks(f18,f19):
  return f18+f19
def totalLiqAssets(f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26):
  return f13 + f14 + f15 + f16 + f17 + f18 + f19 + f20 + f21 + f22 + f23 + f24 + f25 + f26

dfX_03['XX_CUSTOMER_DEPOSITS'] =dfX_03.apply(lambda x: totalDeposits(x['01_CURR_ACC'],x['02_TIME_DEPOSIT'],x['03_SAVINGS'],x['04_OTHER_DEPOSITS']), axis=1 )
dfX_03['XX_TOTAL_LIQUID_LIAB']=dfX_03.apply(lambda x: totalLiquidLiab(x['XX_CUSTOMER_DEPOSITS'],
                                                                         x['05_BANKS_DEPOSITS'],
                                                                         x['06_BORROWING_FROM_PUBLIC'],
                                                                         x['07_INTERBANKS_LOAN_PAYABLE'],
                                                                         x['08_CHEQUES_ISSUED'],
                                                                         x['09_PAY_ORDERS'],
                                                                         x['10_FOREIGN_DEPOSITS_AND_BORROWINGS'],
                                                                         x['11_OFF_BALSHEET_COMMITMENTS'],
                                                                         x['12_OTHER_LIABILITIES']), axis=1 )
dfX_03['XX_BOT_BALANCE']=dfX_03.apply(lambda x: botBalance(x['14_CURRENT_ACC'], x['15_SMR_ACC'], x['16_FOREIGN CURRENCY'], x['17_OTHER DEPOSITS']),axis=1)
dfX_03['XX_BAL_IN_OTHER_BANKS']=dfX_03.apply(lambda x: balOtherBanks(x['18_BANKS_TZ'],x['19_BANKS_ABROAD']),axis =1)
dfX_03['XX_TOTAL_LIQUID_ASSET']=dfX_03.apply(lambda x: totalLiqAssets(x["13_CASH"],
                                                                      x["14_CURRENT_ACC"],
                                                                      x["15_SMR_ACC"],
                                                                      x["16_FOREIGN CURRENCY"],
                                                                      x["17_OTHER DEPOSITS"],
                                                                      x["18_BANKS_TZ"],
                                                                      x["19_BANKS_ABROAD"],
                                                                      x["20_CHEQUES_ITEMS_FOR_CLEARING"],
                                                                      x["21_INTERBANK_LOANS"],
                                                                      x["22_TREASURY_BILLS"],
                                                                      x["23_OTHER_GOV_SECURITIES"],
                                                                      x["24_FOREIGN_CURRENCY"],
                                                                      x["25_COMMERCIAL_BILLS"],
                                                                      x["26_PROMISSORY_NOTES"]), axis=1)
dfX_03['XX_MLA'] = dfX_03.apply(lambda x: (100 * x['XX_TOTAL_LIQUID_ASSET']/x['XX_TOTAL_LIQUID_LIAB']), axis = 1)


   


# In[45]:


dfX_03.head()


# #### Data Cleaning

# ##### Confirm Presence of Required Unique Dates

# In[46]:


dfX_03.REPORTINGDATE.unique()


# In[47]:


#ensure date types in date values, export and import by parsing and saving dates in date format
dfX_03.to_csv(config.file5_III)
dfX_04 = pd.read_csv(config.file5_III, parse_dates=["REPORTINGDATE"])
unnamedCols2 = [col for col in dfX_04.columns if 'Unnamed' in str(col)]
dfX_04.drop(columns=unnamedCols2, inplace=True)


# ##### Ensure Appropriate Data Types for Numeric and Data Types

# In[48]:


#ensured all numeric data have appropriate type float64, date type has appropriate datetime64[ns], and institution code has appropriate type object
dfX_04.info()


# In[49]:


#
#dfX_04.set_index('REPORTINGDATE1').resample('W').interpolate()
dfX_04


# ##### Inspect dataframe to detect null variables

# In[50]:


# inspected presence of null variables in data
dfX_04.info()


# In[51]:


# displayed sample data
dfX_04.head(2)


# ##### Confirm shape of data against anticipated

# In[52]:


# checked shape of data interms of number of rows and columns
dfX_04.shape  #(31589, 32)
dfX_04


# ##### Add Missing Dates for Later Interpolation or Extrapolation of Values

# In[53]:


#Add all missing weekly instcodes and reporting dates, along with empty other columns for later interpolation or exterpolation
instCodes = list(dfX_04['INSTITUTIONCODE'].unique())

dt = pd.date_range("15-01-2010","31-12-2021",freq='W-FRI')   #get dates of fridays using date range

idx = pd.DatetimeIndex(dt)

dfX_05 = pd.DataFrame()

for i in range(len(instCodes)):
  dfX_05_tmp = dfX_04[(dfX_04['INSTITUTIONCODE']==instCodes[i])]
  dfX_05_tmp.set_index('REPORTINGDATE', inplace=True)
  dfX_05 = dfX_05_tmp.reindex(idx)  #add additional rows with dates that include missing fridays but with null columns 
  dfX_05 = pd.concat([dfX_05,dfX_05_tmp], axis = 0)   #concatenate data from all institutions

dfX_05.head(100)  #(625, 32)


# In[54]:


dfX_05_tmp = dfX_04[(dfX_04['INSTITUTIONCODE']=='A5015')]
dfX_05_tmp.set_index('REPORTINGDATE', inplace=True)
dfX_05_tmp.reindex(idx).shape  #add additional rows with dates that include missing fridays but with null columns 


# In[55]:


dfX_05


# ##### Fill Missing Data Values Using Cubic Interpolation

# In[56]:


#replacing missing data using interpolation, method=cubic
dt = pd.date_range("15-01-2010","31-12-2021",freq='W-FRI')
#dt = pd.date_range(dfX_04.index[0], dfX_04.index[-1],freq='W-FRI' )
idx = pd.DatetimeIndex(dt)
dfX_05_tmp = dfX_04[(dfX_04['INSTITUTIONCODE']=='A5015')]    
dfX_05_tmp.set_index('REPORTINGDATE', inplace=True)
dfX_05_tmp = dfX_05_tmp.reindex(idx)
dfX_05_tmp['13_CASH'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['14_CURRENT_ACC'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['15_SMR_ACC'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['16_FOREIGN CURRENCY'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['17_OTHER DEPOSITS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['18_BANKS_TZ'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['19_BANKS_ABROAD'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['20_CHEQUES_ITEMS_FOR_CLEARING'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['21_INTERBANK_LOANS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['22_TREASURY_BILLS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['23_OTHER_GOV_SECURITIES'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['24_FOREIGN_CURRENCY'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['25_COMMERCIAL_BILLS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['26_PROMISSORY_NOTES'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['XX_CUSTOMER_DEPOSITS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['XX_TOTAL_LIQUID_ASSET'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['XX_BOT_BALANCE'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['XX_BAL_IN_OTHER_BANKS'].interpolate(method='cubic', inplace=True)
dfX_05_tmp['XX_MLA'].interpolate(method='cubic', inplace=True)


# In[57]:


dfX_05_tmp.info()


# #### Explore Liquidity and Liquid Assets Trends To Confirm Validity of Data Against Real Liquidity Scenarios 2010-2021

# In[58]:


fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,6))
fig.suptitle('Liquidity and Liquid Asset Trends for Commercial Banks in Tanzania 2010-2021')

fig.text(0.5, -0.01, 'Reporting Date 2010-2021', ha='center')

mask = (dfX_04['XX_MLA']<101)
sns.lineplot(ax=axes[0,0], x=dfX_04['REPORTINGDATE'],y=dfX_04[mask]['XX_MLA'], label="MLA", color='b').set(xlabel=None)
axes[0,0].set_title('MLA')

sns.lineplot(ax=axes[0,1],x=dfX_04['REPORTINGDATE'],y=dfX_04['03_SAVINGS'],label="Savings",color='b').set(xlabel=None)
axes[0,1].set_title('Savings')

sns.lineplot(ax=axes[1,0],x=dfX_04['REPORTINGDATE'],y=dfX_04['01_CURR_ACC'], label="Current Acc", color='b').set(xlabel=None)
axes[1,0].set_title('Current Account')

sns.lineplot(ax=axes[1,1],x=dfX_04['REPORTINGDATE'],y=dfX_04['XX_TOTAL_LIQUID_ASSET'], label="Liquid Assets", color='b').set(xlabel=None)
axes[1,1].set_title('Liquid Assets')


# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Add space between the title and the plots
plt.tight_layout()

#axes[2,1].set_ylabels('MLA', size = 12)
plt.plot()


# In[59]:


#General trends of MLA
mask = (dfX_04['XX_MLA']<101)
sns.lineplot(x=dfX_04['REPORTINGDATE'],y=dfX_04[mask]['XX_MLA'])
plt.xlabel('Reporting Date')
plt.title('MLA Trends for Commercial Banks in Tanzania 2010-2021', pad=20)
plt.plot()
#General decrease in MLA levels overtime for all banks. With structure breaks in 2016 and 2019
#2016 attributed to government contractionary policies to reduce government deposits in commercial banks 


# In[60]:


mask = (dfX_04['INSTITUTIONCODE']=='B5015')
sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04[mask]['XX_MLA']) 
#on of bigg bank, a sharp decrease in liquidity for 2 big banks caused and in other banks as caused by contractionary liquidity policies of government
plt.xlabel('Reporting Date')
plt.title('MLA Trends for Bank B5015 in Tanzania 2010-2021', pad=20)
plt.plot()


# In[61]:


mask = (dfX_04['INSTITUTIONCODE']=='B5015')
sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04[mask]['03_SAVINGS']) 
plt.xlabel('Reporting Date')
plt.title('Savings Trends for Bank B5015 in Tanzania 2010-2021', pad=20)
plt.plot()


# In[62]:


mask = (dfX_04['INSTITUTIONCODE']=='B5912')
sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04[mask]['03_SAVINGS']) 
plt.xlabel('Reporting Date')
plt.title('Savings Trends for Bank B5912 in Tanzania 2010-2021', pad=20)
plt.plot()


# In[63]:


sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04['03_SAVINGS']) #ALL
plt.xlabel('Reporting Date')
plt.title('Savings Trends for Commercial Banks in Tanzania 2010-2021', pad=20)
plt.plot()


# In[64]:


sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04['01_CURR_ACC']) #ALL
plt.xlabel('Reporting Date')
plt.title('Current Account Trends for Commercial Banks in Tanzania 2010-2021', pad=20)
plt.plot()


# In[65]:


sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04['XX_TOTAL_LIQUID_ASSET']) #ALL
plt.xlabel('Reporting Date')
plt.title('Liquid Assets Trends for Commercial Banks in Tanzania 2010-2021', pad=20)
plt.plot()


# In[66]:


mask = (dfX_04['INSTITUTIONCODE']=='B5912')
sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04[mask]['XX_MLA']) 
plt.xlabel('Reporting Date')
plt.title('Liquidity (MLA) Trends for Bank B5912 in Tanzania 2010-2021', pad=20)
plt.plot()


# In[67]:


mask = (dfX_04['INSTITUTIONCODE']=='B5413')
sns.lineplot(x=dfX_04[mask]['REPORTINGDATE'],y=dfX_04[mask]['XX_MLA']) 
plt.xlabel('Reporting Date')
plt.title('Liquidity (MLA) Trends for Bank B5413 in Tanzania 2010-2021', pad=20)
plt.plot()


# In[68]:


dfX_04['INSTITUTIONCODE'].unique()


# #### Interpolation of Missing Values using Cubic Interpolation

# In[69]:


#dfX_05.head()


# In[70]:


#Interpolate intemediary data for individual institutions  
#Note to customize idx within time of institutional data availability  
#interpolated missing values using cubic method

dfX_05 = pd.DataFrame()
for i in range(len(instCodes)):
  maskx= (dfX_03['INSTITUTIONCODE']==instCodes[i])
  dt = pd.date_range(dfX_03[maskx]['REPORTINGDATE'].min(),dfX_03[maskx]['REPORTINGDATE'].max(),freq='W-FRI')
  idx = pd.DatetimeIndex(dt)
  dfX_05_tmp = dfX_04[(dfX_04['INSTITUTIONCODE']==instCodes[i])]     
  dfX_05_tmp.set_index('REPORTINGDATE', inplace=True)
  dfX_05_tmp = dfX_05_tmp.reindex(idx)
  dfX_05_tmp['INSTITUTIONCODE']=dfX_05_tmp['INSTITUTIONCODE'].apply(lambda x: x  if x==instCodes[i] else instCodes[i])
  dfX_05_tmp['01_CURR_ACC'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['02_TIME_DEPOSIT'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['03_SAVINGS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['04_OTHER_DEPOSITS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['05_BANKS_DEPOSITS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['06_BORROWING_FROM_PUBLIC'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['07_INTERBANKS_LOAN_PAYABLE'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['08_CHEQUES_ISSUED'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['09_PAY_ORDERS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['10_FOREIGN_DEPOSITS_AND_BORROWINGS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['11_OFF_BALSHEET_COMMITMENTS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['12_OTHER_LIABILITIES'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['13_CASH'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['14_CURRENT_ACC'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['15_SMR_ACC'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['16_FOREIGN CURRENCY'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['17_OTHER DEPOSITS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['18_BANKS_TZ'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['19_BANKS_ABROAD'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['20_CHEQUES_ITEMS_FOR_CLEARING'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['21_INTERBANK_LOANS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['22_TREASURY_BILLS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['23_OTHER_GOV_SECURITIES'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['24_FOREIGN_CURRENCY'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['25_COMMERCIAL_BILLS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['26_PROMISSORY_NOTES'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['XX_CUSTOMER_DEPOSITS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['XX_TOTAL_LIQUID_ASSET'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['XX_BOT_BALANCE'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['XX_BAL_IN_OTHER_BANKS'].interpolate(method='cubic', inplace=True)
  dfX_05_tmp['XX_MLA'].interpolate(method='cubic', inplace=True)
  x = dfX_05_tmp.isna().any().sum()
  if x > 0:
    print("{}:{}".format(x,instCodes[i]))
    print(dfX_05_tmp.info())
  dfX_05_tmp.reset_index(inplace=True)
  dfX_05_tmp.rename(columns={'index':'REPORTINGDATE'}, inplace=True)
  #dfX_05_tmp2=dfX_05_tmp.index_reset()
  dfX_05 = pd.concat([dfX_05, dfX_05_tmp], axis=0)


# In[71]:


#check shape of resulting dataset
dfX_05.shape  #(31621, 32)


# In[72]:


#check shape of individual banks with varied years of licensing date eg B5919 (year 2018) had 200+ records, B5015/B5115 (prior to 2010) had 650 records
dfX_05[dfX_05['INSTITUTIONCODE']=='B5115'].shape


# In[73]:


dfX_05.info()   #checked existance of null variables and data types in a dataset


# In[74]:


dfX_05[dfX_05['INSTITUTIONCODE']=='B5015'].tail(100)


# In[75]:


#Variable labeling: Already done during anonymisation   
dfX_05.to_csv(config.file5_II)
dfX_06 = pd.read_csv(config.file5_II, parse_dates=["REPORTINGDATE"])
unnamedCols2 = [col for col in dfX_06.columns if 'Unnamed' in str(col)]
dfX_06.drop(columns=unnamedCols2, inplace=True)

banks = pd.read_csv(config.file6)
unnamedCols2 = [col for col in banks.columns if 'Unnamed' in str(col)]
banks.drop(columns=unnamedCols2, inplace=True)


# In[76]:


def encode(initialCode):
  mask =  banks['INSTITUTIONCODE']==initialCode
  return banks[mask]['INSTITUTIONCODE2'].values[0]
   


# In[77]:


#
banks[['INSTITUTIONCODE','INSTITUTIONCODE2']]


# In[78]:


# save 
dfX_06.to_csv(config.file7)
dfX_07 = pd.read_csv(config.file7, parse_dates=["REPORTINGDATE"])
unnamedCols2 = [col for col in dfX_07.columns if 'Unnamed' in str(col)]
dfX_07.drop(columns=unnamedCols2, inplace=True)


# In[79]:


dfX_07.head(3)


# #### Building Monthly Dataset and Interpolation into Weekly Dataset

# In[80]:


#interpolate missing columns from monthly data
# dfX_monthly = pd.read_csv(dataWbook,parse_dates=["BSH_REPORTINGDATE"])
# unnamedCols2 = [col for col in dfX_monthly.columns if 'Unnamed' in str(col)]
# dfX_monthly.drop(columns=unnamedCols2, inplace=True)
# dfX_monthly.shape

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    # interpolate missing columns from monthly data
    dfX_monthly = pd.read_csv(config.dataWbook, parse_dates=["BSH_REPORTINGDATE"])
    unnamedCols2 = [col for col in dfX_monthly.columns if 'Unnamed' in str(col)]
    dfX_monthly.drop(columns=unnamedCols2, inplace=True)

# Warnings are suppressed within the context manager
dfX_monthly.shape


# In[81]:


#dfX_monthly.columns
[col for col in dfX_monthly.columns if 'MLA' in str(col)]


# In[82]:


#
dfX_monthly.head(2)


# In[83]:


#Interpolate to weekly
maskx= (dfX_03['INSTITUTIONCODE']=='B5919')
#dt = pd.date_range("15-01-2010","31-12-2021",freq='W-FRI')
dt = pd.date_range(dfX_03[maskx]['REPORTINGDATE'].min(),dfX_03[maskx]['REPORTINGDATE'].max(),freq='W-FRI')
#dt1 = pd.date_range("15-01-2010","31-12-2021",freq='M')
dt1 = pd.date_range(dfX_03[maskx]['REPORTINGDATE'].min(),dfX_03[maskx]['REPORTINGDATE'].max(),freq='M')
arr_dt = np.array(dt)
arr_dt1 = np.array(dt1)
dt_combined = np.append(dt, dt1)
dt_combined = np.sort(dt_combined)


# In[84]:


#
dt_combined


# In[85]:


#
dfX_03[maskx]['REPORTINGDATE']


# #### Generate missing dates of monthly data, add weekly dates, and interpolate missing data for all banks

# In[86]:


#2.2 Filter columns
#  Generate missing dates of monthly data, add weekly dates, and interpolate missing data
#instCodes2A = list(dfX_monthly['INSTITUTIONCODE'].unique())
instCodes2A = list(banks['INSTITUTIONCODE'].values)
instCodes2B = list(banks['INSTITUTIONCODE2'].values)

dfX_monthly2 = dfX_monthly[['BSH_REPORTINGDATE','INSTITUTIONCODE','F077_ASSETS_TOTAL','F125_LIAB_TOTAL','EWAQ_GrossLoans','EWAQ_Capital',
                            'EWAQ_NPL','EWAQ_NPLsNetOfProvisions','EWAQ_NPLsNetOfProvisions2CoreCapital','INF','LR','DR','IBCM','GDP','EWL_LIQUIDITY RATING','MLA','MLA_CLASS']].copy()
dfX_monthly2.rename(columns={'BSH_REPORTINGDATE':'REPORTINGDATE'}, inplace=True)
idx = pd.DatetimeIndex(dt_combined)

#dfX_monthly_tmp = dfX_monthly2[(dfX_monthly2['INSTITUTIONCODE']=='B5919')]
#dt = pd.date_range(dfX_monthly2['REPORTINGDATE'].min(),dfX_monthly2['REPORTINGDATE'].max(),freq='W-FRI')
#dfX_monthly_tmp.set_index('REPORTINGDATE', inplace=True)

dfX_monthly_01 = pd.DataFrame()
num_rows = 0
for i in range(len(instCodes2A)):
    dfX_monthly_tmp = dfX_monthly2[((dfX_monthly2['INSTITUTIONCODE']==instCodes2A[i]) | (dfX_monthly2['INSTITUTIONCODE']==instCodes2B[i]))]
    dt = pd.date_range(dfX_monthly_tmp['REPORTINGDATE'].min(),dfX_monthly_tmp['REPORTINGDATE'].max(),freq='W-FRI')
    dt1 = pd.date_range(dfX_monthly_tmp['REPORTINGDATE'].min(),dfX_monthly_tmp['REPORTINGDATE'].max(),freq='M')

    dfX_monthly_tmp.set_index('REPORTINGDATE', inplace=True)
    #masky= (dfX_monthly2['INSTITUTIONCODE']==instCodes[i])
    arr_dt = np.array(dt)
    arr_dt1 = np.array(dt1)
    dt_combined = np.append(dt, dt1)
    dt_combined = np.sort(dt_combined)
    idx = pd.DatetimeIndex(dt_combined)
    dfX_monthly_tmp2 = dfX_monthly_tmp.reindex(idx)    
    dfX_monthly_tmp2.reset_index(inplace=True)
    dfX_monthly_tmp2.rename(columns={'index':'REPORTINGDATE'}, inplace=True)
    #['INSTITUTIONCODE'] = dfX_monthly_tmp2['INSTITUTIONCODE'].apply(lambda x: encodeBankNames(x))  #already encoded    
    dfX_monthly_tmp2['INSTITUTIONCODE'] = dfX_monthly_tmp2['INSTITUTIONCODE'].apply(lambda x: x  if x==instCodes2B[i] else instCodes2B[i])       
    dfX_monthly_tmp2['F077_ASSETS_TOTAL'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['F125_LIAB_TOTAL'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWAQ_GrossLoans'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWAQ_Capital'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWAQ_NPL'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWAQ_NPLsNetOfProvisions'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWAQ_NPLsNetOfProvisions2CoreCapital'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['INF'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['LR'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['DR'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['IBCM'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['GDP'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['MLA'].interpolate(method='cubic', inplace=True)
    dfX_monthly_tmp2['EWL_LIQUIDITY RATING'].interpolate(method='cubic', inplace=True) 
    #print("code is {}, shape is {}".format(instCodes2A[i] + " " + instCodes2B[i], dfX_monthly_tmp2.shape))   
    num_rows = num_rows +  dfX_monthly_tmp2.shape[0]
    dfX_monthly_01 = pd.concat([dfX_monthly_01, dfX_monthly_tmp2], axis=0)
    


# In[87]:


#
num_rows #24945


# In[88]:


#
dfX_monthly_01['INSTITUTIONCODE'].unique()


# In[89]:


#
#interpolate.interp1d(x, y, fill_value='extrapolate')
dfX_monthly_01.columns


# In[90]:


#
dfX_monthly_01[(dfX_monthly_01['INSTITUTIONCODE']=='B5919')].head(5)


# In[91]:


#!pip install IO


# In[92]:


#
dfX_monthly_02 = dfX_monthly_01.copy()
dfX_monthly_02


# #### Extrapolation of values that are outside the range

# In[93]:


#2.4 indexing 
# Temporarily remove dates and make index numeric
#source:https://itecnote.com/tecnote/python-extrapolate-pandas-dataframe/
# df = dfX_monthly_01[['F077_ASSETS_TOTAL',
#        'F125_LIAB_TOTAL', 'EWAQ_GrossLoans', 'EWAQ_Capital', 'EWAQ_NPL',
#        'EWAQ_NPLsNetOfProvisions', 'EWAQ_NPLsNetOfProvisions2CoreCapital',
#        'LR', 'DR', 'IBCM', 'GDP','INF', 'EWL_LIQUIDITY RATING','MLA']].copy() 
# di = df.index
# df = df.reset_index().drop('index', 1)


df = dfX_monthly_01[['F077_ASSETS_TOTAL',
       'F125_LIAB_TOTAL', 'EWAQ_GrossLoans', 'EWAQ_Capital', 'EWAQ_NPL',
       'EWAQ_NPLsNetOfProvisions', 'EWAQ_NPLsNetOfProvisions2CoreCapital',
       'LR', 'DR', 'IBCM', 'GDP','INF', 'EWL_LIQUIDITY RATING','MLA']].copy() 
di = df.index
df = df.reset_index(drop=True)


# In[94]:


#
#from scipy.optimize import curve_fit
# Function to curve fit to the data
def func(x, a, b, c, d):
    return a * (x ** 3) + b * (x ** 2) + c * x + d

# Initial parameter guess, just to kick off the optimization
guess = (0.5, 0.5, 0.5, 0.5)
fit_df = df.dropna()
# Place to store function parameters for each column
col_params = {}

# Curve fit each column
for col in fit_df.columns:
    # Get x & y
    x = fit_df.index.astype(float).values
    y = fit_df[col].values
    # Curve fit column and get curve parameters
    params = curve_fit(func, x, y, guess)
    # Store optimized parameters
    col_params[col] = params[0]

# Extrapolate each column
for col in df.columns:
    # Get the index values for NaNs in the column
    x = df[pd.isnull(df[col])].index.astype(float).values
    # Extrapolate those points with the fitted function
    df[col][x] = func(x, *col_params[col])


# In[95]:


#68
# Put date index back
df.index = di



# In[96]:


#69
df.info()


# In[97]:


#70
#info and check null variables
dfX_monthly_03 = df.copy()
dfX_monthly_03[['REPORTINGDATE','INSTITUTIONCODE']]=dfX_monthly_01[['REPORTINGDATE','INSTITUTIONCODE']]


# In[98]:


#
#2.5 extrapolated values
dfX_monthly_03 = dfX_monthly_03[['REPORTINGDATE','INSTITUTIONCODE','F077_ASSETS_TOTAL',
       'F125_LIAB_TOTAL', 'EWAQ_GrossLoans', 'EWAQ_Capital', 'EWAQ_NPL',
       'EWAQ_NPLsNetOfProvisions', 'EWAQ_NPLsNetOfProvisions2CoreCapital',
       'LR', 'DR', 'IBCM', 'GDP', 'INF','EWL_LIQUIDITY RATING','MLA']]


# In[99]:


#
#check if extrapolated values donot have null values
dfX_monthly_03.info()


# In[100]:


#73
#interpolated and extrapolated
dfX_monthly_03.to_csv(config.file7i)


# #### Join Weekly with Monthly In Weekly Format

# In[101]:


dfX_08 = pd.merge(dfX_07, dfX_monthly_03, how="left", on=["REPORTINGDATE","INSTITUTIONCODE"])


# In[102]:


#check columns with NAN
column_with_nan = dfX_08.columns[dfX_08.isnull().any()]
for column in column_with_nan:
    print(column, dfX_08[column].isnull().sum())


# In[103]:


#Export data with null variables for analysis
dfX_08[dfX_08.isnull().any(axis=1)].to_csv(config.null_empty_for_analysis)


# #### EDA For Sampled Banks

# In[104]:


#79
maskz = (dfX_08['INSTITUTIONCODE'] == 'B5015') 
dfX_08[maskz].shape  #(645, 44)


# In[105]:


#80
maskz = (dfX_08['INSTITUTIONCODE'] == 'B5115') 
dfX_08[maskz].shape  #(645, 44) 


# In[106]:


#81
maskz = (dfX_08['INSTITUTIONCODE'] == 'B5919') 
dfX_08[maskz].shape  #(211, 44) 


# In[107]:


#82
maskz = (dfX_08['INSTITUTIONCODE'] == 'B5014') 
dfX_08[maskz].shape  #(645, 44)


# In[108]:


#83
mask_sampled_banks = ((dfX_08['INSTITUTIONCODE'] == 'B5015') | 
                      (dfX_08['INSTITUTIONCODE'] == 'B5115') |  
                       (dfX_08['INSTITUTIONCODE'] == 'B5919') |
                       (dfX_08['INSTITUTIONCODE'] == 'B5014')  )
dfX_08[mask_sampled_banks].shape 


# In[109]:


#84
dfX_08_4banks = dfX_08[mask_sampled_banks].dropna()


# In[110]:


#85
#dfX_08_4banks.isna().sum()
dfX_08_4banks['EWL_LIQUIDITY RATING'].unique()


# In[111]:


#86
dfX_08_4banks.info()


# In[112]:


#87
dfX_09 = dfX_08.dropna()


# In[113]:


#88
#check if MLA values from weekly (calculated) and monthly (interpolated) are close to eath other
dfX_09[['XX_MLA','MLA']]


# In[114]:


#89
dfX_09['EWL_LIQUIDITY RATING'] = dfX_09['EWL_LIQUIDITY RATING'].apply(lambda x: int(x))


# In[115]:


#90
dfX_09['EWL_LIQUIDITY RATING'].unique()


# In[116]:


#91
dfX_09['EWL_LIQUIDITY RATING'].shape #(20903,)


# In[117]:


#92
#dfX_08 = pd.merge(dfX_07, dfX_monthly_03, how="left", on=["REPORTINGDATE","INSTITUTIONCODE"])
#dfX_monthly_01[["REPORTINGDATE","INSTITUTIONCODE"]]


# In[118]:


#93
#dfX_08.shape  #(38750, 44)
#dfX_07.shape  #(38750, 44)
#dfX_monthly_01.shape #(29222, 14)


# #### Classifying banks in MLA Classes of Liquidity Risk Ratings 1 to 5 adopted from Banking Supervision Manual

# In[119]:


#94
#convert XX_MLA into classes 
mlaValue = 0
def getMlaClass(mlaValue):
    mlaClass = 1
    if mlaValue > 40:
        mlaClass = 1
    elif  ((mlaValue >  30) and (mlaValue <=  40)):
        mlaClass = 2
    elif  ((mlaValue >  20) and (mlaValue <=  30)):
        mlaClass = 3
    elif  ((mlaValue >  15) and (mlaValue <=  20)):
        mlaClass = 4
    elif  (mlaValue <=  15):
        mlaClass = 5
    #elif  (mlaValue <=  0.20):
    #  mlaClass = 4
    return mlaClass

dfX_09['MLA_CLASS2'] = dfX_09['MLA'].apply(lambda x: getMlaClass(x))
dfX_09['XX_MLA_CLASS2'] = dfX_09['XX_MLA'].apply(lambda x: getMlaClass(x))


# In[120]:


dfX_09.head()


# In[121]:


dfX_09['MLA_CLASS2'].unique()


# In[122]:


dfX_09['XX_MLA_CLASS2'].unique()


# In[123]:


dfX_09['MLA_CLASS2'].value_counts()


# In[124]:


dfX_09['XX_MLA_CLASS2'].value_counts()


# #### Liquidity Risk Distribution - Less than 20% indicate High Liquidity Risk 

# In[125]:


dfX_09[['MLA']][(dfX_09['MLA']>= 0) & (dfX_09['MLA']<= 100)].hist()
plt.title('MLA -Interpolated')
plt.ylabel('Observations in Submitted Returns')
plt.xlabel('MLA Score')
plt.plot()


# In[126]:


dfX_09[['XX_MLA']][(dfX_09['XX_MLA']>= 0) & (dfX_09['XX_MLA']<= 100)].hist()
plt.title('MLA -Computed')
plt.ylabel('Observations in Submitted Returns')
plt.xlabel('MLA Score')
plt.plot()


# In[127]:


def confusion_matrix(df: pd.DataFrame, col1: str, col2: str):
    """
    Given a dataframe with at least
    two categorical columns, create a 
    confusion matrix of the count of the columns
    cross-counts
    
    use like:
    
    >>> confusion_matrix(test_df, 'actual_label', 'predicted_label')
    """
    return (
            df
            .groupby([col1, col2])
            .size()
            .unstack(fill_value=0)
            )


# In[128]:


confusion_matrix(dfX_09, 'XX_MLA_CLASS2', 'MLA_CLASS2')
#two series of calculated and interpolated MLA are close


# #### Export Resulted Dataset as the Proposed Dataset in data\final\proposed_dataset_21082023.csv

# In[129]:


#confirm expected data shape
dfX_09.shape


# In[130]:


dfX_09.columns


# In[131]:


dfX_09.head()


# In[132]:


#Export to file for use in factors analysis 
dfX_09.to_csv(config.weekly_dataset)


# In[ ]:




