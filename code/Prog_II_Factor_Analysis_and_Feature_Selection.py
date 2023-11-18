#!/usr/bin/env python
# coding: utf-8

# ### Program 2 of 3
# ### Purpose: This code performs factor analysis and feature selection
# ### Inputs: final weekly dataset
# ### Outputs: Selected Liquidity Risk Factors and Features 
# ### Developer: Rweyemamu Barongo rbarongo@gmail.com, ribarongo@bot.go.tz, ribarongo@udsm.ac.tz

# #### Install libraries

# In[ ]:


#get_ipython().system('pip install --upgrade xlrd')


# In[ ]:


#get_ipython().system('pip install sklearn')


# In[ ]:


#get_ipython().system('pip install scikit-learn')


# #### Import libraries

# In[ ]:


#import local libraries
#preliminary libraries
import preparation
from preparation import install_missing_packages
install_missing_packages()
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import date
import warnings
#configuration of data files
import config_unix_filesystem as config
from config_unix_filesystem import check_model_results_file
from config_unix_filesystem import check_if_exist_or_create_folders
from config_unix_filesystem import check_data_files_II
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt


try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False
    
if IN_COLAB==True:    
    from google.colab import drive
    drive.mount('/content/gdrive/', force_remount=True)
    sys.path.append(config.path_to_module)


# In[ ]:





# #### inspect configurations and dataset

# In[ ]:


check_data_files_II()
check_if_exist_or_create_folders()
check_model_results_file()


# #### Load data from files

# In[ ]:


df_new = pd.read_csv(config.weekly_dataset, parse_dates=["REPORTINGDATE"])
unnamedCols2 = [col for col in df_new.columns if 'Unnamed' in str(col)]
df_new.drop(columns=unnamedCols2, inplace=True)

dataset = pd.read_csv(config.dataWbook) #, sheet_name="finalDataset_240622", header=1)
unnamedCols = [col for col in dataset.columns if 'Unnamed' in col]
dataset.drop(columns=unnamedCols, inplace=True)
dataset.shape 


# In[ ]:


# dfX_monthly.shape 


# #### Explore dataset

# In[ ]:


df_new.head()


# In[ ]:


#Number of data rows and columns
sno_val = df_new.shape
sno_val


# In[ ]:


#columns names
df_new.columns


# #### Compute Derived Variables

# In[ ]:


df_new['CORE_DEPOSITS']=df_new.apply(lambda x: x['01_CURR_ACC']+x['03_SAVINGS']+x['04_OTHER_DEPOSITS'], axis=1)
df_new['TOTAL_DEPOSITS']=df_new.apply(lambda x: x['05_BANKS_DEPOSITS']+x['XX_CUSTOMER_DEPOSITS'], axis=1)
df_new['GL_TO_TOTAL_FUNDING']=df_new.apply(lambda x: x['EWAQ_GrossLoans']/(x['F125_LIAB_TOTAL']+0.0001), axis=1)
df_new['CD_TO_TOTAL_FUNDING']=df_new.apply(lambda x: x['CORE_DEPOSITS']/(x['F125_LIAB_TOTAL']+0.0001), axis=1)
df_new['CD_TO_TOTAL_ASSET']=df_new.apply(lambda x: x['CORE_DEPOSITS']/(x['F077_ASSETS_TOTAL']+0.0001), axis=1)
df_new['CD_TO_TOTAL_DEPOSIT']=df_new.apply(lambda x: x['CORE_DEPOSITS']/(x['TOTAL_DEPOSITS']+0.0001), axis=1)
df_new['LiqAsset2DemandLiab']=df_new.apply(lambda x: x['XX_TOTAL_LIQUID_ASSET']/(x['CORE_DEPOSITS'] + x['02_TIME_DEPOSIT']+0.0001), axis=1)
df_new['ExcessShortTLiab2LongTAsset']=df_new.apply(lambda x: (x['CORE_DEPOSITS'] + x['02_TIME_DEPOSIT']-x['XX_TOTAL_LIQUID_ASSET'])/(x['F077_ASSETS_TOTAL'] - x['XX_TOTAL_LIQUID_ASSET']+0.0001), axis=1)
df_new['CD_TO_TOTAL_ASSET']=df_new.apply(lambda x: x['CORE_DEPOSITS']/(x['F077_ASSETS_TOTAL']+0.0001), axis=1)
df_new['GL_TO_TOTAL_DEPOSITS']=df_new.apply(lambda x: x['EWAQ_GrossLoans']/(x['CORE_DEPOSITS'] + x['02_TIME_DEPOSIT']+0.0001), axis=1)
df_new['LIQASSET2TOTALASSET']=df_new.apply(lambda x: x['XX_TOTAL_LIQUID_ASSET']/(x['F077_ASSETS_TOTAL'] +0.0001), axis=1)
df_new['BANKSIZE']= np.log10(df_new['F077_ASSETS_TOTAL'])
df_new['LOAN2DEPOSIT']= df_new.apply(lambda x: x['EWAQ_GrossLoans']/(x['CORE_DEPOSITS'] + x['02_TIME_DEPOSIT'] +0.0001), axis=1)
df_new['LIQASSET2DEPOSIT']= df_new.apply(lambda x: x['XX_TOTAL_LIQUID_ASSET']/(x['CORE_DEPOSITS'] + x['02_TIME_DEPOSIT']+0.0001), axis=1)
df_new['CURRENTRATIO']= df_new.apply(lambda x: x['XX_TOTAL_LIQUID_ASSET']/(x['XX_TOTAL_LIQUID_LIAB'] + 0.0001), axis=1)
df_new['LIQASSET2TOTALASSET']= df_new.apply(lambda x: x['XX_TOTAL_LIQUID_ASSET']/(x['F077_ASSETS_TOTAL'] + 0.0001), axis=1)
df_new['VOLATILEDEPOSITS2LIAB']= df_new.apply(lambda x: (x['XX_TOTAL_LIQUID_ASSET']-x['CORE_DEPOSITS'])/(x['F125_LIAB_TOTAL'] + 0.0001), axis=1)
df_new['LOAN2ASSETS']= df_new.apply(lambda x: x['EWAQ_GrossLoans']/(x['F077_ASSETS_TOTAL']+0.0001), axis=1)
df_new['DOMESTICDEPOSIT2ASSETS']= df_new.apply(lambda x: (x['TOTAL_DEPOSITS']-x['10_FOREIGN_DEPOSITS_AND_BORROWINGS'])/(x['F077_ASSETS_TOTAL']+0.0001), axis=1)
df_new['LOAN2COREDEPOSIT']= df_new.apply(lambda x: x['EWAQ_GrossLoans']/(x['CORE_DEPOSITS']+0.0001), axis=1)
df_new['BOTBAL2TOTALDEPOSIT']= df_new.apply(lambda x: x['XX_BOT_BALANCE']/(x['TOTAL_DEPOSITS']+0.0001), axis=1)



# In[ ]:


#Explore new dataset
df_new.head(5)


# #### Factors Analysis Using Correlation

# In[ ]:


df_new


# In[ ]:


df_new.iloc[:,2:]


# In[ ]:


df_new.iloc[:,2:].corr()


# In[ ]:


df_new.shape


# In[ ]:


df_factors_cor = df_new.iloc[:,2:].corr()
df_factors_cor


# In[ ]:


df_factors_cor.to_csv(config.factors_corr)


# In[ ]:


#visualize correlation of all variables for all institutions
plt.figure(figsize=(30,30))
sns.heatmap(df_new.iloc[:,2:].corr(),annot=True, vmin = 0.2, vmax=1, cmap="Greens", fmt='0.1')
plt.title('Significant Correlation of All Variables for Commercial Banks in Tanzanian 2010-2021', pad=20)
plt.xlabel('Liquidity Risk Variables')
plt.ylabel('Liquidity Risk Variables')
plt.plot()


# In[ ]:


plt.figure(figsize=(30,30))
sns.heatmap(df_new.iloc[:,2:][df_new['INSTITUTIONCODE']=='B5015'].corr(),annot=True, vmin = 0.2, vmax=1, cmap="Greens", fmt='0.1')
plt.title('Signifiant Positive Correlation of All Variables for Bank B5015 in Tanzanian 2010-2021', pad=20)
plt.xlabel('Liquidity Risk Variables')
plt.ylabel('Liquidity Risk Variables')
plt.plot()



# In[ ]:


plt.figure(figsize=(40,40))
sns.heatmap(df_new.iloc[:,2:].corr(),annot=True, vmin = -1, vmax=-0.2, cmap="Greens", fmt='0.1')
plt.title('Significant Negative Correlation of Variables for Commercial Banks in Tanzanian 2010-2021', pad=20)
plt.xlabel('Liquidity Risk Variables')
plt.ylabel('Liquidity Risk Variables')
plt.plot()


# **Analysis for Factors Identified In Literature**

# In[ ]:


df_new.columns


# In[ ]:


df_factors = df_new[['CURRENTRATIO',
                         'LIQASSET2DEPOSIT',
                         'TOTAL_DEPOSITS',
                         'CORE_DEPOSITS',
                         'LOAN2DEPOSIT',
                         'VOLATILEDEPOSITS2LIAB',
                         'BOTBAL2TOTALDEPOSIT',
                         'LOAN2COREDEPOSIT',
                         'DOMESTICDEPOSIT2ASSETS',
                         'BANKSIZE',
                         'GDP',
                         'INF',
                         'LOAN2ASSETS',
                         'EWAQ_NPL',
                         'EWAQ_NPLsNetOfProvisions',
                         'EWAQ_NPLsNetOfProvisions2CoreCapital',
                         'LR',
                         'CD_TO_TOTAL_ASSET',
                         'LIQASSET2TOTALASSET',
                         'ExcessShortTLiab2LongTAsset']]


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df_factors.corr(method='spearman', min_periods=1),annot=True, vmin = -1, vmax=1, cmap="Greys", fmt='0.1')
plt.title('Correlation Analysis of Liquidity Riks Factors')
plt.xlabel('Liquidity Risk Factors')
plt.ylabel('Liquidity Risk Factors')
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(df_factors.corr(method='spearman', min_periods=1),annot=True, vmin = -0.6, vmax=0.6, cmap="Greys", fmt='0.1')
plt.title('Correlation Analysis of Liquidity Riks Factors')
plt.xlabel("Liquidity Risk Factors' variables")
plt.ylabel("Liquidity Risk Factors' variables")
plt.show()


# 

# #### Detailed Pearson, Spearman, and Kendall Correlation for Factors Analysis on All, Big, Intermediate, and Small Banks

# In[ ]:


"""
Clustering of Banks was done in a separate process using Elbow method and KNN algorithm on
Total Assets of Banks as of 31 Dec 2021 that included 32 banks out of 38. 
Three clusters were obtained that included 2 big banks, 10 intermediate banks, and 20 small banks 
"""


# In[ ]:


#On monthly dataset - all banks
dataset_corr_pearson=dataset.iloc[:,2:-1].corr(method='pearson', min_periods=1)
dataset_corr_pearson = dataset_corr_pearson['MLA'].reset_index()
dataset_corr_pearson.rename(columns = {'MLA':'Pearson'}, inplace = True)
dataset_corr_spearman=dataset.iloc[:,2:-1].corr(method='spearman', min_periods=1)
dataset_corr_spearman  = dataset_corr_spearman['MLA'].reset_index()
dataset_corr_spearman.rename(columns = {'MLA':'Spearman'}, inplace = True)
dataset_corr_kendall=dataset.iloc[:,2:-1].corr(method='kendall', min_periods=1)
dataset_corr_kendall= dataset_corr_kendall['MLA'].reset_index()
dataset_corr_kendall.rename(columns = {'MLA':'Kendall'}, inplace = True)
#df_corr_kendall

dataset_corr = pd.concat([dataset_corr_pearson,dataset_corr_spearman], axis=0)
dataset_corr = pd.concat([dataset_corr,dataset_corr_kendall], axis=0)


dataset_corr = dataset_corr_pearson.merge(dataset_corr_spearman[['index', 'Spearman']])
dataset_corr = dataset_corr.merge(dataset_corr_kendall[['index', 'Kendall']])
dataset_corr.rename(columns = {'index':'Variables'}, inplace = True)
dataset_corr.to_csv(config.monthly_correlations)
dataset_corr


# In[ ]:


df = dataset.iloc[:,2:-1]
df


# In[ ]:





# In[ ]:


mask = ((df_new['MLA'] > 0) & (df_new['MLA'] < 100))


# In[ ]:


#on final weekly dataset - all banks
df_new1 = df_new.iloc[:,2:-1][mask]
df_corr_pearson=df_new1.corr(method='pearson', min_periods=1)
df_corr_pearson = df_corr_pearson['XX_MLA_CLASS2'].reset_index()
df_corr_pearson.rename(columns = {'XX_MLA_CLASS2':'Pearson'}, inplace = True)
df_corr_spearman=df_new1.corr(method='spearman', min_periods=1)
df_corr_spearman  = df_corr_spearman['XX_MLA_CLASS2'].reset_index()
df_corr_spearman.rename(columns = {'XX_MLA_CLASS2':'Spearman'}, inplace = True)
df_corr_kendall=df_new1.corr(method='kendall', min_periods=1)
df_corr_kendall= df_corr_kendall['XX_MLA_CLASS2'].reset_index()
df_corr_kendall.rename(columns = {'XX_MLA_CLASS2':'Kendall'}, inplace = True)
#df_corr_kendall

#pd.merge(df_corr_pearson, df_corr_spearman, df_corr_kendall, on='index')
#df_corr = pd.merge(df_corr_pearson, df_corr_spearman, how='inner', left_on='index')
df_corr = df_corr_pearson.merge(df_corr_spearman[['index', 'Spearman']])
df_corr = df_corr.merge(df_corr_kendall[['index', 'Kendall']])
df_corr.rename(columns = {'index':'Variables'}, inplace = True)
df_corr.to_csv(config.correlations_int)


# In[ ]:


df_corr


# In[ ]:


#On final weekly dataset Big Banks
mask_inst =( (df_new['INSTITUTIONCODE']=='B5015') | (df_new['INSTITUTIONCODE']=='B5912'))
df_new1 = df_new.iloc[:,2:-1][mask][mask_inst]
df_corr_pearson=df_new1.corr(method='pearson', min_periods=1)
df_corr_pearson = df_corr_pearson['XX_MLA_CLASS2'].reset_index()
df_corr_pearson.rename(columns = {'XX_MLA_CLASS2':'Pearson'}, inplace = True)
df_corr_spearman=df_new1.corr(method='spearman', min_periods=1)
df_corr_spearman  = df_corr_spearman['XX_MLA_CLASS2'].reset_index()
df_corr_spearman.rename(columns = {'XX_MLA_CLASS2':'Spearman'}, inplace = True)
df_corr_kendall=df_new1.corr(method='kendall', min_periods=1)
df_corr_kendall= df_corr_kendall['XX_MLA_CLASS2'].reset_index()
df_corr_kendall.rename(columns = {'XX_MLA_CLASS2':'Kendall'}, inplace = True)
#df_corr_kendall

#pd.merge(df_corr_pearson, df_corr_spearman, df_corr_kendall, on='index')
#df_corr = pd.merge(df_corr_pearson, df_corr_spearman, how='inner', left_on='index')
df_corr = df_corr_pearson.merge(df_corr_spearman[['index', 'Spearman']])
df_corr = df_corr.merge(df_corr_kendall[['index', 'Kendall']])
df_corr.rename(columns = {'index':'Variables'}, inplace = True)
df_corr.to_csv(config.correlations_int)
#dataset_corr1


# In[ ]:


df_corr


# In[ ]:


dataset_corr


# In[ ]:


#On final weekly dataset Intermediate Banks
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100) & (
    (df_new['INSTITUTIONCODE']=='B5412') | (df_new['INSTITUTIONCODE']=='B5512') |
    (df_new['INSTITUTIONCODE']=='B5213') | (df_new['INSTITUTIONCODE']=='B5413') |
    (df_new['INSTITUTIONCODE']=='B5613') | (df_new['INSTITUTIONCODE']=='B5014') |
    (df_new['INSTITUTIONCODE']=='B5914') | (df_new['INSTITUTIONCODE']=='B5115') |
    (df_new['INSTITUTIONCODE']=='B5515') | (df_new['INSTITUTIONCODE']=='B5815'))
df_new1 = df_new.iloc[:,2:-1][mask]
df_corr_pearson=df_new1.corr(method='pearson', min_periods=1)
df_corr_pearson = df_corr_pearson['XX_MLA_CLASS2'].reset_index()
df_corr_pearson.rename(columns = {'XX_MLA_CLASS2':'Pearson'}, inplace = True)
df_corr_spearman=df_new1.corr(method='spearman', min_periods=1)
df_corr_spearman  = df_corr_spearman['XX_MLA_CLASS2'].reset_index()
df_corr_spearman.rename(columns = {'XX_MLA_CLASS2':'Spearman'}, inplace = True)
df_corr_kendall=df_new1.corr(method='kendall', min_periods=1)
df_corr_kendall= df_corr_kendall['XX_MLA_CLASS2'].reset_index()
df_corr_kendall.rename(columns = {'XX_MLA_CLASS2':'Kendall'}, inplace = True)
#df_corr_kendall

#pd.merge(df_corr_pearson, df_corr_spearman, df_corr_kendall, on='index')
#df_corr = pd.merge(df_corr_pearson, df_corr_spearman, how='inner', left_on='index')
df_corr = df_corr_pearson.merge(df_corr_spearman[['index', 'Spearman']])
df_corr = df_corr.merge(df_corr_kendall[['index', 'Kendall']])
df_corr.rename(columns = {'index':'Variables'}, inplace = True)
df_corr.to_csv(config.correlations_int)


# In[ ]:


df_corr


# In[ ]:


#On final weekly dataset Small Banks
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100) & (
    (df_new['INSTITUTIONCODE']=='B5812') | (df_new['INSTITUTIONCODE']=='B5813') |
    (df_new['INSTITUTIONCODE']=='B5913') | (df_new['INSTITUTIONCODE']=='B5114') |
    (df_new['INSTITUTIONCODE']=='B5814') | (df_new['INSTITUTIONCODE']=='B5215') |
    (df_new['INSTITUTIONCODE']=='B5016') | (df_new['INSTITUTIONCODE']=='B5116') |
    (df_new['INSTITUTIONCODE']=='B5716') | (df_new['INSTITUTIONCODE']=='B5117') |

    (df_new['INSTITUTIONCODE']=='B5417') | (df_new['INSTITUTIONCODE']=='B5717') |
    (df_new['INSTITUTIONCODE']=='B5917') | (df_new['INSTITUTIONCODE']=='B5018') |
    (df_new['INSTITUTIONCODE']=='B5318') | (df_new['INSTITUTIONCODE']=='B5418') |
    (df_new['INSTITUTIONCODE']=='B5619') | (df_new['INSTITUTIONCODE']=='B5719') |
    (df_new['INSTITUTIONCODE']=='B5919') | (df_new['INSTITUTIONCODE']=='B5120') )
df_new1 = df_new.iloc[:,2:-1][mask]
df_corr_pearson=df_new1.corr(method='pearson', min_periods=1)
df_corr_pearson = df_corr_pearson['XX_MLA_CLASS2'].reset_index()
df_corr_pearson.rename(columns = {'XX_MLA_CLASS2':'Pearson'}, inplace = True)
df_corr_spearman=df_new1.corr(method='spearman', min_periods=1)
df_corr_spearman  = df_corr_spearman['XX_MLA_CLASS2'].reset_index()
df_corr_spearman.rename(columns = {'XX_MLA_CLASS2':'Spearman'}, inplace = True)
df_corr_kendall=df_new1.corr(method='kendall', min_periods=1)
df_corr_kendall= df_corr_kendall['XX_MLA_CLASS2'].reset_index()
df_corr_kendall.rename(columns = {'XX_MLA_CLASS2':'Kendall'}, inplace = True)
#df_corr_kendall

#pd.merge(df_corr_pearson, df_corr_spearman, df_corr_kendall, on='index')
#df_corr = pd.merge(df_corr_pearson, df_corr_spearman, how='inner', left_on='index')
df_corr = df_corr_pearson.merge(df_corr_spearman[['index', 'Spearman']])
df_corr = df_corr.merge(df_corr_kendall[['index', 'Kendall']])
df_corr.rename(columns = {'index':'Variables'}, inplace = True)
df_corr.to_csv(config.correlations_small)


# In[ ]:


df_corr


# In[ ]:


df_new.columns


# ##### Scatterplot Analysis of Liquidity Risk versus Factors

# In[ ]:


listFactors = ['CURRENTRATIO', 'LIQASSET2DEPOSIT', 
               'TOTAL_DEPOSITS', 'CORE_DEPOSITS','GL_TO_TOTAL_DEPOSITS',
       'VOLATILEDEPOSITS2LIAB', 'BOTBAL2TOTALDEPOSIT', 'LOAN2ASSETS',
       'LOAN2DEPOSIT', 'LOAN2COREDEPOSIT','DOMESTICDEPOSIT2ASSETS',
       'BANKSIZE','INFLATION', 'GDP',
       'GL_TO_TOTAL_FUNDING', 'EWAQ_NPL','EWAQ_NPLsNetOfProvisions',
       'EWAQ_NPLsNetOfProvisions2CoreCapital', 'NETINTERESTINCOME','LR',
       'CD_TO_TOTAL_ASSET', 'LIQASSET2TOTALASSET','ExcessShortTLiab2LongTAsset']
listVarNames = ['Current Ratio', 'Liq Asset to Deposit', 
       'Deposits', 'Core Deposits','Loans to Deposits',
       'Volatile Deposits to Liab.', 'Credit at C.Bank to Deposit', 'Loan to Assets',
       'Loan to Deposit', 'Loan to Core Deposit','Domestic Deposit to Assets',
       'Bank Size','Inflation', 'GDP',
       'G.Loan to Funding', 'NPL','NPLs Net Of Prov.',
       'NPLsNetOfProv to CoreCapital', 'Net Interest Income','Lending Rate',
       'Core Dep. to Assets', 'Liq. Asset to Assets','Excess S.T.Liab to L.T.Asset']


# In[ ]:


fig = plt.figure() 
fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('Liquidity Risk Versus Liquidity Risk Factors for Commercial Banks in Tanzania 2010-2021')
ncol = 4
nrow = 6
activeCol = 0
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100)
mask2 = (dataset['MLA'] > 0) & (dataset['MLA'] < 100)
for i in range(nrow):
    for j in range(ncol):
        #plt.subplot(ncol, nrow, activeCol)
        if listFactors[activeCol] == 'INFLATION':
            axes[i][j].scatter(dataset[mask2].loc[:,'INF'], dataset[mask2]['MLA'])
        elif listFactors[activeCol]=='NETINTERESTINCOME':
            axes[i][j].scatter(dataset[mask2].loc[:,'EWE_NetInterestIncome'], dataset[mask2]['MLA'])
        else:
            axes[i][j].scatter(df_new[mask].loc[:,listFactors[activeCol]], df_new[mask]['MLA'])
        axes[i][j].set(xlabel=listVarNames[activeCol], ylabel="MLA")
        axes[i][j].set_title(listVarNames[activeCol] + ' vs MLA')
        #axes[i][j].set_xlabel('Crosses', labelpad = 5)
        
        #axes[i][j].title(listVarNames[activeCol] + " vs MLA")
        #axes[i][j].xlabel(listVarNames[activeCol])
        #axes[i][j].ylabel('MLA')
        if activeCol + 1 < len(listFactors):
            activeCol=activeCol+1

#Add separate colourbar axes
#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

#Autoscale none
#fig.colorbar(axes[0][0], cax=cbar_ax)

footnote = "Note: Liquidity Risk versus Factors uses Weekly Dataset, with the exception of Inflation and Net Interest Income that use the Collected Monthly dataset"
plt.figtext(0.5, 0.01, footnote, ha="center")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[ ]:


#B5015, B5912

fig = plt.figure() 
fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('Liquidity Risk Versus Liquidity Risk Factors for Big Commercial Banks in Tanzania 2010-2021')
ncol = 4
nrow = 6
activeCol = 0
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100) & ( (df_new['INSTITUTIONCODE']=='B5015') | (df_new['INSTITUTIONCODE']=='B5912'))
mask2 = (dataset['MLA'] > 0) & (dataset['MLA'] < 100) & ( (df_new['INSTITUTIONCODE']=='B5015') | (df_new['INSTITUTIONCODE']=='B5912'))
for i in range(nrow):
    for j in range(ncol):
        #plt.subplot(ncol, nrow, activeCol)
        if listFactors[activeCol] == 'INFLATION':
            axes[i][j].scatter(dataset[mask2].loc[:,'INF'], dataset[mask2]['MLA'])
        elif listFactors[activeCol]=='NETINTERESTINCOME':
            axes[i][j].scatter(dataset[mask2].loc[:,'EWE_NetInterestIncome'], dataset[mask2]['MLA'])
        else:
            axes[i][j].scatter(df_new[mask].loc[:,listFactors[activeCol]], df_new[mask]['MLA'])
        axes[i][j].set(xlabel=listVarNames[activeCol], ylabel="MLA")
        axes[i][j].set_title(listVarNames[activeCol] + ' vs MLA')
        #axes[i][j].set_xlabel('Crosses', labelpad = 5)
        
        #axes[i][j].title(listVarNames[activeCol] + " vs MLA")
        #axes[i][j].xlabel(listVarNames[activeCol])
        #axes[i][j].ylabel('MLA')
        if activeCol + 1 < len(listFactors):
            activeCol=activeCol+1

#Add separate colourbar axes
#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

#Autoscale none
#fig.colorbar(axes[0][0], cax=cbar_ax)

footnote = "Note: Liquidity Risk versus Factors uses Weekly Dataset, with the exception of Inflation and Net Interest Income that use the Collected Monthly dataset"
plt.figtext(0.5, 0.01, footnote, ha="center")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[ ]:


#intermediate banks
"""
B5412, B5512, B5213, B5413, B5613, B5014, B5914, B5115, 
B5515, B5815

"""

fig = plt.figure() 
fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('Liquidity Risk Versus Liquidity Risk Factors for Intermediate Commercial Banks in Tanzania 2010-2021')
ncol = 4
nrow = 6
activeCol = 0
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100) & (
    (df_new['INSTITUTIONCODE']=='B5412') | (df_new['INSTITUTIONCODE']=='B5512') |
    (df_new['INSTITUTIONCODE']=='B5213') | (df_new['INSTITUTIONCODE']=='B5413') |
    (df_new['INSTITUTIONCODE']=='B5613') | (df_new['INSTITUTIONCODE']=='B5014') |
    (df_new['INSTITUTIONCODE']=='B5914') | (df_new['INSTITUTIONCODE']=='B5115') |
    (df_new['INSTITUTIONCODE']=='B5515') | (df_new['INSTITUTIONCODE']=='B5815'))
mask2 = (dataset['MLA'] > 0) & (dataset['MLA'] < 100) & ( 
    (df_new['INSTITUTIONCODE']=='B5412') | (df_new['INSTITUTIONCODE']=='B5512') |
    (df_new['INSTITUTIONCODE']=='B5213') | (df_new['INSTITUTIONCODE']=='B5413') |
    (df_new['INSTITUTIONCODE']=='B5613') | (df_new['INSTITUTIONCODE']=='B5014') |
    (df_new['INSTITUTIONCODE']=='B5914') | (df_new['INSTITUTIONCODE']=='B5115') |
    (df_new['INSTITUTIONCODE']=='B5515') | (df_new['INSTITUTIONCODE']=='B5815')    )
for i in range(nrow):
    for j in range(ncol):
        #plt.subplot(ncol, nrow, activeCol)
        if listFactors[activeCol] == 'INFLATION':
            axes[i][j].scatter(dataset[mask2].loc[:,'INF'], dataset[mask2]['MLA'])
        elif listFactors[activeCol]=='NETINTERESTINCOME':
            axes[i][j].scatter(dataset[mask2].loc[:,'EWE_NetInterestIncome'], dataset[mask2]['MLA'])
        else:
            axes[i][j].scatter(df_new[mask].loc[:,listFactors[activeCol]], df_new[mask]['MLA'])
        axes[i][j].set(xlabel=listVarNames[activeCol], ylabel="MLA")
        axes[i][j].set_title(listVarNames[activeCol] + ' vs MLA')
        #axes[i][j].set_xlabel('Crosses', labelpad = 5)
        
        #axes[i][j].title(listVarNames[activeCol] + " vs MLA")
        #axes[i][j].xlabel(listVarNames[activeCol])
        #axes[i][j].ylabel('MLA')
        if activeCol + 1 < len(listFactors):
            activeCol=activeCol+1

#Add separate colourbar axes
#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

#Autoscale none
#fig.colorbar(axes[0][0], cax=cbar_ax)
footnote = "Note: Liquidity Risk versus Factors uses Weekly Dataset, with the exception of Inflation and Net Interest Income that use the Collected Monthly dataset"
plt.figtext(0.5, 0.01, footnote, ha="center")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[ ]:


#small banks
"""
B5812, B5813, B5913, B5114, B5814, B5215, B5016, B5116, 
B5716, B5117, B5417, B5717, B5917, B5018, B5318, B5418, 
B5619, B5719, B5919, B5120
"""

fig = plt.figure() 
fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('Liquidity Risk Versus Liquidity Risk Factors for Small Commercial Banks in Tanzania 2010-2021')
ncol = 4
nrow = 6
activeCol = 0
mask = (df_new['MLA'] > 0) & (df_new['MLA'] < 100) & (
    (df_new['INSTITUTIONCODE']=='B5812') | (df_new['INSTITUTIONCODE']=='B5813') |
    (df_new['INSTITUTIONCODE']=='B5913') | (df_new['INSTITUTIONCODE']=='B5114') |
    (df_new['INSTITUTIONCODE']=='B5814') | (df_new['INSTITUTIONCODE']=='B5215') |
    (df_new['INSTITUTIONCODE']=='B5016') | (df_new['INSTITUTIONCODE']=='B5116') |
    (df_new['INSTITUTIONCODE']=='B5716') | (df_new['INSTITUTIONCODE']=='B5117') |

    (df_new['INSTITUTIONCODE']=='B5417') | (df_new['INSTITUTIONCODE']=='B5717') |
    (df_new['INSTITUTIONCODE']=='B5917') | (df_new['INSTITUTIONCODE']=='B5018') |
    (df_new['INSTITUTIONCODE']=='B5318') | (df_new['INSTITUTIONCODE']=='B5418') |
    (df_new['INSTITUTIONCODE']=='B5619') | (df_new['INSTITUTIONCODE']=='B5719') |
    (df_new['INSTITUTIONCODE']=='B5919') | (df_new['INSTITUTIONCODE']=='B5120') )
mask2 = (dataset['MLA'] > 0) & (dataset['MLA'] < 100) & ( 
    (df_new['INSTITUTIONCODE']=='B5812') | (df_new['INSTITUTIONCODE']=='B5813') |
    (df_new['INSTITUTIONCODE']=='B5913') | (df_new['INSTITUTIONCODE']=='B5114') |
    (df_new['INSTITUTIONCODE']=='B5814') | (df_new['INSTITUTIONCODE']=='B5215') |
    (df_new['INSTITUTIONCODE']=='B5016') | (df_new['INSTITUTIONCODE']=='B5116') |
    (df_new['INSTITUTIONCODE']=='B5716') | (df_new['INSTITUTIONCODE']=='B5117') |

    (df_new['INSTITUTIONCODE']=='B5417') | (df_new['INSTITUTIONCODE']=='B5717') |
    (df_new['INSTITUTIONCODE']=='B5917') | (df_new['INSTITUTIONCODE']=='B5018') |
    (df_new['INSTITUTIONCODE']=='B5318') | (df_new['INSTITUTIONCODE']=='B5418') |
    (df_new['INSTITUTIONCODE']=='B5619') | (df_new['INSTITUTIONCODE']=='B5719') |
    (df_new['INSTITUTIONCODE']=='B5919') | (df_new['INSTITUTIONCODE']=='B5120') )  
for i in range(nrow):
    for j in range(ncol):
        #plt.subplot(ncol, nrow, activeCol)
        if listFactors[activeCol] == 'INFLATION':
            axes[i][j].scatter(dataset[mask2].loc[:,'INF'], dataset[mask2]['MLA'])
        elif listFactors[activeCol]=='NETINTERESTINCOME':
            axes[i][j].scatter(dataset[mask2].loc[:,'EWE_NetInterestIncome'], dataset[mask2]['MLA'])
        else:
            axes[i][j].scatter(df_new[mask].loc[:,listFactors[activeCol]], df_new[mask]['MLA'])
        axes[i][j].set(xlabel=listVarNames[activeCol], ylabel="MLA")
        axes[i][j].set_title(listVarNames[activeCol] + ' vs MLA')
        #axes[i][j].set_xlabel('Crosses', labelpad = 5)
        
        #axes[i][j].title(listVarNames[activeCol] + " vs MLA")
        #axes[i][j].xlabel(listVarNames[activeCol])
        #axes[i][j].ylabel('MLA')
        if activeCol + 1 < len(listFactors):
            activeCol=activeCol+1

#Add separate colourbar axes
#cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])

#Autoscale none
#fig.colorbar(axes[0][0], cax=cbar_ax)
footnote = "Note: Liquidity Risk versus Factors uses Weekly Dataset, with the exception of Inflation and Net Interest Income that use the Collected Monthly dataset"
plt.figtext(0.5, 0.01, footnote, ha="center")

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[ ]:


df_new[['XX_MLA_CLASS2','01_CURR_ACC']].corr(method='pearson', min_periods=1)


# In[ ]:





# #### Features Selection

# ##### (1) Test Multi-Collinearity and Features Correlation with Label

# In[ ]:


#.corr()
plt.figure(figsize=(20,20))
sns.heatmap(df_factors.iloc[:,2:-1].corr(),annot=True, vmin = -1, vmax=-0.2, cmap="Greens", fmt='0.1')



# In[ ]:





# In[ ]:


#### Feature selection 
listVar = ['01_CURR_ACC',
'02_TIME_DEPOSIT',
'03_SAVINGS',
'05_BANKS_DEPOSITS',
'06_BORROWING_FROM_PUBLIC',
'07_INTERBANKS_LOAN_PAYABLE',
'08_CHEQUES_ISSUED',
'09_PAY_ORDERS',
'10_FOREIGN_DEPOSITS_AND_BORROWINGS',
'11_OFF_BALSHEET_COMMITMENTS',
'13_CASH',
'14_CURRENT_ACC',
'15_SMR_ACC',
'19_BANKS_ABROAD',
'21_INTERBANK_LOANS',
'22_TREASURY_BILLS',
'24_FOREIGN_CURRENCY',
'XX_CUSTOMER_DEPOSITS',
'XX_TOTAL_LIQUID_ASSET',
'XX_BOT_BALANCE',
'XX_BAL_IN_OTHER_BANKS',
'F077_ASSETS_TOTAL',
'F125_LIAB_TOTAL',
'EWAQ_GrossLoans',
'EWAQ_Capital',
'MLA_CLASS2', #
'TOTAL_DEPOSITS'
]

colsAll = []
for i in range(len(listVar)):
    cols = [col for col in   df_new.columns if  listVar[i] in col]
    colsAll.append(cols)
print(colsAll)


# In[ ]:


#X = df_new[['x1','x2','x3','x4','x5','x6','x7','x8','x9','EWL_LIQUIDITY RATING']]
"""
scaled_X = df_new[['EWL_LIQASSET_TOTAL',
'EWL_LIQLIAB_TOTAL',
'EWL_09. TOTAL DEPOSITS',

'EWL_LIQLIAB_TIMEDEPOSIT',
'EWL_Gross Loans to Total Deposits ',
'F077_ASSETS_TOTAL',
'F002_ASSET_BAL_BOT',
'LR',
'x7',
'x11']]

Y = df_new[['EWL_LIQUIDITY RATING']]
"""


# In[ ]:


df_new.columns


# In[ ]:


#X = df_new[['x1','x2','x3','x4','x5','x6','x7','x8','x9']]
""""
(1) The features below worked for a bank B5015 and gave accuracy between 0.5 to 0.62 depending on tuning, problem was uniform accuracy
    and uniform loss bank accuracy 0.62 wit 88 records
scaled_X = df_new[['EWL_LIQASSET_TOTAL',
'EWL_LIQLIAB_TOTAL',
'EWL_09. TOTAL DEPOSITS',
'EWL_LIQLIAB_TIMEDEPOSIT',
'EWL_Gross Loans to Total Deposits ',
'F077_ASSETS_TOTAL',
'F002_ASSET_BAL_BOT',
'LR']]
Y = df_new[['EWL_LIQUIDITY RATING']]

"""
listInst = ['B5015','B5115','B5114']

mask = ( #(df_new['INSTITUTIONCODE'] == 'B5014') |   #
        # (df_new['INSTITUTIONCODE'] == 'B5015')   #|   #acc 1.0  
         # (df_new['INSTITUTIONCODE'] == 'B5115')  |   #acc 0.75 
          # (df_new['INSTITUTIONCODE'] == 'B5015') 
          ( df_new['INSTITUTIONCODE'] == df_new['INSTITUTIONCODE'])
         #(df_new['INSTITUTIONCODE'] == 'B5919')     #
        #(df_new['INSTITUTIONCODE'] == 'B5213')
        ) 

"""
scaled_X = df_new[mask][['EWL_LIQASSET_TOTAL',
'EWL_LIQLIAB_TOTAL',
'EWL_09. TOTAL DEPOSITS',
'EWL_LIQLIAB_TIMEDEPOSIT',
'EWL_Gross Loans to Total Deposits ',
'F077_ASSETS_TOTAL',
'F002_ASSET_BAL_BOT',
'LR']]
"""
"""
scaled_X = df_new[mask][['01_CURR_ACC',
'02_TIME_DEPOSIT',
'03_SAVINGS',
'05_BANKS_DEPOSITS',
'06_BORROWING_FROM_PUBLIC',
'07_INTERBANKS_LOAN_PAYABLE',
'08_CHEQUES_ISSUED',
'09_PAY_ORDERS',
'10_FOREIGN_DEPOSITS_AND_BORROWINGS',
'11_OFF_BALSHEET_COMMITMENTS',
'13_CASH',
'14_CURRENT_ACC',
'15_SMR_ACC',
'19_BANKS_ABROAD',
'21_INTERBANK_LOANS',
'22_TREASURY_BILLS',
'24_FOREIGN_CURRENCY',
'XX_CUSTOMER_DEPOSITS',
'XX_TOTAL_LIQUID_ASSET',
'XX_BOT_BALANCE',
'XX_BAL_IN_OTHER_BANKS',
'F077_ASSETS_TOTAL',
'F125_LIAB_TOTAL',
'EWAQ_GrossLoans',
'EWAQ_Capital']]
"""



"""
scaled_X = df_new[mask][[
#'02_TIME_DEPOSIT',
#'03_SAVINGS','
#'INSTITUTIONCODE',
'XX_TOTAL_LIQUID_ASSET',
'F125_LIAB_TOTAL',
'XX_CUSTOMER_DEPOSITS',
'05_BANKS_DEPOSITS',
'02_TIME_DEPOSIT',
'03_SAVINGS',
'22_TREASURY_BILLS',
'EWAQ_GrossLoans',
'EWAQ_Capital',
'F077_ASSETS_TOTAL',
'F125_LIAB_TOTAL',
'LR',
'06_BORROWING_FROM_PUBLIC',
'07_INTERBANKS_LOAN_PAYABLE',
'19_BANKS_ABROAD',
'EWAQ_NPLsNetOfProvisions',  #significant contribution
'GDP', #little contribution
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
#'EWAQ_NPL'  poor contribution
#'XX_BAL_IN_OTHER_BANKS' poor contribution
'XX_BOT_BALANCE', #significant contribution
'21_INTERBANK_LOANS',
'GL_TO_TOTAL_FUNDING', #not very significant accuracy boost
'CD_TO_TOTAL_FUNDING',  #significant accuracy boost
#'LiqAsset2DemandLiab',
'ExcessShortTLiab2LongTAsset',
#'CD_TO_TOTAL_ASSET'
#'GL_TO_TOTAL_DEPOSITS'  #insignificant impact, reduction in accuracy, has high correlation with LiqAsset2DemLiab(-0.9)
'LIQASSET2TOTALASSET'
]]
"""

scaled_X = df_new[mask][[
#'02_TIME_DEPOSIT',
#'03_SAVINGS','
#'INSTITUTIONCODE',
##'XX_TOTAL_LIQUID_ASSET',
##'F125_LIAB_TOTAL',
##'XX_CUSTOMER_DEPOSITS',
##'05_BANKS_DEPOSITS',
##'02_TIME_DEPOSIT',
##'03_SAVINGS',
##'22_TREASURY_BILLS',
##'EWAQ_GrossLoans',
##'EWAQ_Capital',
'BANKSIZE', 
#'F077_ASSETS_TOTAL',
##'F125_LIAB_TOTAL',
'LR',
##'06_BORROWING_FROM_PUBLIC',
##'07_INTERBANKS_LOAN_PAYABLE',
##'19_BANKS_ABROAD',
'EWAQ_NPLsNetOfProvisions',  #significant contribution
'GDP', #little contribution
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
#'EWAQ_NPL'  poor contribution
#'XX_BAL_IN_OTHER_BANKS' poor contribution
#'XX_BOT_BALANCE', #significant contribution
##'21_INTERBANK_LOANS',
#'GL_TO_TOTAL_FUNDING', #not very significant accuracy boost
#'CD_TO_TOTAL_FUNDING',  #significant accuracy boost   #total funding includes loans, other than just deposit
##'LiqAsset2DemandLiab',
'ExcessShortTLiab2LongTAsset',
'CD_TO_TOTAL_ASSET',
#'GL_TO_TOTAL_DEPOSITS'  #insignificant impact, reduction in accuracy, has high correlation with LiqAsset2DemLiab(-0.9)
'LIQASSET2TOTALASSET',
#'LOAN2DEPOSIT' #,   
'LIQASSET2DEPOSIT', #RWEY
'TOTAL_DEPOSITS'
]]
#Y = df_new[mask][['EWL_LIQUIDITY RATING']]
#Y = df_new[mask][['EWL_LIQUIDITY RATING']]
Y = df_new[mask][['XX_MLA_CLASS2']]


# 

# In[ ]:





# In[ ]:


# df_tmp_results


# In[ ]:


factors = pd.concat([scaled_X, Y], axis = 1)
factors.columns


# In[ ]:


#.corr()
plt.figure(figsize=(20,20))
#sns.heatmap(factors.corr(),annot=True, vmin = -1, vmax=-0.2, cmap="Greens", fmt='0.1')


# In[ ]:


scaled_X.info()


# In[ ]:


df_new.shape


# In[ ]:


df_new[mask].head()


# ##### Correlation Analysis by Clusters (Big, Intermediate Banks, Small Banks)
# 

# #### Ranking features based on importance

# In[ ]:


df_factors = df_new.rename(columns={"CURRENTRATIO": "X1", 
                           "LIQASSET2DEPOSIT": "X2",
                           "TOTAL_DEPOSITS": "X3",
                           "CORE_DEPOSITS": "X4",
                           "LOAN2DEPOSIT": "X5",
                           "VOLATILEDEPOSITS2LIAB": "X6",
                           "BOTBAL2TOTALDEPOSIT": "X7",
                           "LOAN2COREDEPOSIT": "X10",
                           "DOMESTICDEPOSIT2ASSETS": "X11",
                           "BANKSIZE": "X12",
                           "GDP": "X14",
                           "LOAN2ASSETS": "X15",
                           "EWAQ_NPL": "X16",
                           "EWAQ_NPLsNetOfProvisions": "X17",
                           "EWAQ_NPLsNetOfProvisions2CoreCapital": "X18",
                           "LR": "X20",
                           "CD_TO_TOTAL_ASSET": "X21",
                           "LIQASSET2TOTALASSET": "X22",
                           "ExcessShortTLiab2LongTAsset": "X23",
                           "XX_MLA_CLASS2": "Y"}, errors="raise")


# In[ ]:


# Train a classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1)
clf.fit(df_factors[['X1', 
                    'X2',
                    'X3',
                    'X4',
                    'X5',
                    'X6',
                    'X7',
                    'X10',
                    'X11',
                    'X12',
                    'X14',
                    'X15',
                    'X16',
                    'X17',
                    'X18',
                    'X20',
                    'X21',
                    'X22',
                    'X23']].values, 
        df_factors['Y'].values)

# Index sort the most important features
sorted_feature_weight_idxes = np.argsort(clf.feature_importances_)[::-1] # Reverse sort

# Get the most important features names and weights

most_important_features = np.take_along_axis(
    np.array(df_factors[['X1', 
                    'X2',
                    'X3',
                    'X4',
                    'X5',
                    'X6',
                    'X7',
                    'X10',
                    'X11',
                    'X12',
                    'X14',
                    'X15',
                    'X16',
                    'X17',
                    'X18',
                    'X20',
                    'X21',
                    'X22',
                    'X23','Y']].iloc[:, 1:20].columns.tolist()), 
    sorted_feature_weight_idxes, axis=0)
most_important_weights = np.take_along_axis(
    np.array(clf.feature_importances_), 
    sorted_feature_weight_idxes, axis=0)

# Show
list(zip(most_important_features, most_important_weights))

# [('Feature1', 0.59), ('Feature2', 0.41)]
#source:https://gist.github.com/YousefGh/f6c42d5384cb53ac159082415fe43faa#file-kmeans_interp11-py


# In[ ]:


import matplotlib.pyplot as plt
plt.barh(most_important_features, most_important_weights)
plt.xlabel('Most Important Weights')
plt.ylabel('Most Important Features')
plt.title('Feature Importance ')
plt.show()


# #### Confirmation of features 

# In[ ]:





# In[ ]:


#fig, ax = plt.subplots()
#fig.suptitle('A single ax with no data')
#fig, axes = plt.subplots(2, 2)
#fig.suptitle('2 row x 2 columns axes with no dat')
#fig, ax = plt.subplots(2, 2)
fig, axes = plt.subplots(2, 4, sharex=True, figsize=(14,7))
fig.suptitle('Benchmarking Bank B5916 with Industry Prior to its Failure in August 2018')
#axes[0].set_title('Title of the first chart')

#sns.barplot(ax=axes[0], x=bulbasaur.index, y=bulbasaur.values)
#axes[0].set_title(bulbasaur.name)
df_new['Core Dep./Dep.'] = df_new['CD_TO_TOTAL_DEPOSIT']
df_new['Net NPLs'] = df_new['EWAQ_NPLsNetOfProvisions']
df_new['Liq. Assets/Assets'] = df_new['LIQASSET2TOTALASSET']
df_new['Liq. Assets/Deposits'] = df_new['LIQASSET2DEPOSIT']
df_new['Net NPLs/Core Cap.'] = df_new['EWAQ_NPLsNetOfProvisions2CoreCapital']
df_new['GL/Deposits'] = df_new['GL_TO_TOTAL_DEPOSITS']
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(ax=axes[0,0], x=df_new['REPORTINGDATE'],y=df_new['Core Dep./Dep.'],label="TZ Banks", color='b')
sns.lineplot(ax=axes[0,0], x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['Core Dep./Dep.'], label="Bank B5916", color='r')
axes[0,0].set_title('Core Deposits to Deposits')
#axes[0,0].set_ylabels('Core Deposits/Total Assets', size = 12)

sns.lineplot(ax=axes[0,1],x=df_new['REPORTINGDATE'],y=df_new['Net NPLs'],label="TZ Banks",color='b')
sns.lineplot(ax=axes[0,1],x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['Net NPLs'], label="Bank B5916", color='r')
axes[0,1].axhline(0.05, color='r', linestyle='--')
axes[0,1].set_title('Net NPLs')
#axes[0,1].set_ylabels('Net NPLs', size = 12)
sns.lineplot(ax=axes[0,2], x=df_new[df_new['LIQASSET2TOTALASSET']<1.5]['REPORTINGDATE'],y=df_new[df_new['Liq. Assets/Assets']<1.5]['Liq. Assets/Assets'],label="TZ Banks",color='b')
sns.lineplot(ax=axes[0,2], x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['Liq. Assets/Assets'], label="Bank B5916", color='r')
axes[0,2].set_title('Liquid Assets to Assets')
#axes[1,0].set_ylabels('Liquid Assets/Assets', size = 12)
mask_all = (df_new['LIQASSET2DEPOSIT']> -2) & (df_new['LIQASSET2DEPOSIT']<2)
sns.lineplot(ax=axes[0,3],x=df_new[mask_all]['REPORTINGDATE'],y=df_new[mask_all]['Liq. Assets/Deposits'],label="TZ Banks",color='b')
sns.lineplot(ax=axes[0,3],x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['Liq. Assets/Deposits'], label="Bank B5916", color='r')
axes[0,3].set_title('Liquid Assets to Deposits')
#axes[1,1].set_ylabels('Liquid Assets/Deposits', size = 12)
sns.lineplot(ax=axes[1,0],x=df_new['REPORTINGDATE'],y=df_new['Net NPLs/Core Cap.'],label="TZ Banks", color='b')
sns.lineplot(ax=axes[1,0],x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['Net NPLs/Core Cap.'], label="Bank B5916", color='r')
axes[1,0].set_title('Net NPLs to Core Capital')
#axes[2,0].set_ylabels('Net NPLs/Core Capital', size = 12)
sns.lineplot(ax=axes[1,1],x=df_new['REPORTINGDATE'],y=df_new['MLA'],label="TZ Banks", color='b')
sns.lineplot(ax=axes[1,1],x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['MLA'], label="Bank B5916", color='r')
axes[1,1].axhline(20, color='r', linestyle='--')
axes[1,1].set_title('MLA')

mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
mask = ((df_new['GL_TO_TOTAL_DEPOSITS']>-3) & (df_new['GL_TO_TOTAL_DEPOSITS']<3))
sns.lineplot(ax=axes[1,2],x=df_new['REPORTINGDATE'],y=df_new[mask]['GL/Deposits'],label="Banks in TZ", color='b')
sns.lineplot(ax=axes[1,2],x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM][mask]['GL/Deposits'], label="Bank B5916", color='r')
axes[1,2].axhline(0.80, color='r', linestyle='--')
axes[1,2].set_title('Gross Loans to Deposits')

mask1  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']<='25-FEB-2017'))
mask2  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']>'25-FEB-2017'))
sns.lineplot(ax=axes[1,3],x=df_new[mask1]['REPORTINGDATE'],y=df_new[mask1]['MLA'], label="Negative Liquidity Risk",color = 'b')
sns.lineplot(ax=axes[1,3],x=df_new[mask2]['REPORTINGDATE'],y=df_new[mask2]['MLA'], label="Positive Liquidity Risk",color='r')
axes[1,3].axhline(20, color='r', linestyle='--')
axes[1,3].set_title('MLA trends')
#axes[2,1].set_ylabels('MLA', size = 12)
#fig.text(0.5, 0.04, 'Reporting Date', ha='center')
#fig.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')

footnote = "Note: Bank B5916 had a seemingly good MLA 3 years prior to its failure but practically was under Liquidity Risk which is revealed by poor performanc in Liquidity Risk factors"
plt.figtext(0.5, 0.01, footnote, ha="center")
# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Add space between the title and the plots
plt.tight_layout()

plt.plot()



# In[ ]:


#LOAN TO DEPOSIT
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
mask = ((df_new['GL_TO_TOTAL_DEPOSITS']>-3) & (df_new['GL_TO_TOTAL_DEPOSITS']<3))
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new[mask]['GL_TO_TOTAL_DEPOSITS'],label="Commercial Banks in TZ", color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM][mask]['GL_TO_TOTAL_DEPOSITS'], label="Bank B5916", color='r')
#plt.axhline(20, color='r', linestyle='--')
plt.title('Benchmarking Gross Loan to Deposit in Bank B5916 prior to failure')
plt.ylabel('Gloss Loan to Total Deposits')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#CD_TO_TOTAL_DEPOSIT
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new['CD_TO_TOTAL_DEPOSIT'],label="Commercial Banks in TZ", color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['CD_TO_TOTAL_DEPOSIT'], label="Bank B5916", color='r')
#plt.axhline(20, color='r', linestyle='--')
plt.title('Benchmarking Deposit Structure in Bank B5916 prior to failure')
plt.ylabel('Core Deposit to Total Deposit')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#Liquidity Trends Bank B5916
mask1  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']<='25-FEB-2017'))
mask2  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']>'25-FEB-2017'))
plt.plot(df_new[mask1]['REPORTINGDATE'],df_new[mask1]['MLA'], color = 'b')
plt.plot(df_new[mask2]['REPORTINGDATE'],df_new[mask2]['MLA'], color='r')
plt.axhline(20, color='r', linestyle='--')
plt.title('Liquidity Trends in Bank B5916 prior to Failure')
plt.ylabel('MLA')
plt.xlabel('Reporting Periods')
plt.show()


# In[ ]:


#NPL Net of Provisions
mask1  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']<='30-NOV-2013'))
mask2  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']>'30-NOV-2013'))
plt.plot(df_new[mask1]['REPORTINGDATE'],df_new[mask1]['EWAQ_NPLsNetOfProvisions'], color = 'b')
plt.plot(df_new[mask2]['REPORTINGDATE'],df_new[mask2]['EWAQ_NPLsNetOfProvisions'], color='r')
plt.axhline(0.05, color='r', linestyle='--')
plt.title('Trends of NPL Net of Provisions in Bank B5916 prior to failure')
plt.ylabel('NPL Net of Provisions')
plt.xlabel('Reporting Periods')
plt.show()


# In[ ]:


#Quality of Credit Portfolio
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new['EWAQ_NPLsNetOfProvisions'],label="Commercial Banks in TZ",color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['EWAQ_NPLsNetOfProvisions'], label="Bank B5916", color='r')
plt.axhline(0.05, color='r', linestyle='--')
plt.title('Benchmarking Quality of Credit in Bank B5916 prior to failure')
plt.ylabel('NPL Net of Provisions')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#Liquid assets to total assets
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new[df_new['LIQASSET2TOTALASSET']<1.5]['REPORTINGDATE'],y=df_new[df_new['LIQASSET2TOTALASSET']<1.5]['LIQASSET2TOTALASSET'],label="Commercial Banks in TZ",color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['LIQASSET2TOTALASSET'], label="Bank B5916", color='r')
#plt.axhline(0.05, color='r', linestyle='--')
plt.title('Benchmarking Assets Structure in Bank B5916 prior to failure')
plt.ylabel('Liquid Asset to Total Assets')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#Liquid assets to deposits
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
mask_all = (df_new['LIQASSET2DEPOSIT']> -2) & (df_new['LIQASSET2DEPOSIT']<2)
sns.lineplot(x=df_new[mask_all]['REPORTINGDATE'],y=df_new[mask_all]['LIQASSET2DEPOSIT'],label="Commercial Banks in TZ",color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['LIQASSET2DEPOSIT'], label="Bank B5916", color='r')
#plt.axhline(0.05, color='r', linestyle='--')
plt.title('Benchmarking Liquid Assets to Deposits in Bank B5916 prior to failure')
plt.ylabel('Liquid Asset to Deposits Ratio')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#CD_TO_TOTAL_ASSET  #Stability of Deposits 
mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new['CD_TO_TOTAL_ASSET'],label="Commercial Banks in TZ",color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['CD_TO_TOTAL_ASSET'], label="Bank B5916", color='r')
#plt.axhline(0.05, color='r', linestyle='--')
plt.title('Benchmarking Trends of Funding Structure in Bank B5916 prior to failure')
plt.ylabel('Core Deposit to Total Assets')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new['EWAQ_NPLsNetOfProvisions2CoreCapital'],label="Commercial Banks in TZ")
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['EWAQ_NPLsNetOfProvisions2CoreCapital'], label="Bank B5916")
plt.axhline(0.05, color='r', linestyle='--')
plt.title('Trends of NPL Net of Provisions To Core Capital in Bank B5916 prior to failure')
plt.ylabel('NPL Net of Provisions to Core Capital')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


mask_bankM = (df_new['INSTITUTIONCODE']=='B5916')
sns.lineplot(x=df_new['REPORTINGDATE'],y=df_new['MLA'],label="Commercial Banks in TZ", color='b')
sns.lineplot(x=df_new[mask_bankM]['REPORTINGDATE'],y=df_new[mask_bankM]['MLA'], label="Bank B5916", color='r')
plt.axhline(20, color='r', linestyle='--')
plt.title('Benchmarking Liquidity in Bank B5916 prior to failure')
plt.ylabel('Minimum Liquid Assets Ratio')
plt.xlabel('Reporting Periods')
plt.legend()
plt.show()


# In[ ]:


#NPL Net of Provisions to Core Capital
mask1  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']<='30-NOV-2013'))
mask2  = ((df_new['INSTITUTIONCODE']=='B5916')&(df_new['REPORTINGDATE']>'30-NOV-2013'))
plt.plot(df_new[mask1]['REPORTINGDATE'],df_new[mask1]['EWAQ_NPLsNetOfProvisions'], color = 'b')
plt.plot(df_new[mask2]['REPORTINGDATE'],df_new[mask2]['EWAQ_NPLsNetOfProvisions'], color='r')
plt.axhline(0.05, color='r', linestyle='--')
plt.title('Trends of NPL Net of Provisions in Bank B5916 prior to failure')
plt.ylabel('NPL Net of Provisions')
plt.xlabel('Reporting Periods')
plt.show()


# In[ ]:





# In[ ]:


factors = pd.concat([scaled_X, Y], axis = 1)
factors.columns


# In[ ]:





# In[ ]:





# In[ ]:


scaled_X.info()


# In[ ]:


#df_new['MLAclass']=df_new['MLA'].apply(lambda x: 1 if x > 80 else)
#x*10 if x<2 elif x<4 x**2 else x+10


# **Feature and Label selection**

# In[ ]:


#20
#X = df_new[['x1','x2','x3','x4','x5','x6','x7','x8','x9']]

listInst = ['B5015','B5115','B5114']

mask = ( #(df_new['INSTITUTIONCODE'] == 'B5014') |   #
        # (df_new['INSTITUTIONCODE'] == 'B5015')   #|   #acc 1.0   
         # (df_new['INSTITUTIONCODE'] == 'B5115')  |   #acc 0.75  
          # (df_new['INSTITUTIONCODE'] == 'B5015') 
          ( df_new['INSTITUTIONCODE'] == df_new['INSTITUTIONCODE'])
         #(df_new['INSTITUTIONCODE'] == 'B5919')     #
        #(df_new['INSTITUTIONCODE'] == 'B5213')
        ) 

"""
scaled_X = df_new[mask][['EWL_LIQASSET_TOTAL',
'EWL_LIQLIAB_TOTAL',
'EWL_09. TOTAL DEPOSITS',
'EWL_LIQLIAB_TIMEDEPOSIT',
'EWL_Gross Loans to Total Deposits ',
'F077_ASSETS_TOTAL',
'F002_ASSET_BAL_BOT',
'LR']]
"""
"""
scaled_X = df_new[mask][['01_CURR_ACC',
'02_TIME_DEPOSIT',
'03_SAVINGS',
'05_BANKS_DEPOSITS',
'06_BORROWING_FROM_PUBLIC',
'07_INTERBANKS_LOAN_PAYABLE',
'08_CHEQUES_ISSUED',
'09_PAY_ORDERS',
'10_FOREIGN_DEPOSITS_AND_BORROWINGS',
'11_OFF_BALSHEET_COMMITMENTS',
'13_CASH',
'14_CURRENT_ACC',
'15_SMR_ACC',
'19_BANKS_ABROAD',
'21_INTERBANK_LOANS',
'22_TREASURY_BILLS',
'24_FOREIGN_CURRENCY',
'XX_CUSTOMER_DEPOSITS',
'XX_TOTAL_LIQUID_ASSET',
'XX_BOT_BALANCE',
'XX_BAL_IN_OTHER_BANKS',
'F077_ASSETS_TOTAL',
'F125_LIAB_TOTAL',
'EWAQ_GrossLoans',
'EWAQ_Capital']]
"""



"""
scaled_X = df_new[mask][[
#'02_TIME_DEPOSIT',
#'03_SAVINGS','
#'INSTITUTIONCODE',
'XX_TOTAL_LIQUID_ASSET',
'F125_LIAB_TOTAL',
'XX_CUSTOMER_DEPOSITS',
'05_BANKS_DEPOSITS',
'02_TIME_DEPOSIT',
'03_SAVINGS',
'22_TREASURY_BILLS',
'EWAQ_GrossLoans',
'EWAQ_Capital',
'F077_ASSETS_TOTAL',
'F125_LIAB_TOTAL',
'LR',
'06_BORROWING_FROM_PUBLIC',
'07_INTERBANKS_LOAN_PAYABLE',
'19_BANKS_ABROAD',
'EWAQ_NPLsNetOfProvisions',  #significant contribution
'GDP', #little contribution
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
#'EWAQ_NPL'  poor contribution
#'XX_BAL_IN_OTHER_BANKS' poor contribution
'XX_BOT_BALANCE', #significant contribution
'21_INTERBANK_LOANS',
'GL_TO_TOTAL_FUNDING', #not very significant accuracy boost
'CD_TO_TOTAL_FUNDING',  #significant accuracy boost
#'LiqAsset2DemandLiab',
'ExcessShortTLiab2LongTAsset',
#'CD_TO_TOTAL_ASSET'
#'GL_TO_TOTAL_DEPOSITS'  #insignificant impact, reduction in accuracy, has high correlation with LiqAsset2DemLiab(-0.9)
'LIQASSET2TOTALASSET'
]]
"""
scaled_X = df_new[[
'REPORTINGDATE',    
'INSTITUTIONCODE', 
#scaled_X = df_new[mask][[
#'02_TIME_DEPOSIT',
#'03_SAVINGS','
#'INSTITUTIONCODE',
##'XX_TOTAL_LIQUID_ASSET',
##'F125_LIAB_TOTAL',
##'XX_CUSTOMER_DEPOSITS',
##'05_BANKS_DEPOSITS',
##'02_TIME_DEPOSIT',
##'03_SAVINGS',
##'22_TREASURY_BILLS',
##'EWAQ_GrossLoans',
##'EWAQ_Capital',
'BANKSIZE', 
#'F077_ASSETS_TOTAL',
##'F125_LIAB_TOTAL',
'LR',   
##'06_BORROWING_FROM_PUBLIC',
##'07_INTERBANKS_LOAN_PAYABLE',
##'19_BANKS_ABROAD',
'EWAQ_NPLsNetOfProvisions',  #significant contribution
#'GDP', #little contribution
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
'EWAQ_NPL',  #poor contribution
#'XX_BAL_IN_OTHER_BANKS' poor contribution
#'XX_BOT_BALANCE', #significant contribution
##'21_INTERBANK_LOANS',
#'GL_TO_TOTAL_FUNDING', #not very significant accuracy boost
#'CD_TO_TOTAL_FUNDING',  #significant accuracy boost   #total funding includes loans, other than just deposit
##'LiqAsset2DemandLiab',
'ExcessShortTLiab2LongTAsset',
'CD_TO_TOTAL_ASSET',
#'GL_TO_TOTAL_DEPOSITS'  #insignificant impact, reduction in accuracy, has high correlation with LiqAsset2DemLiab(-0.9)
'LIQASSET2TOTALASSET',
#'LOAN2DEPOSIT' #,   
'LIQASSET2DEPOSIT', #RWEY
'TOTAL_DEPOSITS'
]]
#Y = df_new[mask][['EWL_LIQUIDITY RATING']]
#Y = df_new[mask][['EWL_LIQUIDITY RATING']]
#Y = df_new[mask][['XX_MLA_CLASS2']]
Y = df_new[['XX_MLA_CLASS2']]


# In[ ]:


scaled_X.shape


# In[ ]:


#df_X = dataset[['x7','x11','x8','EWL_Core Deposits to Total Funding','EWL_Gross Loans to Total Deposits ','F006_ASSET_BAL_OTHER_BANKS','F014_ASSET_INV_DEBT_SECURITIES_TBILLS','INF','LR','EWL_LIQASSET_TBILLS','EWL_LIQASSET_TOTAL','LiqRisk','EWL_LIQUIDITY RATING','MLA']]
#X = df_new[['x7','x11','x8','EWL_Core Deposits to Total Funding']] #,'EWL_Gross Loans to Total Deposits ','F006_ASSET_BAL_OTHER_BANKS','F014_ASSET_INV_DEBT_SECURITIES_TBILLS','INF','LR','EWL_LIQASSET_TBILLS','EWL_LIQASSET_TOTAL']]
#Y = df_new[['LiqRisk']]

#'LiqRisk','EWL_LIQUIDITY RATING','MLA'


# In[ ]:


#Export Model Inputs and Labels
scaled_X.to_csv(config.model_inputs_X) #("inputs.csv")
Y.to_csv(config.model_inputs_Y) #("labels.csv")


# In[ ]:




