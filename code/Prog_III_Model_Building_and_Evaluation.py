#!/usr/bin/env python
# coding: utf-8

# ### Program 3 of 3
# ### Purpose: Model Building and Evaluation
# ### Programs: Data split in train and test, data scaling, build and train models, model tests and results
# ### Inputs: Labeled data for training and testing
# ### Outputs: Models and Test results of alternative RF, MLP, and Hybrid RF-MLP models 
# ### Developer: Rweyemamu Barongo rbarongo@gmail.com, ribarongo@bot.go.tz, ribarongo@udsm.ac.tz

# #### Install Libraries

# In[2]:


#working with xls files
#!pip install --upgrade xlrd


# In[3]:


#!pip install xlwt


# In[4]:


#for loading saved machine learning models
#get_ipython().system('pip install pickle')


# In[5]:


#get_ipython().system('pip install tensorflow')


# In[6]:


#get_ipython().system('pip install scikit-learn')


# In[7]:


#get_ipython().system('pip install imbalanced-learn')


# In[8]:


#get_ipython().system('pip install pandas')


# In[9]:


#get_ipython().system('pip install matplotlib')


# In[10]:


#get_ipython().system('pip install seaborn')


# #### Import Libraries

# In[11]:

#preliminary libraries
import preparation
from preparation import install_missing_packages
install_missing_packages()

#data processing libraries
import pandas as pd
import numpy as np

#data visualisation libraries
import matplotlib.pyplot as plt
import seaborn as sns

#configuration of data files
import config_unix_filesystem as config
from config_unix_filesystem import check_model_results_file
from config_unix_filesystem import check_if_exist_or_create_folders
from config_unix_filesystem import check_data_files_III

#setup colab libraries if working in colab
try:
    import google.colab
except:
    print("It's ok. You just don't need google.colab libraries outside google platform")
from datetime import date
import time as tm

#data spliting
from sklearn.model_selection import train_test_split

#data scaling
from sklearn.preprocessing import StandardScaler

#chains and combinations
from itertools import chain, combinations

#machine learning algorithms
from sklearn.linear_model import LogisticRegression

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Deep Learning 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU, PReLU, ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

#machine learning metrics
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import geometric_mean_score
#ROC curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

from sklearn.datasets import make_circles

#upsampling
from sklearn.utils import resample

#visualization
from matplotlib.pyplot import figure
#loading saved model
import pickle

#other imports
from datetime import time


# #### inspect configurations and input data

# In[12]:


check_data_files_III()
check_if_exist_or_create_folders()
check_model_results_file()


# #### Models Configurations

# In[13]:


# RF
num_estimators = 199

#MLP
#Model selection
num_model = 6                    #1 to 9 for M1 to M9
loss_function = 'sparse_categorical_crossentropy'
number_of_epochs = 30
training_validation_split = 0.3


# #### Load data from files

# In[14]:


dataset = pd.read_csv(config.dataWbook) #, sheet_name="finalDataset_240622", header=1)
unnamedCols = [col for col in dataset.columns if 'Unnamed' in col]
dataset.drop(columns=unnamedCols, inplace=True)
dataset.shape 

df_inputs_X = pd.read_csv(config.model_inputs_X, parse_dates=["REPORTINGDATE"])
unnamedCols2 = [col for col in df_inputs_X.columns if 'Unnamed' in str(col)]
df_inputs_X.drop(columns=unnamedCols2, inplace=True)

df_inputs_Y = pd.read_csv(config.model_inputs_Y)
unnamedCols2 = [col for col in df_inputs_Y.columns if 'Unnamed' in str(col)]
df_inputs_Y.drop(columns=unnamedCols2, inplace=True)

#df_results = pd.read_csv(results, parse_dates=["date"])
df_results = pd.read_csv(config.results)
unnamedCols2 = [col for col in df_results.columns if 'Unnamed' in str(col)]
df_results.drop(columns=unnamedCols2, inplace=True)
try:
    df_results['date'] = pd.to_datetime(df_results['date']) #, unit='d', origin='1899-12-30')
except:
    print("column or file {} do not exist".format(config.results))


# In[15]:


#temporary storage of test results
try:
    sno_val = max(0,df_results['sno'].max()) + 1
    sno_val
except:
    sno_val = 1

df_tmp_results = df_results.iloc[0:0]
df_tmp_results['sno']= [sno_val]
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'date'] = date.today() 


# #### Select input features and target column

# In[16]:


#model input features without identity of banks and reporting date
scaled_X = df_inputs_X[[  
'BANKSIZE', 
'LR',
'EWAQ_NPLsNetOfProvisions',  
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
'EWAQ_NPL', 
'ExcessShortTLiab2LongTAsset',
'CD_TO_TOTAL_ASSET',
'LIQASSET2TOTALASSET',  
'LIQASSET2DEPOSIT', 
'TOTAL_DEPOSITS'
]]

#model input features with bank codes and reporting date for practical performance benchmarking
scaled_X_bm = df_inputs_X[[
'REPORTINGDATE',    
'INSTITUTIONCODE',    
'BANKSIZE', 
'LR',
'EWAQ_NPLsNetOfProvisions',  
'EWAQ_NPLsNetOfProvisions2CoreCapital', #little contribution
'EWAQ_NPL', 
'ExcessShortTLiab2LongTAsset',
'CD_TO_TOTAL_ASSET',
'LIQASSET2TOTALASSET',  
'LIQASSET2DEPOSIT', 
'TOTAL_DEPOSITS'
]]

#model target/label
Y = df_inputs_Y[['XX_MLA_CLASS2']]


# In[17]:


#recording input variables/features that were used
if 'BANKSIZE' in scaled_X.columns:
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'BANKSIZE'] = 'ok' 
if 'LR' in scaled_X.columns:    
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'LR'] = 'ok' 
if 'EWAQ_NPLsNetOfProvisions' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'EWAQ_NPLsNetOfProvisions'] = 'ok' 
if 'EWAQ_NPLsNetOfProvisions2CoreCapital' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'EWAQ_NPLsNetOfProvisions2CoreCapital'] = 'ok' 
if 'EWAQ_NPL' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'EWAQ_NPL'] = 'ok'     
if 'ExcessShortTLiab2LongTAsset' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'ExcessShortTLiab2LongTAsset'] = 'ok' 
if 'CD_TO_TOTAL_ASSET' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'CD_TO_TOTAL_ASSET'] = 'ok' 
if 'LIQASSET2TOTALASSET' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'LIQASSET2TOTALASSET'] = 'ok' 
if 'LIQASSET2DEPOSIT' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'LIQASSET2DEPOSIT'] = 'ok' 
if 'TOTAL_DEPOSITS' in scaled_X.columns: 
    df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'TOTAL_DEPOSITS'] = 'ok' 



# In[18]:


factors = pd.concat([scaled_X, Y], axis = 1)
factors.columns


# In[19]:


#transpose target values
Y.T.values[0]


# In[20]:


Y.T.values[0].shape


# **Split Dataset into Train and Test datasets**

# In[21]:


#without bank codes and reporting date
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(scaled_X,Y,test_size=0.33,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(scaled_X.values,Y.T.values[0],test_size=0.33,random_state=42)


# In[22]:


##with bank codes for practical performance benchmarking
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(scaled_X,Y,test_size=0.33,random_state=42)
X_train_bm, X_test_bm, y_train_bm, y_test_bm = train_test_split(scaled_X_bm.values,Y.T.values[0],test_size=0.33,random_state=42)


# In[23]:


result = X_train[2:, :] == X_train_bm[2:, 2:]
print(result)


# In[24]:


result.shape


# In[25]:


result = X_test[2:, :] == X_test_bm[2:, 2:]
print(result)


# In[26]:


result.shape


# In[27]:


result = y_test == y_test_bm
print(result)


# **Scaling datasets (using Standard Scaler)**

# In[28]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[29]:


#temporarily scrapped bank codes for scaling
X_train_bm_scrapped=sc.fit_transform(X_train_bm[:,2:])
X_test_bm_scrapped=sc.transform(X_test_bm[:,2:])

# Concatenate the first two columns from the original arrays with the scaled arrays
X_train_bm2 = np.hstack((X_train_bm[:, :2], X_train_bm_scrapped))
X_test_bm2 = np.hstack((X_test_bm[:, :2], X_test_bm_scrapped))


# In[30]:


X_train_bm2


# In[31]:


X_train.shape[1]


# In[ ]:





# #### Building Machine Learning Models

# **(a) Model Building using Imbalanced Data (Original data before treating imbalances)**

# <!-- #### Logistic Regression With Imbalanced Data (for performance comparison with target models) -->

# #### Random Forest With Imbalanced Data (for performance comparison with target models)

# In[32]:


# train model
#num_estimators = 199
rfc = RandomForestClassifier(n_estimators=num_estimators).fit(X_train, y_train)

# save the model to disk
pickle.dump(rfc, open(config.rfc, 'wb'))

# load the model from disk
rfc = pickle.load(open(config.rfc, 'rb'))

# predict on test set
rfc_pred = rfc.predict(X_test)

rfc_accuracy = accuracy_score(y_test, rfc_pred)
rfc_precision = precision_score(y_test, rfc_pred, average='weighted')
#0.9995
rfc_f1_score = f1_score(y_test, rfc_pred,average='weighted')
#0.8666
rfc_recall = recall_score(y_test, rfc_pred,average='weighted')
#0.7878
print("Evaluating Random Forest \n Confusion Matrix:\n{}\n ClassificationReport:\n{}\n Random Forest Accuracy={} \n Precision={} \n F1 Score={} \nRecall={}".format(confusion_matrix(y_test,rfc_pred),
                                                                                                                    classification_report(y_test,rfc_pred),
                                                                                                                    rfc_accuracy,rfc_precision,rfc_f1_score,rfc_recall))




# In[33]:


#collect results into a dataframe
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_Acc'] = rfc_accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_Prec'] = rfc_precision
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_F1'] = rfc_f1_score
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_Recall'] = rfc_recall
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_Remarks'] = '{} estimators'.format(num_estimators)


# #### Building a MLP ANN model With Imbalanced Data (for performance comparison with target models)

# In[34]:


#models MLP_model_M1 to MLP_model_M9 are alternative best performing among various tested models
sc=StandardScaler()
X_train_scaled=sc.fit_transform(X_train)
X_test_scaled=sc.transform(X_test)
number_of_neurons = []


#Model M1
# Add number of neurons in layers
neurons_M1 = []
neurons_M1.append(24)
neurons_M1.append(16)
MLP_model_M1 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M1[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M1[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M2
# Add number of neurons in layers
neurons_M2 = []
neurons_M2.append(512)
neurons_M2.append(250)
neurons_M2.append(120)
MLP_model_M2 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M2[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(24,activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M2[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M2[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M3
# Add number of neurons in layers
neurons_M3 = []
neurons_M3.append(512)
neurons_M3.append(250)
neurons_M3.append(120)
neurons_M3.append(60)
MLP_model_M3 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M3[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M3[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M3[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M3[3],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M4
# Add number of neurons in layers
neurons_M4 = []
neurons_M4.append(520)
neurons_M4.append(250)
MLP_model_M4 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M4[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(24,activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M4[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),  
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M5
# Add number of neurons in layers
neurons_M5 = []
neurons_M5.append(512)
neurons_M5.append(250)
neurons_M5.append(120)
neurons_M5.append(80)
neurons_M5.append(60)
MLP_model_M5 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M5[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(24,activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M5[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M5[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M5[3],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M5[4],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M6
# Add number of neurons in layers
neurons_M6 = []
neurons_M6.append(512)
neurons_M6.append(250)
neurons_M6.append(120)
neurons_M6.append(80)
neurons_M6.append(60)                         
MLP_model_M6 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M6[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(512,activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M6[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M6[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M6[3],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M6[4],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)
                         
#Model M7
# Add number of neurons in layers
neurons_M7 = []
neurons_M7.append(512)
neurons_M7.append(350)
neurons_M7.append(250)
neurons_M7.append(120)
neurons_M7.append(80)
neurons_M7.append(60)  
MLP_model_M7 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M7[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M7[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2)
     tf.keras.layers.Dense(neurons_M7[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M7[3],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M7[4],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M7[5],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M8
# Add number of neurons in layers
neurons_M8 = []
neurons_M8.append(512)
neurons_M8.append(250)
neurons_M8.append(300)                         
MLP_model_M8 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M8[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(512,activation = 'relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M8[1],activation = 'relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M8[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)

#Model M9
# Add number of neurons in layers
neurons_M9 = []
neurons_M9.append(682)
neurons_M9.append(512)
neurons_M9.append(350)
neurons_M9.append(250)
neurons_M9.append(120)
neurons_M9.append(80)
neurons_M9.append(60)                         
MLP_model_M9 = tf.keras.models.Sequential(
    [
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(neurons_M9[0],activation = 'relu',input_dim=X_train.shape[1]),
     #tf.keras.layers.Dropout(0.2),
     #tf.keras.layers.Dense(512,activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M9[1],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M9[2],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M9[3],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M9[4],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2), 
     tf.keras.layers.Dense(neurons_M9[5],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(neurons_M9[6],activation = 'relu'),
     #tf.keras.layers.Dropout(0.2),        
     #tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(6,activation = 'softmax')
    ]
)



# In[35]:


#selection of MLP model and parameters for training and testing
# modelMLP = MLP_model_M6   
# number_of_neurons = neurons_M6

modelMLP = MLP_model_M6   
#number_of_neurons = num_neurons
num_neurons = neurons_M6
#model architecture
arch = "10-"
for i in range(len(number_of_neurons)):
    arch = "{}{}-".format(arch,str(number_of_neurons[i]))
arch = arch + str("6")

dropout = 'None'
if num_model == 1:
    model_selection = MLP_model_M1   
    num_neurons = neurons_M1 
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 2:
    model_selection = MLP_model_M2   
    num_neurons = neurons_M2
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 3:
    model_selection = MLP_model_M3   
    num_neurons = neurons_M3
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 4:
    model_selection = MLP_model_M4   
    num_neurons = neurons_M4
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 5:
    model_selection = MLP_model_M5   
    num_neurons = neurons_M5 
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 6:
    model_selection = MLP_model_M6   
    num_neurons = neurons_M6 
    optimization_function = 'adam'   #adam or SGD etc    
if num_model == 7:
    model_selection = MLP_model_M7   
    num_neurons = neurons_M7
    optimization_function = 'adam'   #adam or SGD etc  
if num_model == 8:
    model_selection = MLP_model_M8   
    num_neurons = neurons_M8 
    optimization_function = 'adam'   #adam or SGD etc  
    dropout = '0.2'
if num_model == 9:
    model_selection = MLP_model_M9   
    num_neurons = neurons_M9 
    optimization_function = 'adam'   #adam or SGD etc  
modelMLP = model_selection
var_optimizer = optimization_function
var_loss = loss_function
var_arch = 'Model {}: {} Dropout {}'.format(num_model,arch,dropout)
var_epochs = number_of_epochs
var_validation_split = training_validation_split
var_remarks_MLP = "Architecture:{}, Loss function: {}, Optimizer: {}".format(var_arch, var_loss, var_optimizer.upper())

modelMLP.compile(loss = var_loss,optimizer = var_optimizer,metrics = ['accuracy'])


# In[36]:


print(var_remarks_MLP)


# In[37]:


#get_ipython().run_cell_magic('time', '', '

#Model training and validation
#%%time
t0 = tm.time()
model_history = modelMLP.fit(X_train_scaled,y_train,validation_split=var_validation_split,epochs = var_epochs)
training_time  = tm.time()-t0


# In[38]:


#MLP model summary
modelMLP.summary()


# In[39]:


#Plots of MLP Training and Validation Before Dataset Balancing
accuracy = model_history.history['accuracy']
val_accuracy = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs  = range(len(accuracy))
#fig = plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
#fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('MLP Model M{} Training and Validation With Imbalance Data'.format(num_model))
ax1.plot(epochs, accuracy, label='Training')  
ax1.plot(epochs, val_accuracy, label='Validation')
ax1.set(xlabel='Number of Epochs', ylabel='Accuracy')
ax1.set_title('Accuracy Evaluation')
ax1.legend()
ax2.plot(epochs, loss,label='Training')
ax2.plot(epochs, val_loss,label='Validation')
ax2.legend()
ax2.set(xlabel='Number of Epochs', ylabel='Loss')
ax2.set_title('Loss Evaluation')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[40]:


#Alternative estimation of the mean accuracy 
cvs = []
ev = modelMLP.evaluate(X_test_scaled, y_test, verbose=0)
cvs.append(ev[1] * 100)
ev
print(np.mean(cvs))


# In[41]:


#Saving the model
modelMLP.save(config.MLP)
model = tf.keras.models.load_model(config.MLP)


# In[42]:


cvs = []
ev = model.evaluate(X_test_scaled, y_test, verbose=0)
cvs.append(ev[1] * 100)
ev
print(np.mean(cvs))

#https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/  CV Scores


# In[43]:


y_pred = model.predict(X_test_scaled)
y_pred


# In[44]:


#accuracy_score(y_test, y_pred, normalize=False)


# In[45]:


#model_history.history


# In[46]:


#import Keras.Backend.argmax()
#import numpy as np
best_model_accuracy = model_history.history['accuracy'][np.argmin(model_history.history['loss'])]
best_model_accuracy


# In[47]:


best_model_val_accuracy = model_history.history['val_accuracy'][np.argmin(model_history.history['val_loss'])]
best_model_val_accuracy


# In[48]:


#import numpy as np
print("accuracy: ", np.mean(np.array(accuracy)))
print("val accuracy: ", np.mean(np.array(val_accuracy)))
print("loss: ", np.mean(np.array(loss)))
print("val loss: ", np.mean(np.array(val_loss)))
print("###")
print("best_model_accuracy ",best_model_accuracy )
print("best_model_val_accuracy ",best_model_val_accuracy )


# In[49]:


# demonstration of calculating metrics for a neural network model using sklearn
# predict on test set
ann_pred = model.predict(X_test)

# predict crisp classes for test set (reduce 2D (actual+predicted) to 1D
yhat_classes = np.argmax(ann_pred,axis=1) #model.predict_classes(X_test, verbose=0)


# reduce to 1d array
yhat_probs = ann_pred[:, 0]

accuracy = accuracy_score(y_test, yhat_classes)

precision = precision_score(y_test, yhat_classes, average='weighted')

recall = recall_score(y_test, yhat_classes, average='weighted')

f1 = f1_score(y_test, yhat_classes, average='weighted')

#kappa = cohen_kappa_score(y_test, yhat_classes)
#print('Cohens kappa: %f' % kappa)


matrix = confusion_matrix(y_test, yhat_classes)
classification_reportRF = classification_report(y_test, yhat_classes)

#print(matrix)
print("Evaluating ANN \n Confusion Matrix: \n{}\nClassificationReport:\n{}\n Random Forest\n Accuracy={} \n Precision ={} \n F1 Score={} \n Recall={}".format(matrix,classification_reportRF,accuracy,precision,f1,recall))



# In[50]:


#https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18


# In[51]:


df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLP_Acc'] = accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLP_Prec'] = precision
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLP_F1'] = f1
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLP_Recall'] = recall
#df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLP_Remarks'] = var_remarks_MLP


# In[52]:


print(var_remarks_MLP)


# **(b) Model Building after Treating Data Imbalancies**

# **Treatment of Data Imbalances**

# #### Upsampling

# In[53]:


#before 
factors.XX_MLA_CLASS2.value_counts()
"""
1    9100
2    6016
3    5449
4     197
5     116
"""


# In[ ]:





# In[54]:


# concatenate our training data back together
#y_train = Y
X_traind = pd.DataFrame(scaled_X)
y_traind = pd.DataFrame(Y)
X = pd.concat([X_traind, y_traind], axis=1)

# separate minority and majority classes
one = X[X.XX_MLA_CLASS2==1]
two = X[X.XX_MLA_CLASS2==2]
three = X[X.XX_MLA_CLASS2==3]
four = X[X.XX_MLA_CLASS2==4]
five = X[X.XX_MLA_CLASS2==5]


# upsample minority
from sklearn.utils import resample
two_upsampled = resample(two,
                          replace=True, # sample with replacement
                          n_samples=len(one), # match number in majority class
                          random_state=42) # reproducible results
upsampled = pd.concat([one, two_upsampled], axis=0)
three_upsampled = resample(three,
                          replace=True, # sample with replacement
                          n_samples=len(one), # match number in majority class
                          random_state=42) # reproducible results
upsampled = pd.concat([upsampled, three_upsampled], axis=0)
four_upsampled = resample(four,
                          replace=True, # sample with replacement
                          n_samples=len(one), # match number in majority class
                          random_state=42) # reproducible results
upsampled = pd.concat([upsampled, four_upsampled], axis=0)
five_upsampled = resample(five,
                          replace=True, # sample with replacement
                          n_samples=len(one), # match number in majority class
                          random_state=42) # reproducible results
upsampled = pd.concat([upsampled, five_upsampled], axis=0)
# combine majority and upsampled minority
#upsampled = pd.concat([one, two_upsampled, three_upsampled, four_upsampled, five_upsampled])

# check new class counts
upsampled.XX_MLA_CLASS2.value_counts()



# In[55]:


#Upscale BM

X_traind_bm = pd.DataFrame(scaled_X_bm)
y_traind_bm = pd.DataFrame(Y)
X_bm = pd.concat([X_traind_bm, y_traind_bm], axis=1)

# separate minority and majority classes
one_bm = X_bm[X_bm.XX_MLA_CLASS2==1]
two_bm = X_bm[X_bm.XX_MLA_CLASS2==2]
three_bm = X_bm[X_bm.XX_MLA_CLASS2==3]
four_bm = X_bm[X_bm.XX_MLA_CLASS2==4]
five_bm = X_bm[X_bm.XX_MLA_CLASS2==5]


# upsample minority
#from sklearn.utils import resample
two_upsampled_bm = resample(two_bm,
                          replace=True, # sample with replacement
                          n_samples=len(one_bm), # match number in majority class
                          random_state=42) # reproducible results
upsampled_bm = pd.concat([one_bm, two_upsampled_bm], axis=0)
three_upsampled_bm = resample(three,
                          replace=True, # sample with replacement
                          n_samples=len(one_bm), # match number in majority class
                          random_state=42) # reproducible results
upsampled_bm = pd.concat([upsampled_bm, three_upsampled_bm], axis=0)
four_upsampled_bm = resample(four,
                          replace=True, # sample with replacement
                          n_samples=len(one_bm), # match number in majority class
                          random_state=42) # reproducible results
upsampled_bm = pd.concat([upsampled_bm, four_upsampled_bm], axis=0)
five_upsampled_bm = resample(five,
                          replace=True, # sample with replacement
                          n_samples=len(one_bm), # match number in majority class
                          random_state=42) # reproducible results
upsampled_bm = pd.concat([upsampled_bm, five_upsampled_bm], axis=0)
# combine majority and upsampled minority
#upsampled = pd.concat([one, two_upsampled, three_upsampled, four_upsampled, five_upsampled])

# check new class counts
upsampled_bm.XX_MLA_CLASS2.value_counts()



# In[56]:


upsampled_bm.head()


# In[57]:


scaled_X.shape


# In[58]:


scaled_X_bm.shape


# **Redo Data Spliting into Train and Test datasets**

# In[59]:


#Split and again 
# Separate input features and target

YUS = upsampled.XX_MLA_CLASS2
scaled_XUS = upsampled.drop('XX_MLA_CLASS2', axis=1)


# setting up testing and training sets
X_trainUS, X_testUS, y_trainUS, y_testUS = train_test_split(scaled_XUS.values, YUS, test_size=0.33, random_state=42)



# In[60]:


#split benchmark
YUS_bm = upsampled_bm.XX_MLA_CLASS2
scaled_XUS_bm = upsampled_bm.drop('XX_MLA_CLASS2', axis=1)

X_trainUS_bm, X_testUS_bm, y_trainUS_bm, y_testUS_bm = train_test_split(scaled_XUS_bm.values,YUS_bm,test_size=0.33,random_state=42)


# In[61]:


y_testUS


# In[62]:


y_testUS_bm


# In[63]:


#upsampled_bm.iloc[:,2:]


# In[64]:


#tests
# result = X_trainUS[:,:] == X_trainUS_bm[:,2:]
# print(result)
# result = X_testUS[:,:] == X_testUS_bm[:,2:]
# print(result)
# result = y_trainUS == y_trainUS_bm
# print(result)
# result = y_testUS == y_testUS_bm
# print(result)


# In[65]:


y_testUS_bm.shape


# In[66]:


X_trainUS


# **Redo Data Scaling**

# In[67]:


#from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_trainUS=sc.fit_transform(X_trainUS)
X_testUS=sc.transform(X_testUS)


# In[68]:


X_trainUS_bm_scaled=sc.fit_transform(X_trainUS_bm[:,2:])
X_testUS_bm_scaled=sc.transform(X_testUS_bm[:,2:])


# In[69]:


# Concatenate the first two columns from the original arrays with the scaled arrays
X_trainUS_bm2 = np.hstack((X_trainUS_bm[:, :2], X_trainUS_bm_scaled))
X_testUS_bm2 = np.hstack((X_testUS_bm[:, :2], X_testUS_bm_scaled))


# In[70]:


np.isnan(y_testUS)


# In[71]:


X_trainUS


# In[72]:


np.sum(np.isnan(y_testUS_bm))


# In[73]:


#save to file
columns1=['BANKSIZE', 'LR', 'EWAQ_NPLsNetOfProvisions',
       'EWAQ_NPLsNetOfProvisions2CoreCapital', 'EWAQ_NPL',
       'ExcessShortTLiab2LongTAsset', 'CD_TO_TOTAL_ASSET',
       'LIQASSET2TOTALASSET', 'LIQASSET2DEPOSIT', 'TOTAL_DEPOSITS']
columns2=['REPORTINGDATE','INSTITUTIONCODE','BANKSIZE', 'LR', 'EWAQ_NPLsNetOfProvisions',
       'EWAQ_NPLsNetOfProvisions2CoreCapital', 'EWAQ_NPL',
       'ExcessShortTLiab2LongTAsset', 'CD_TO_TOTAL_ASSET',
       'LIQASSET2TOTALASSET', 'LIQASSET2DEPOSIT', 'TOTAL_DEPOSITS']

# Use np.savetxt to save the NumPy array as a CSV file

df_X_testUS = pd.DataFrame(X_testUS,columns=columns1)
df_X_testUS.to_csv(config.X_testUS, index=False)
df_X_testUS = pd.read_csv(config.X_testUS)
df_X_testUS.columns = columns1

df_Y_testUS = pd.DataFrame(y_testUS,columns=['XX_MLA_CLASS2'])
df_Y_testUS.to_csv(config.Y_testUS, index=False)
df_Y_testUS = pd.read_csv(config.Y_testUS)
df_Y_testUS.columns = ['XX_MLA_CLASS2']

# Use np.savetxt to save the NumPy array as a CSV file

df_X_testUS_bm2 = pd.DataFrame(X_testUS_bm2,columns=columns2)
df_X_testUS_bm2.to_csv(config.X_testUS_bm2, index=False)
df_X_testUS_bm2 = pd.read_csv(config.X_testUS_bm2)
df_X_testUS_bm2.columns = columns2

df_Y_testUS_bm2 = pd.DataFrame(y_testUS_bm,columns=['XX_MLA_CLASS2'])
df_Y_testUS_bm2.to_csv(config.Y_testUS_bm, index=False)
df_Y_testUS_bm2 = pd.read_csv(config.Y_testUS_bm)
df_Y_testUS_bm2.columns = ['XX_MLA_CLASS2']



# In[74]:


y_testUS.head()


# In[75]:


type(df_Y_testUS)


# In[ ]:





# In[76]:


X_testUS


# In[77]:


df_y_testUS = pd.DataFrame(y_testUS, columns=['XX_MLA_CLASS2'])
df_y_testUS.sort_index()


# In[78]:


df_X_testUS


# In[79]:


#merge testing data
test_data = df_X_testUS
test_data['Y'] = df_Y_testUS['XX_MLA_CLASS2']
test_data


# In[80]:


#merge practical benchmarking data
benchmark_data = df_X_testUS_bm2
benchmark_data['Y'] = df_Y_testUS_bm2['XX_MLA_CLASS2']
benchmark_data 


# <!-- #### Logistic Regression After Treating Data Imbalances -->

# #### Random Forest After Treating Data Imbalances

# In[81]:


#num_estimators = 199

# train model
rfcUS = RandomForestClassifier(n_estimators=num_estimators).fit(X_trainUS, y_trainUS)

# save the model to disk
rfcUS_filename = config.rfcUS
with open(rfcUS_filename, 'wb') as rfcUS_file:
    pickle.dump(rfcUS, rfcUS_file)

# load the model from disk
with open(rfcUS_filename, 'rb') as rfcUS_file:
    rfcUS = pickle.load(rfcUS_file)

# predict on test set
rfcUS_pred = rfcUS.predict(X_testUS)

rfcUS_accuracy = accuracy_score(y_testUS, rfcUS_pred)

rfcUS_precision = precision_score(y_testUS, rfcUS_pred,average='weighted')

rfcUS_f1_score = f1_score(y_testUS, rfcUS_pred,average='weighted')

rfcUS_recall = recall_score(y_testUS, rfcUS_pred,average='weighted')



# #### MLP ANN After Treating Data Imbalances

# In[82]:


#selection of MLP model and parameters for training and testing
# modelMLPUS = MLP_model_M6  #.95 accuracy
# number_of_neurons = neurons_M6
modelMLPUS = model_selection   
number_of_neurons = num_neurons


#model architecture
arch = "10-"
for i in range(len(number_of_neurons)):
    arch = "{}{}-".format(arch,str(number_of_neurons[i]))
arch = arch + str("6")


dropout = 'None'
if num_model == 1:
    model_selection = MLP_model_M1   
    num_neurons = neurons_M1 
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 2:
    model_selection = MLP_model_M2   
    num_neurons = neurons_M2
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 3:
    model_selection = MLP_model_M3   
    num_neurons = neurons_M3
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 4:
    model_selection = MLP_model_M4   
    num_neurons = neurons_M4
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 5:
    model_selection = MLP_model_M5   
    num_neurons = neurons_M5 
    optimization_function = 'sgd'   #adam or SGD etc
if num_model == 6:
    model_selection = MLP_model_M6   
    num_neurons = neurons_M6 
    optimization_function = 'adam'   #adam or SGD etc    
if num_model == 7:
    model_selection = MLP_model_M7   
    num_neurons = neurons_M7
    optimization_function = 'adam'   #adam or SGD etc  
if num_model == 8:
    model_selection = MLP_model_M8   
    num_neurons = neurons_M8 
    optimization_function = 'adam'   #adam or SGD etc  
    dropout = '0.2'
if num_model == 9:
    model_selection = MLP_model_M9   
    num_neurons = neurons_M9 
    optimization_function = 'adam'   #adam or SGD etc 
var_optimizer = optimization_function
var_loss = loss_function
var_arch = 'Model M{}: {} Dropout {}'.format(num_model,arch,dropout)
var_epochs = number_of_epochs
var_validation_split = training_validation_split

var_remarks_MLP = 'MLP Architecture:{}, Loss:{}, Optimizer:{}, epochs:{}, validation_split:{}'.format(var_arch, 
                                                                                   var_loss,
                                                                                   var_optimizer.upper(),
                                                                                   str(var_epochs),
                                                                                   str(var_validation_split))


modelMLPUS.compile(loss = var_loss,optimizer = var_optimizer,metrics = ['accuracy'])


# In[83]:



#Model training and validation
#%%time
t0 = tm.time()
modelMLP_history = modelMLPUS.fit(X_trainUS,y_trainUS,validation_split=var_validation_split,epochs = var_epochs)
training_timeUS  = tm.time()-t0 



# In[84]:


modelMLPUS.summary()


# In[85]:


modelMLPUS.save(config.MLPUS)
modelMLPUS = tf.keras.models.load_model(config.MLPUS)


# In[86]:


#Plots of MLP Training and Validation Before Dataset Balancing
accuracyUS = modelMLP_history.history['accuracy']
val_accuracyUS = modelMLP_history.history['val_accuracy']
lossUS = modelMLP_history.history['loss']
val_lossUS = modelMLP_history.history['val_loss']
epochsUS  = range(len(accuracyUS))
#fig = plt.figure() 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
#fig, axes = plt.subplots(nrows = 6, ncols = 4, figsize=(15,12)) #, sharex=True, sharey = True)
fig.suptitle('MLP Model M{} Training and Validation - Balanced Data'.format(num_model))
ax1.plot(epochsUS, accuracyUS, label='Training')  
ax1.plot(epochsUS, val_accuracyUS, label='Validation')
ax1.set(xlabel='Number of Epochs', ylabel='Accuracy')
ax1.set_title('Accuracy Evaluation')
ax1.legend()
ax2.plot(epochsUS, lossUS,label='Training')
ax2.plot(epochsUS, val_lossUS,label='Validation')
ax2.legend()
ax2.set(xlabel='Number of Epochs', ylabel='Loss')
ax2.set_title('Loss Evaluation')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.3,
                    hspace=1.0)
plt.show()


# In[87]:


#Evaluating MLP M6 Training and Validation after Upsampling
best_model_accuracy = modelMLP_history.history['accuracy'][np.argmin(modelMLP_history.history['loss'])]
best_model_val_accuracy = modelMLP_history.history['val_accuracy'][np.argmin(modelMLP_history.history['val_loss'])]
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_BestTrAcc'] = best_model_accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_BestValAcc'] = best_model_val_accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_TrAcc'] = np.mean(np.array(accuracyUS))
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_TrLoss'] = np.mean(np.array(lossUS))
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_ValAcc'] = np.mean(np.array(val_accuracyUS))
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_ValLoss'] = np.mean(np.array(val_lossUS))
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_TrTime'] = training_time



print("Best Training accuracy: {} \n"
      "Best Training validation accuracy: {} \n"
      "Mean Training accuracy: {} \n"
      "Mean Training validation accuracy: {} \n"
      "Mean Training loss: {} \n"
      "Mean Training validation loss: {}\n"
      "Training time: {} seconds"
      .format(best_model_accuracy,
              best_model_val_accuracy,
              np.mean(np.array(accuracy)),
              np.mean(np.array(val_accuracy)),
              np.mean(np.array(loss)),
              np.mean(np.array(val_loss)),
              training_time
             ))


# In[88]:


# predict on test set
ann_pred = modelMLPUS.predict(X_testUS)

# predict crisp classes for test set (reduce 2D (actual+predicted) to 1D
yhat_classes = np.argmax(ann_pred,axis=1) #model.predict_classes(X_test, verbose=0)


# reduce to 1d array
yhat_probs = ann_pred[:, 0]

MLPUS_accuracy = accuracy_score(y_testUS, yhat_classes)

MLPUS_precision = precision_score(y_testUS, yhat_classes, average='weighted')

MLPUS_recall = recall_score(y_testUS, yhat_classes, average='weighted')

MLPUS_f1 = f1_score(y_testUS, yhat_classes, average='weighted')

# confusion matrix
matrix = confusion_matrix(y_testUS, yhat_classes)



# #### Hybrid RF-MLP Model

# In[89]:


y_testUSF = y_testUS.to_numpy().reshape(15015,)
y_testUSF.shape


# In[90]:


# HML = pd.DataFrame()
# if ((str(num_estimators) == str(199)) and (modelMLPUS == MLP_model_M6)):
HML = pd.DataFrame({'Y': y_testUSF, 'RF': rfcUS_pred, 'MLP': yhat_classes}, columns=['Y','RF','MLP'])
HML['HML']=HML.apply(lambda x: x.RF if x.RF>x.MLP else x.MLP,axis=1 )
# else:
#     HML = pd.DataFrame()

HML.head()


# In[91]:


HML.head()


# In[92]:


y_hml=HML['HML'].to_numpy().reshape(15015,)
y_hml


# In[93]:


hml_accuracy = accuracy_score(y_testUS, y_hml)

hml_precision = precision_score(y_testUS, y_hml,average='weighted')
#0.9995
hml_f1_score = f1_score(y_testUS, y_hml,average='weighted')
#0.8666
hml_recall = recall_score(y_testUS, y_hml,average='weighted')



# In[ ]:





# ### Models Performance Test Results

# #### Accuracy, Precision, F1 Score, Recall

# In[94]:


#Evaluating RF results
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_Acc'] = rfcUS_accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_Prec'] = rfcUS_precision
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_F1'] = rfcUS_f1_score
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_Recall'] = rfcUS_recall
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RF_Remarks'] = '{} estimators'.format(num_estimators)



matrix = confusion_matrix(y_testUS,rfcUS_pred)
classification_report_RFUS = classification_report(y_testUS,rfcUS_pred)
#print results
print("Evaluating Random Forest R2 with {} Estimators\n Confusion Matrix:\n{}\n ClassificationReport:\n{}\n Random Forest Accuracy={} \n Precision ={} \n F1 Score={} \nRecall={}".format(str(num_estimators),matrix,
                                                                                                                    classification_report_RFUS,
                                                                                                                    rfcUS_accuracy,rfcUS_precision,rfcUS_f1_score,rfcUS_recall))





# In[ ]:





# In[95]:


#MLP Results
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_Acc'] = MLPUS_accuracy
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_Prec'] = MLPUS_precision
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_F1'] = MLPUS_f1
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_Recall'] = MLPUS_recall
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_Remarks'] = var_remarks_MLP 


print("Evaluating ANN \n Confusion Matrix: \n{}\n ClassificationReport:\n{}\n MLP ANN Accuracy={} \n Precision ={} \n F1 Score={} \nRecall={}".format(matrix,classification_report(y_testUS,yhat_classes),MLPUS_accuracy,MLPUS_precision,MLPUS_f1,MLPUS_recall))



# In[96]:


#Hybrid RF-MLP
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_Acc'] = hml_accuracy 
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_Prec'] = hml_precision
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_F1'] = hml_f1_score
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_Recall'] = hml_recall
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_Remarks'] = "Hybrid of R2(RF) and M6(MLP) Models" 

print("Hybrid Machine Learning \n Confusion Matrix:\n{}\n ClassificationReport:\n{}\n Hybrid ML Accuracy={} \n Precision ={} \n F1 Score={} \nRecall={}".format(confusion_matrix(y_testUS,y_hml),
                                                                                                                     classification_report(y_testUS,y_hml),
                                                                                                                     hml_accuracy,hml_precision,hml_f1_score,hml_recall))





# #### ROC Curve

# In[97]:


# ROC Curve
def roc_auc(y_testUS,yhat_classes, model):
    y_testUS_onehot=label_binarize(y_testUS, classes=[1,2,3,4,5])
    yhat_classes_onehot=label_binarize(yhat_classes, classes=[1,2,3,4,5])
    #y_testUS_onehot.shape
    #y_testUS, yhat_classes
    n_classes = yhat_classes_onehot.shape[1]
    n_classes
    fpr = dict()
    tpr = dict()
    roc_auc= dict()
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i],_ = roc_curve(y_testUS_onehot[:,i], yhat_classes_onehot[:,i])
        roc_auc[i]=auc(fpr[i],tpr[i])        
    colors = cycle(['blue','red','green','yellow','purple'])
    #colors = ['blue','red','green','yellow','purple']
    for i,color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i],color=color, lw=2,
                label='ROC curve of class {0} (AUC ={1:0.2f})'.format(i+1,roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for {} model'.format(model))
    plt.legend(loc="lower right")
    plt.show()    

def computeRoc_auc(y_testUS,yhat_classes, model):
    y_testUS_onehot=label_binarize(y_testUS, classes=[1,2,3,4,5])
    yhat_classes_onehot=label_binarize(yhat_classes, classes=[1,2,3,4,5])
    #y_testUS_onehot.shape
    #y_testUS, yhat_classes
    n_classes = yhat_classes_onehot.shape[1]
    n_classes
    fpr = dict()
    tpr = dict()
    roc_auc= dict()
    lw = 2
    for i in range(n_classes):
        fpr[i], tpr[i],_ = roc_curve(y_testUS_onehot[:,i], yhat_classes_onehot[:,i])
         
        if model == 'RF':
            varname = 'RFUS_ROC_Class{}'.format(i+1)
            df_tmp_results.loc[df_tmp_results['sno']==sno_val, varname] = roc_auc[i]=auc(fpr[i],tpr[i])
        elif model == 'MLP':
            varname = 'MLPUS_ROC_Class{}'.format(i+1)
            df_tmp_results.loc[df_tmp_results['sno']==sno_val, varname] = roc_auc[i]=auc(fpr[i],tpr[i])
        elif model == 'HML':
            varname = 'HML_ROC_Class{}'.format(i+1)
            df_tmp_results.loc[df_tmp_results['sno']==sno_val, varname] = roc_auc[i]=auc(fpr[i],tpr[i])
computeRoc_auc(y_testUS,yhat_classes, 'MLP')
computeRoc_auc(y_testUS,rfcUS_pred, 'RF')
computeRoc_auc(y_testUS,y_hml, 'HML')

#y_hml


# In[98]:


#RF ROC
roc_auc(y_testUS, rfcUS_pred, 'RF') 


# In[99]:


#MLP ROC
roc_auc(y_testUS,yhat_classes, 'MLP')  


# In[100]:


#HML ROC
roc_auc(y_testUS,y_hml, 'HML') 


# #### Cohen's Kappa

# In[101]:


def cohens(y,yhat,model):
    cohenskappa = cohen_kappa_score(y, yhat)
    return 'Cohens Kappa of {} is {:.3f}'.format(model,cohenskappa)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_CohensKappa'] = cohen_kappa_score(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_CohensKappa'] = cohen_kappa_score(y_testUS, yhat_classes)   
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_CohensKappa'] = cohen_kappa_score(y_testUS, y_hml) 

print("{}\n{}\n{}".format(cohens(y_testUS, rfcUS_pred, 'RF'),
                      cohens(y_testUS, yhat_classes, 'MLP'),
                         cohens(y_testUS, y_hml, 'HML')))   


# #### Balanced Accuracy(BA)

# In[102]:


#from sklearn.metrics import balanced_accuracy_score
def balanced_accuracy(y,yhat,model):
    BA = balanced_accuracy_score(y, yhat)
    return 'BA of {} is {:.3f}'.format(model,BA)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_BA'] = balanced_accuracy_score(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_BA'] = balanced_accuracy_score(y_testUS, yhat_classes)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_BA'] = balanced_accuracy_score(y_testUS, y_hml)  

print("{}\n{}\n{}".format(balanced_accuracy(y_testUS,rfcUS_pred, 'RF'),
                      balanced_accuracy(y_testUS,yhat_classes, 'MLP'),
                         balanced_accuracy(y_testUS,y_hml, 'HML')))    


# #### Geometric Mean (G-MEAN)

# In[103]:


#from imblearn.metrics import geometric_mean_score
def gmean(y,yhat,model):
    gm = geometric_mean_score(y, yhat)
    return 'G-mean of {} is {:.3f}'.format(model,gm)
    
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_GMEAN'] = geometric_mean_score(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_GMEAN'] = geometric_mean_score(y_testUS, yhat_classes)   
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_GMEAN'] = geometric_mean_score(y_testUS, y_hml) 

print("{}\n{}\n{}".format(gmean(y_testUS,rfcUS_pred, 'RF'),
                      gmean(y_testUS,yhat_classes, 'MLP'),
                     gmean(y_testUS,y_hml, 'HML')) )    
    


# #### Negative Likelihood Ratio

# In[104]:


def liR(y,yhat,model):
    #tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    lir = (1 - sensitivity)/specificity
    return 'Negative Likelihood Ratio of {} is {:.3f}'.format(model,lir)
def computeLiR(y,yhat):
    #tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    lir = (1 - sensitivity)/specificity
    return lir
    
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_NLiR'] = computeLiR(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_NLiR'] = computeLiR(y_testUS, yhat_classes)   
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_NLiR'] = computeLiR(y_testUS, y_hml) 

print("{}\n{}\n{}".format(liR(y_testUS,rfcUS_pred, 'RF'),
                      liR(y_testUS,yhat_classes, 'MLP'),
                     liR(y_testUS,y_hml, 'HML')) )       


# #### Discriminant Power

# In[105]:


def diP(y,yhat,model):
    #tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    import math
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    a = math.log(sensitivity/(1-sensitivity))
    b = math.log(specificity/(1-specificity))
    dp = math.sqrt(3/(22/7)*(a+b) )     
    return 'Discriminant Power of {} is {:.3f}'.format(model,dp)
    
def computeDiP(y,yhat):
    #tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    import math
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    a = math.log(sensitivity/(1-sensitivity))
    b = math.log(specificity/(1-specificity))
    dp = math.sqrt(3/(22/7)*(a+b) )     
    return dp  
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_DP'] = computeDiP(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_DP'] = computeDiP(y_testUS, yhat_classes)   
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_DP'] = computeDiP(y_testUS, y_hml) 

print("{}\n{}\n{}".format(diP(y_testUS,rfcUS_pred, 'RF'),
                      diP(y_testUS,yhat_classes, 'MLP'),
                         diP(y_testUS,y_hml, 'HML')) )      


# #### Youden’s Index

# In[106]:


def yInd(y,yhat,model):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    y_index = sensitivity - (1- specificity)     
    return 'Youden\'s Index of {} is {:.3f}'.format(model,y_index)
    
def ComputeYInd(y,yhat):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp + fn)
    y_index = sensitivity - (1- specificity)     
    return y_index
    
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_YInd'] = ComputeYInd(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_YInd'] = ComputeYInd(y_testUS, yhat_classes)   
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_YInd'] = ComputeYInd(y_testUS, y_hml) 

print("{}\n{}\n{}".format(yInd(y_testUS,rfcUS_pred, 'RF'),
                      yInd(y_testUS,yhat_classes, 'MLP'),
                      yInd(y_testUS,y_hml, 'HML')) )         


# #### Type I and Type II errors

# In[107]:


def typeI_errors(y,yhat,model):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    typeI = fn / (tp+fn)  
    return 'Type I error of {} is {:.3f}'.format(model,typeI)

    
def typeII_errors(y,yhat,model):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    typeII = fp / (tn + fp)    
    return 'Type II error of {} is {:.3f}'.format(model,typeII)
    
def computeTypeI_errors(y,yhat):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    typeI = fn / (tp+fn)  
    return typeI
    
def computeTypeII_errors(y,yhat):
    cm= confusion_matrix(y, yhat)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1] 
    typeII = fp / (tn + fp)    
    return typeII

df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_TypeIError'] = computeTypeI_errors(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_TypeIError'] = computeTypeI_errors(y_testUS, yhat_classes) 
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_TypeIError'] = computeTypeI_errors(y_testUS, y_hml) 
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'RFUS_TypeIIError'] = computeTypeII_errors(y_testUS, rfcUS_pred)
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'MLPUS_TypeIIError'] = computeTypeII_errors(y_testUS, yhat_classes) 
df_tmp_results.loc[df_tmp_results['sno']==sno_val, 'HMLUS_TypeIIError'] = computeTypeII_errors(y_testUS, y_hml) 

print("{}\n{}\n{}\n{}\n{}\n{}".format(typeI_errors(y_testUS,rfcUS_pred, 'RF'),
                      typeI_errors(y_testUS,yhat_classes, 'MLP'),
                      typeI_errors(y_testUS,y_hml, 'HML'),
                             typeII_errors(y_testUS,rfcUS_pred, 'RF'),
                      typeII_errors(y_testUS,yhat_classes, 'MLP'),
                      typeII_errors(y_testUS,y_hml, 'HML')) )  


# 

# In[108]:


# #update hybdrid data
# # Set values to empty where RF_Remarks is not '199 estimators' and model M6 does not contain '10-512-250-120-80-60-6'
# condition = (df_tmp_results['RF_Remarks'] != '199 estimators') & (~df_tmp_results['MLPUS_Remarks'].str.contains('10-512-250-120-80-60-6'))

# # Columns to set as empty
# columns_to_empty = ['HMLUS_Acc', 'HMLUS_Prec', 'HMLUS_F1', 'HMLUS_Recall', 
#                     'HML_ROC_Class1', 'HML_ROC_Class2', 'HML_ROC_Class3', 'HML_ROC_Class4', 'HML_ROC_Class5',
#                     'HMLUS_CohensKappa', 'HMLUS_BA', 'HMLUS_GMEAN', 'HMLUS_NLiR', 'HMLUS_DP', 'HMLUS_YInd',
#                     'HMLUS_TypeIError', 'HMLUS_TypeIIError','HMLUS_Remarks']
# , 
# # Update the DataFrame based on the conditions
# df_tmp_results.loc[condition, columns_to_empty] = ''

# condition2 = (df_results['RF_Remarks'] != '199 estimators') & (~df_results['MLPUS_Remarks'].str.contains('10-512-250-120-80-60-6'))

# # Columns to set as empty
# columns_to_empty = ['HMLUS_Acc', 'HMLUS_Prec', 'HMLUS_F1', 'HMLUS_Recall', 
#                     'HML_ROC_Class1', 'HML_ROC_Class2', 'HML_ROC_Class3', 'HML_ROC_Class4', 'HML_ROC_Class5',
#                     'HMLUS_CohensKappa', 'HMLUS_BA', 'HMLUS_GMEAN', 'HMLUS_NLiR', 'HMLUS_DP', 'HMLUS_YInd',
#                     'HMLUS_TypeIError', 'HMLUS_TypeIIError','HMLUS_Remarks']
# , 
# # Update the DataFrame based on the conditions
# df_results.loc[condition2, columns_to_empty] = ''


# In[109]:


#Save results
df_results = pd.concat([df_results, df_tmp_results], axis=0)
df_results.to_csv(config.results)


# In[110]:


#display updated results file
df_results_display = pd.read_csv(config.results)
unnamedCols2 = [col for col in df_results_display.columns if 'Unnamed' in str(col)]
df_results_display.drop(columns=unnamedCols2, inplace=True)
#df_results_display['date'] = pd.to_datetime(df_results_display['date'])  
df_results_display[['sno', 'date','MLPUS_Acc','MLPUS_Prec','MLPUS_F1','MLPUS_Recall','MLPUS_TrTime']].tail()



# In[111]:


df_results_display.tail()


# In[112]:


#Mean RF results after Dataset Balancing (Except ROC)
#df_results_display.groupby(['RF_Remarks'])['RF_Remarks','RFUS_Acc','RFUS_Prec','RFUS_F1','RFUS_Recall','RFUS_BA','RFUS_CohensKappa','RFUS_DP','RFUS_GMEAN','RFUS_NLiR','RFUS_TypeIError','RFUS_TypeIIError','RFUS_YInd'].mean()
df_results_display.groupby(['RF_Remarks'])[['RFUS_Acc', 'RFUS_Prec', 'RFUS_F1', 'RFUS_Recall', 'RFUS_BA', 'RFUS_CohensKappa', 'RFUS_DP', 'RFUS_GMEAN', 'RFUS_NLiR', 'RFUS_TypeIError', 'RFUS_TypeIIError', 'RFUS_YInd']].mean()


# In[113]:


#Mean RF ROC results after Dataset Balancing
df_results_display.groupby(['RF_Remarks'])[['RFUS_ROC_Class1','RFUS_ROC_Class2','RFUS_ROC_Class3','RFUS_ROC_Class4','RFUS_ROC_Class5']].mean()


# In[ ]:





# In[114]:


#Mean HML results (except ROC)
df_results_display.groupby(['HMLUS_Remarks'])[['HMLUS_Acc','HMLUS_Prec','HMLUS_F1','HMLUS_Recall','HMLUS_BA','HMLUS_CohensKappa','HMLUS_DP','HMLUS_GMEAN','HMLUS_NLiR','HMLUS_TypeIError','HMLUS_TypeIIError','HMLUS_YInd']].mean()
 


# In[115]:


#Mean HML ROC results 
df_results_display.groupby(['HMLUS_Remarks'])[['HML_ROC_Class1','HML_ROC_Class2','HML_ROC_Class3','HML_ROC_Class4','HML_ROC_Class5']].mean()


# In[ ]:





# In[116]:


#Mean MLP results (Except ROC)
df_results_display.groupby(['MLPUS_Remarks'])[['MLPUS_Acc','MLPUS_Prec','MLPUS_F1','MLPUS_Recall','MLPUS_BA','MLPUS_CohensKappa','MLPUS_DP','MLPUS_GMEAN','MLPUS_NLiR','MLPUS_TypeIError','MLPUS_TypeIIError','MLPUS_YInd']].mean()
  


# In[117]:


#Mean MLP ROC results
df_results_display.groupby(['MLPUS_Remarks'])[['MLPUS_ROC_Class1', 'MLPUS_ROC_Class2', 'MLPUS_ROC_Class3', 'MLPUS_ROC_Class4', 'MLPUS_ROC_Class5']].mean()




# In[ ]:




