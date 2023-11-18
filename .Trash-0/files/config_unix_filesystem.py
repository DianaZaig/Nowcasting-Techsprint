#CONFIGURATION FILE
import pandas as pd
import os
# This code may execute in colab or in other environment
# Colab part works if code detect colab environment
# This experiment was conducted in colab and in desktop jupyter application for reproducibility
try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB == False:
    # Collected data
    # monthly data after anonymization and before merging
    dataWbook = 'data/original/Preprocessed_14062023.csv'
    # weekly secondary data after anonymization and before merging
    file1 = 'data/original/WK_Liquidity20100115_20151231.xls'
    file2 = 'data/original/WK_Liquidity20160101_20211231.xls'
    weeklyDataWbook = 'data/original/weeklyDataWbook.csv'

    # intermediate data files during data preparation
    # file for intermediate data during data coding
    file3 = 'data/intermediate/liquidityReturns_20230706_raw.csv'
    # file for variable labeling
    file4 = 'data/static/liquidityVariableLabeling.csv'  # liq variable annotation
    # intermediate data after denoising and scaling of features
    dfX = "data/intermediate/Preprocessed_03072023_v8_denoised_scaled_features_tavana.csv"

    null_empty_for_analysis = 'data/intermediate/null_empty.csv'
    file5 = 'data/intermediate/liquidityReturns_20230706_pivoted_01.csv'
    file5_II = 'data/intermediate/liqData_20230707_pivoted_interpolated_a3.csv'
    file5_III = 'data/intermediate/liquidityReturns_20230706_pivoted_03.csv'
    file6 = 'data/static/banksCodes.csv'
    file7 = 'data/intermediate/liqData_20230707_pivoted_interpolated_07_a3.csv'
    file7i = 'data/intermediate/monthly_in_weekly_format_missing_filed.csv'

    # weekly dataset after merging collected datasets
    weekly_dataset = 'data/final/proposed_dataset_21082023.csv'

    # intermediate data during factors analysis
    factors_corr = 'data/intermediate/factors_corr.csv'
    correlations = 'data/intermediate/correlations.csv'
    monthly_correlations = 'data/intermediate/monthly_correlations.csv'
    correlations_MLA = 'data/intermediate/correlations_MLA.csv'
    correlations_big = 'data/intermediate/correlations_big.csv'
    monthly_correlationsSBM = 'data/intermediate/monthly_correlationsSBM.csv'
    monthly_correlationsIBM = 'data/intermediate/monthly_correlationsIBM.csv'
    correlations_int = 'data/intermediate/correlations_int.csv'
    monthly_correlationsBBM = 'data/intermediate/monthly_correlationsBBM.csv'
    correlations_small = 'data/intermediate/correlations_small.csv'
    # model inputs and labels
    model_inputs_X = 'data/model_input/inputs.csv'
    inputs_bm = 'data/model_input/inputs_bm.csv'
    model_inputs_Y = 'data/model_input/label.csv'

    # models
    lrc = 'data/results/lrc.sav'
    lrcUS = 'data/results/lrcUS.sav'
    MLP = 'data/results/MLP.h5'
    MLPUS = 'data/results/MLPUS.h5'
    rfc = 'data/results/rfc.sav'
    rfcUS = 'data/results/rfcUS.sav'

    # model intermediate results
    rf_estimators_accuracy = 'data/intermediate/rf_estimators_accuracy.csv'
    optimal_MLPs = 'data/intermediate/optimal_MLPs.csv'

    # models test
    X_testUS_bm2 = 'data/model_test/X_testUS_bm2.csv'
    X_testUS_bm = 'data/model_test/X_testUS_bm.csv'
    X_testUS = 'data/model_test/X_testUS.csv'
    Y_testUS_bm2 = 'data/model_test/X_testUS_bm2.csv'
    Y_testUS_bm = 'data/model_test/X_testUS_bm.csv'
    Y_testUS = 'data/model_test/Y_testUS.csv'

    # model results
    results = 'data/results/models_results.csv'
    results_path = 'data/results/'
    data_path = 'data/'

    # !pip install xlwt
else:
    
    from google.colab import drive
    drive.mount('/content/gdrive/', force_remount=True)
    #import local libraries
    import sys
    path_to_module = '/content/gdrive/MyDrive/RESEARCH/MSCDISSERTATION/Dissertation/Approach2_LiqReturns_v003/'
    sys.path.append(path_to_module)
    
    #collected data
    #monthly data after anonymisation and  before merging
    dataWbook = path_to_module +'data/original/Preprocessed_14062023.csv'    
    #weekly data after anonymisation and  before merging
    file1 = path_to_module +'data/original/WK_Liquidity20100115_20151231.xls'
    file2 = path_to_module +'data/original/WK_Liquidity20160101_20211231.xls'
    weeklyDataWbook = path_to_module +'data/original/weeklyDataWbook.csv'

    #intermediate data files during data preparation
    #file for intermediate data during data coding 
    file3 =path_to_module +'data/intermediate/liquidityReturns_20230706_raw.csv'    
    #file for variable labeling  
    file4 = path_to_module +'data/static/liquidityVariableLabeling.csv' 
    #intermediate data after denoising and scaling of features
    dfX = path_to_module +'data/intermediate/Preprocessed_03072023_v8_denoised_scaled_features_tavana.csv'
    #other intermediary data
    factors_corr = path_to_module +'data/intermediate/factors_corr.csv'
    null_empty_for_analysis = path_to_module +'data/intermediate/null_empty.csv'
    file5 = path_to_module +'data/intermediate/liquidityReturns_20230706_pivoted_01.csv'     
    file5_II = path_to_module +'data/intermediate/liqData_20230707_pivoted_interpolated_a3.csv' 
    file5_III = path_to_module +'data/intermediate/liquidityReturns_20230706_pivoted_03.csv'
    file6 = path_to_module +'data/static/banksCodes.csv' 
    file7 = path_to_module +'data/intermediate/liqData_20230707_pivoted_interpolated_07_a3.csv' 
    file7i = path_to_module +'data/intermediate/monthly_in_weekly_format_missing_filed.csv'  
       
    #resulted weekly dataset after merging, transforming, and cleaning collected datasets 
    weekly_dataset =path_to_module +'data/final/proposed_dataset_21082023.csv'
    
    #intermediate data during factors analysis
    factors_corr =  path_to_module +'data/intermediate/factors_corr.csv'
    correlations =  path_to_module +'data/intermediate/correlations.csv'
    monthly_correlations = path_to_module +'data/intermediate/monthly_correlations.csv'
    correlations_MLA =  path_to_module +'data/intermediate/correlations_MLA.csv'
    correlations_big =  path_to_module +'data/intermediate/correlations_big.csv'
    monthly_correlationsSBM =  path_to_module +'data/intermediate/monthly_correlationsSBM.csv'
    monthly_correlationsIBM =  path_to_module +'data/intermediate/monthly_correlationsIBM.csv'
    correlations_int =  path_to_module +'data/intermediate/correlations_int.csv'
    monthly_correlationsBBM =  path_to_module +'data/intermediate/monthly_correlationsBBM.csv'
    correlations_small =  path_to_module +'data/intermediate/correlations_small.csv'   
    #model inputs and labels
    model_inputs_X = path_to_module +'data/model_input/inputs.csv'
    model_inputs_Y = path_to_module +'data/model_input/label.csv'
    inputs_bm = path_to_module +'data/model_input/inputs_bm.csv'
    #model results
    rf_estimators_accuracy = path_to_module +'data/intermediate/rf_estimators_accuracy.csv'
    optimal_MLPs = path_to_module +'data/intermediate/optimal_MLPs.csv'
    
    #models
    lrc = path_to_module + 'data/results/lrc.sav'
    lrcUS = path_to_module + 'data/results/lrcUS.sav'
    MLP = path_to_module + 'data/results/MLP.h5'
    MLPUS = path_to_module + 'data/results/MLPUS.h5'
    rfc = path_to_module + 'data/results/rfc.sav'
    rfcUS = path_to_module + 'data/results/rfcUS.sav'
    
    #models test
    X_testUS_bm2 = path_to_module + 'data/model_test/X_testUS_bm2.csv'
    X_testUS_bm = path_to_module + 'data/model_test/X_testUS_bm.csv'
    X_testUS = path_to_module + 'data/model_test/X_testUS.csv'
    Y_testUS_bm2 = path_to_module + 'data/model_test/Y_testUS_bm2.csv'
    Y_testUS_bm = path_to_module + 'data/model_test/Y_testUS_bm.csv'
    Y_testUS = path_to_module + 'data/model_test/Y_testUS.csv'


    #model results
    results = path_to_module +'data/results/models_results.csv'
    results_path = path_to_module +'data/results/'
    data_path = path_to_module +'data/'
    
#Check if key files exist:

def check_model_results_file():
    # Define the header columns
    header_columns = [
        'sno', 'date', 'BANKSIZE', 'LR', 'EWAQ_NPLsNetOfProvisions', 'EWAQ_NPL', 'EWAQ_NPLsNetOfProvisions2CoreCapital',
        'CD_TO_TOTAL_ASSET', 'LIQASSET2TOTALASSET', 'LIQASSET2DEPOSIT', 'ExcessShortTLiab2LongTAsset', 'TOTAL_DEPOSITS',
        'HML_ROC_Class1', 'HML_ROC_Class2', 'HML_ROC_Class3', 'HML_ROC_Class4', 'HML_ROC_Class5', 'HMLUS_Acc', 'HMLUS_BA',
        'HMLUS_CohensKappa', 'HMLUS_DP', 'HMLUS_F1', 'HMLUS_GMEAN', 'HMLUS_NLiR', 'HMLUS_Prec', 'HMLUS_Recall', 'HMLUS_Remarks',
        'HMLUS_TypeIError', 'HMLUS_TypeIIError', 'HMLUS_YInd', 'MLP_Acc', 'MLP_F1', 'MLP_Prec', 'MLP_Recall', 'MLPUS_Acc',
        'MLPUS_BA', 'MLPUS_BestTrAcc', 'MLPUS_BestValAcc', 'MLPUS_CohensKappa', 'MLPUS_DP', 'MLPUS_F1', 'MLPUS_GMEAN',
        'MLPUS_NLiR', 'MLPUS_Prec', 'MLPUS_Recall', 'MLPUS_Remarks', 'MLPUS_ROC_Class1', 'MLPUS_ROC_Class2', 'MLPUS_ROC_Class3',
        'MLPUS_ROC_Class4', 'MLPUS_ROC_Class5', 'MLPUS_TrAcc', 'MLPUS_TrLoss', 'MLPUS_TrTime', 'MLPUS_TypeIError',
        'MLPUS_TypeIIError', 'MLPUS_ValAcc', 'MLPUS_ValLoss', 'MLPUS_YInd', 'RF_Acc', 'RF_F1', 'RF_Prec', 'RF_Recall',
        'RF_Remarks', 'RFUS_Acc', 'RFUS_BA', 'RFUS_CohensKappa', 'RFUS_DP', 'RFUS_F1', 'RFUS_GMEAN', 'RFUS_NLiR',
        'RFUS_Prec', 'RFUS_Recall', 'RFUS_ROC_Class1', 'RFUS_ROC_Class2', 'RFUS_ROC_Class3', 'RFUS_ROC_Class4', 'RFUS_ROC_Class5',
        'RFUS_TypeIError', 'RFUS_TypeIIError', 'RFUS_YInd'
    ]

    # Check if the Excel file exists
    file_path = results
    try:
        # Try to read the existing file
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        # If the file doesn't exist, create it with the header columns
        df = pd.DataFrame(columns=header_columns)
        df.to_excel(file_path, index=False, header=False)
        print(f"File '{file_path}' created with header columns.")
    #else:
    #    print(f"File '{file_path}' already exists.")
    


def check_if_exist_or_create_folders():
    # List of folders to check and create
    folders = [data_path + 'datafinal', 
               data_path + 'intermediate', 
               data_path + 'model_input', 
               data_path + 'model_test', 
               data_path + 'results', 
               data_path +'static']

    # Get the current working directory
    current_dir = os.getcwd()

    # Iterate through each folder
    for folder in folders:
        # Create the full path for the folder
        folder_path = os.path.join(current_dir, folder)

        # Check if the folder exists
        if not os.path.exists(folder_path):
            # If it doesn't exist, create the folder
            os.makedirs(folder_path)
            print(f"Folder '{folder}' created.")
        #else:
        #    print(f"Folder '{folder}' already exists.")   
        
def check_data_files_I():
    # List of data files to check
    data_files = [weeklyDataWbook, dataWbook, file1, file2,weekly_dataset]

    # Get the current working directory
    current_dir = os.getcwd()

    # Flag to check if any file is missing
    files_missing = False

    # Iterate through each data file
    for data_file in data_files:
        # Create the full path for the file
        file_path = os.path.join(current_dir, data_file)

        # Check if the file exists
        if not os.path.exists(file_path):
            # If it doesn't exist, set the flag to True
            files_missing = True
            print(f"Source data file '{data_file}' is missing.")
        #else:
        #    print(f"File '{data_file}' exists.")

    # Check if any file is missing and report
    if files_missing:
        print("One or more of the key data source files is missing.")
    #else:
    #    print("All key data source files exist.")        

def check_data_files_II():
    # List of data files to check
    data_files = [weekly_dataset, dataWbook]

    # Get the current working directory
    current_dir = os.getcwd()

    # Flag to check if any file is missing
    files_missing = False

    # Iterate through each data file
    for data_file in data_files:
        # Create the full path for the file
        file_path = os.path.join(current_dir, data_file)

        # Check if the file exists
        if not os.path.exists(file_path):
            # If it doesn't exist, set the flag to True
            files_missing = True
            print(f"Source data file '{data_file}' is missing.")
        #else:
        #    print(f"File '{data_file}' exists.")

    # Check if any file is missing and report
    if files_missing:
        print("One or more of the key data source files is missing.")
    #else:
    #    print("All key data source files exist.")    
    
def check_data_files_III():
    # List of data files to check
    data_files = [model_inputs_X, model_inputs_Y]

    # Get the current working directory
    current_dir = os.getcwd()

    # Flag to check if any file is missing
    files_missing = False

    # Iterate through each data file
    for data_file in data_files:
        # Create the full path for the file
        file_path = os.path.join(current_dir, data_file)

        # Check if the file exists
        if not os.path.exists(file_path):
            # If it doesn't exist, set the flag to True
            files_missing = True
            print(f"Source data file '{data_file}' is missing.")
        #else:
        #    print(f"File '{data_file}' exists.")

    # Check if any file is missing and report
    if files_missing:
        print("One or more of the key data source files is missing.")
    #else:
    #    print("All key data source files exist.")          

# Now you can use the DataFrame 'df' for further operations if needed.

#print(file5)    