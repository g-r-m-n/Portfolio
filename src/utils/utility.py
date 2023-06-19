# -*- coding: utf-8 -*-
import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import sklearn.metrics as metrics
from lightgbm import LGBMRegressor, log_evaluation, early_stopping 
from sklearn.metrics import mean_absolute_error
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import RandomizedSearchCV
import pmdarima as pm
import pickle
from matplotlib import pyplot as plt
import os
from sklearn.metrics import  f1_score, balanced_accuracy_score, make_scorer
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from pandas.api.types import is_numeric_dtype
from scipy.stats import pearsonr
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow import keras


def add_lagged(df1, x_vars, lag = 1):
    df1 = df1.copy()
    df_lagged= df1[x_vars+['y_var_lagged']].groupby(level='CCY').shift(lag)
    df1 = pd.merge(df1, df_lagged ,left_index=True, right_index=True, suffixes=('', '_lagged'+str(lag)))
        
    return df1


def get_lstm(df1_train, df1_test, x_vars_plus, x_vars, c = '_Total_', PRINT = 1, LOAD = 1,  output_folder_model = '', LOG_TRANSFORM = 0,    look_back = 1):
    """Run the LSTM model
        LSTM RNN stands for Long Short-Term Memory recurrent neural network 
        LSTM networks use memory blocks - instead of neurons - connected through layers.

        A block has components that make it smarter than a classical neuron and a memory for recent sequences. A block contains gates that manage the block’s state and output. A block operates upon an input sequence, and each gate within a block uses the sigmoid activation units to control whether it is triggered or not, making the change of state and addition of information flowing through the block conditional.

        There are three types of gates within a unit:

        *    Forget Gate: conditionally decides what information to throw away from the block
        *    Input Gate: conditionally decides which values from the input to update the memory state
        *    Output Gate: conditionally decides what to output based on input and the memory of the block       
       """
    if PRINT:
        print('\n'+'-'*100)
        print('LSTM RNN model for '+c+'\n')
        if LOG_TRANSFORM:
            print('\nUsing log-transformation on the response\n')
    # prepare the data
    
    ## lag:
    ## normalize the dataset
    ## Reason: LSTMs are sensitive to the scale of the input data, specifically when the sigmoid (default) or tanh activation functions are used.
    scaler = MinMaxScaler(feature_range=(0, 1))

    df1_train_c = df1_train.loc[df1_train.index.get_level_values('CCY')==c,['y_var']+x_vars_plus]

    ## get y and X train
    y_train = df1_train_c['y_var'].droplevel('CCY')

    ## scale the data set:
    scaler.fit(df1_train_c)
    df1_train_s = scaler.transform(df1_train_c)

    ## get y and X train
    y_train_s = df1_train_s[:,0]
    if LOG_TRANSFORM:
        y_train_s = np.log(y_train_s+1)    
    X_train_s = df1_train_s[:,1:df1_train_s.shape[1]]

    ## reshape input to be [samples, time steps, features]
    X_train_s = np.reshape(X_train_s, (X_train_s.shape[0], look_back, len(x_vars)+1) )


    ## get y and X test
    if not (df1_test is None):

        df1_test_c  = df1_test.loc[ df1_test.index.get_level_values('CCY')==c, ['y_var']+x_vars_plus]
        y_test = df1_test_c['y_var'].droplevel('CCY')
       
        ## scale the data set:
        df1_test_s  = scaler.transform(df1_test_c)

        ## get y and X test
        y_test_s = df1_test_s[:,0]
        if LOG_TRANSFORM:
            y_test_s = np.log(y_test_s+1)
        X_test_s = df1_test_s[:,1:df1_test_s.shape[1]]

        ## reshape input to be [samples, time steps, features]
        X_test_s  = np.reshape(X_test_s,  (X_test_s.shape[0],  look_back, len(x_vars)+1) )


    # get folder name of the trained model:
    folder_name_trained_model = output_folder_model + 'trained_model_lstm_'+c
    # create output_folder if not existant:
    os.makedirs(folder_name_trained_model, exist_ok=True)

    # check if trained model already exists:
    if (os.path.isdir(folder_name_trained_model)) and LOAD:
        # load the trained model:        
        lstm_model = keras.models.load_model(folder_name_trained_model)
    else:
        # create and check the LSTM network
        # The network has a visible layer with len(x_vars_plus) input, a hidden layer with 4 LSTM blocks, and an output layer that makes a single value prediction. The default sigmoid activation function is used for the LSTM blocks. 
        lstm_model = Sequential()
        lstm_model.add(LSTM(1, input_shape=(  look_back, len(x_vars)+1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        if PRINT:
            # show the model summary:
            print(lstm_model.summary())

        # This callback will stop the training when there is no improvement in the loss for two consecutive epochs.
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        # Fit the LSTM
        if PRINT:
            print('\nStart training ...\n')
        # The network is trained a number of epochs, and a batch size of 1 is used.
        lstm_model.fit(X_train_s, y_train_s,   epochs=100, batch_size=1, verbose=2, callbacks=[callback])

        # save the trained model
        lstm_model.save(folder_name_trained_model)

    # make predictions
    if PRINT:
        print('\nGet predictions ...\n')
    trainPredict_s = lstm_model.predict(X_train_s)
    testPredict_s  = lstm_model.predict(X_test_s)

    # invert predictions
    trainPredict  = scaler.inverse_transform(np.concatenate((trainPredict_s,df1_train_s[:,1:df1_train_s.shape[1]]),axis=1))[:,0]
    testPredict   = scaler.inverse_transform(np.concatenate((testPredict_s,df1_test_s[:,1:df1_test_s.shape[1]]),axis=1))[:,0]     

    if LOG_TRANSFORM:
        trainPredict = np.exp(trainPredict) - 1
        testPredict  = np.exp(testPredict)  - 1

    return  {'trainPredict':trainPredict, 'testPredict':testPredict, 'y_train':y_train, 'y_test':y_test}



def plot_observed_vs_predicted(y_obs, y_pred, title1 = 'Observed vs predicted'):
    """plot Actual vs Fitted"""
    plt.rcParams.update({'figure.figsize':(24,5)})
    plt.plot(pd.Series(y_obs,index=y_obs.index))
    plt.plot(pd.Series(y_pred, index=y_obs.index), color='darkgreen')
    plt.tick_params('x', labelrotation=45)
    plt.title(title1)
    plt.legend(labels=['Observed', 'Predicted'])
    plt.xlabel('Date')
    plt.grid()       
    plt.show()



def plot_descriptives_per_moth(df):
    """plot descriptive graphs per month"""
    # create ratios dataset:
    ratios_CCY = df.copy().reset_index(drop=True)
    # drop total
    ratios_CCY = ratios_CCY[ratios_CCY.CCY!='_Total_']
    # calculate earnings
    ratios_CCY['Earnings'] = ratios_CCY['Revenue'] + ratios_CCY['Interests']
    # calculate costs
    ratios_CCY['Costs'] = ratios_CCY[['ChangeRequestCosts', 'AdministrationCosts',
            'OverheadCosts']].sum(axis =1)
    # calculate gross profit
    ratios_CCY['Profit'] = ratios_CCY['Earnings'] + ratios_CCY['Costs']
    # select relevant columns:
    ratios_CCY =ratios_CCY[['Date','CCY','Revenue','Interests','Costs','Earnings','Profit']]  
    # create a year - month column
    ratios_CCY['Year_Month'] = ratios_CCY['Date'].dt.year.astype(str) + '_' +ratios_CCY['Date'].dt.month.astype(str) 
    # aggregate to year-month
    ratios_CCY = ratios_CCY[['CCY','Year_Month','Revenue','Interests','Costs','Earnings','Profit']].groupby(['CCY','Year_Month']).sum()
    ratios_CCY['Earnings_to_Costs'] = ratios_CCY['Earnings']/abs(ratios_CCY['Costs'])
    ratios_CCY.reset_index(inplace=True)
    # long to wide format
    ratios_CCY = pd.pivot(ratios_CCY, index='Year_Month', columns='CCY', values=['Revenue','Interests','Costs','Earnings','Earnings_to_Costs','Profit'])


    plt.rcParams.update({'figure.figsize':(24,7)})
    fig, axes = plt.subplots(1, 2, sharex=False)
    ratios_CCY['Revenue'].plot(ax=axes[0])
    axes[0].set_title('Revenue')
    axes[0].tick_params('x', labelrotation=45)
    ratios_CCY['Interests'].plot(ax=axes[1])
    axes[1].set_title('Interests')
    axes[1].tick_params('x', labelrotation=45)
    plt.show()

    plt.rcParams.update({'figure.figsize':(24,7)})
    fig, axes = plt.subplots(1, 2, sharex=False)
    ratios_CCY['Earnings'].plot(ax=axes[0])
    axes[0].set_title('Earnings = Revenue + Interests')
    axes[0].tick_params('x', labelrotation=45)
    ratios_CCY['Costs'].plot(ax=axes[1])
    axes[1].set_title('Costs')
    axes[1].tick_params('x', labelrotation=45)
    plt.show()

    plt.rcParams.update({'figure.figsize':(24,7)})
    fig, axes = plt.subplots(1, 2, sharex=False)
    ratios_CCY['Profit'].plot(ax=axes[0])
    axes[0].set_title('Profit')
    axes[0].tick_params('x', labelrotation=45)
    ratios_CCY['Earnings_to_Costs'].plot(ax=axes[1])
    axes[1].set_title('Earnings_to_Costs')
    axes[1].tick_params('x', labelrotation=45)
    plt.show()

    return ratios_CCY
    

def get_arima(df1_train, smodels, x_vars, c = '_Total_', PRINT = 1, LOAD = 1, PLOT = 0, output_folder_model = ''):
    """Run the SARIMA model
       SARIMA stands for Seasonal Autoregressive Integrated Moving Average """
    if PRINT:
        print('\n'+'-'*100)
        print('SARIMA model for '+c+'\n')

    file_name_trained_model = output_folder_model + 'trained_model_sarima_'+c+'.pkl'

    # check if trained model already exists:
    if (os.path.isfile(file_name_trained_model)) and LOAD:
        # load the trained model:
        smodel = pickle.load(open(file_name_trained_model, "rb"))
    else:
        y_train = df1_train.loc[df1_train.index.get_level_values('CCY')==c,'y_var'].droplevel('CCY')
        X_train = df1_train.loc[df1_train.index.get_level_values('CCY')==c,x_vars].droplevel('CCY')

        # Setup the auto-ARIMA model
        smodel = pm.arima.AutoARIMA(start_p=1, start_q=1,            
                test='kpss',  # Kwiatkowski–Phillips–Schmidt–Shin. type of unit root test to use in order to detect stationarity if stationary is False and d is None. 
                m=5, # seasonal differencing: set for m=5 for daily data (there is no data for Saturday and Sunday)
                max_p=3, # The order of the auto-regressive (AR) model (i.e., the number of lag observations). A time series is considered AR when previous values in the time series are very predictive of later values. An AR process will show a very gradual decrease in the ACF plot.
                max_q=3, # The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series data. An MA process is a linear combination of past errors.
                d=None, # The order of first-differencing. if set to None, the value will be automatially determined.
                seasonal=True,
                max_P = 2, # The order of the seasonal component for the auto-regressive (AR) model.
                max_D = 1, # The integration order of the seasonal process.
                max_Q = 2, # The order of the seasonal component of the moving average (MA) model.                 
                trace=True,
                error_action='ignore',  
                suppress_warnings=True, 
                stepwise=True)
        # Fit stepwise the auto-ARIMA model:
        smodel.fit(y=y_train,  X=X_train)
                                    
        # save the trained model:
        pickle.dump(smodel, open(file_name_trained_model, "wb"))

    if PRINT:
        print(smodel.summary())

    # plot diagnostics
    if PLOT:
        smodel.plot_diagnostics(figsize=(24,5))
        plt.show()

    smodels[c] = smodel

    return smodels





def get_error_stats_in(df1_train, smodels, df, res_stats_in, x_vars, c = '_Total_', PLOT = 1):
    """Forecast in the training sample and get the error statistics"""

    # get the model
    smodel = smodels[c] 
    # get y:train
    y_train = df1_train.loc[df1_train.index.get_level_values('CCY')==c,'y_var'].droplevel('CCY')
    X_train = df1_train.loc[df1_train.index.get_level_values('CCY')==c,x_vars].droplevel('CCY')

    # Get the error statistics of the predictions and observations of the training period:
    predictions_training = smodel.predict_in_sample(X_train, start = min(y_train.index), end=max(y_train.index))
    res_stats_in_c = error_statistics(y_train, predictions_training, df, colname = c)

    res_stats_in[c] = res_stats_in_c

    if PLOT:
        # plot Actual vs Fitted
        plt.rcParams.update({'figure.figsize':(24,5)})
        plt.plot(pd.Series(y_train,index=y_train.index))
        plt.plot(pd.Series(predictions_training.values, index=y_train.index), color='darkgreen')
        plt.tick_params('x', labelrotation=45)
        plt.title(c+'  in-sample: Observed vs predicted')
        plt.legend(labels=['Observed', 'Predicted'])
        plt.xlabel('Date')
        plt.grid()       
        plt.show()

    return res_stats_in



def plot_model_results(trained_model, df1, X_test, y_test, y_var = 'y_var', condition_variable = 'CCY', output_folder_plots = '', title1='Prediction', title2= 'importance', SAVE_OUTPUT = 0):
    """function to predict and plot model results"""
    
    predictions = trained_model.predict(X_test)
    
    #plot Observed vs prediction for the horizon of the dataset
    items = np.unique(df1.index.get_level_values(condition_variable))
    fig, axs = plt.subplots(nrows=1, ncols=len(items), figsize = (24, 7))
    for ix, i in enumerate(items):
        #fig = plt.figure(figsize=(16,8))
        inx_i = y_test.index.get_level_values(condition_variable)==i
        #calculate MAE
        #mae = np.round(mean_absolute_error(y_test[inx_i], predictions[inx_i]), 3) 
        
        axs[ix].plot(pd.Series(y_test[inx_i].values,index=y_test.index.get_level_values('Date')[inx_i]), color='red')
        axs[ix].plot(pd.Series(predictions[inx_i], index=y_test.index.get_level_values('Date')[inx_i]), color='green')
        axs[ix].tick_params('x', labelrotation=45)
        axs[ix].set_title(f'{i} : observed vs prediction')
        axs[ix].set_xlabel('Date')
        axs[ix].set_ylabel(y_var+' : '+str(i))
        axs[ix].legend(labels=['Observed', 'Prediction'])
        axs[ix].grid()

    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title1+'_'+str(i)+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title1+'_'+str(i)+ '.png', dpi=100,bbox_inches="tight")
    plt.show()

    #create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': trained_model.feature_name_,
        'importance': trained_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    #plot variable importances of the model
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance, y=df_importances.feature, orient='h')
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title2+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title2+ '.png', dpi=100,bbox_inches="tight")    
    plt.show()



def get_error_stats_out(df1_test, smodels, df, res_stats_out, x_vars, horizon, c = '_Total_', DYNAMIC_FORCASTING = 1, PLOT = 1, df1_train = None):
    """Forecast in the hold-out sample and get the error statistics"""
    # get the model
    smodel = smodels[c] 

    # get y: and X test
    y_test = df1_test.loc[df1_test.index.get_level_values('CCY')==c,'y_var'].droplevel('CCY')
    if DYNAMIC_FORCASTING:
        X_test = df1_test.loc[df1_test.index.get_level_values('CCY')==c, x_vars].droplevel('CCY')
    else:
        X_train = df1_train.loc[df1_train.index.get_level_values('CCY')==c, x_vars].droplevel('CCY')
        X_test = pd.DataFrame(np.tile(X_train.mean(axis=0),(horizon,1)),index=y_test.index)

    # Get the predictions of the test period:
    fitted, confint = smodel.predict(n_periods=horizon, return_conf_int=True, X = X_test)
    index_of_fc = pd.date_range(X_test.index[-1], periods = horizon, freq='MS')

    # get the error statistics:
    res_stats_out_c = error_statistics(y_test, fitted.values, df, colname = c)    
    res_stats_out[c] = res_stats_out_c
    
    if PLOT:
        # plot Actual vs Fitted
        # make series for plotting purpose
        fitted_series = pd.Series(fitted.values, index=index_of_fc)
        lower_series  = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series  = pd.Series(confint[:, 1], index=index_of_fc)

        # Plot
        plt.rcParams.update({'figure.figsize':(24,5)})
        plt.plot( pd.Series(y_test.values, index=index_of_fc))
        plt.plot(fitted_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                        lower_series, 
                        upper_series, 
                        color='k', alpha=.15)
        plt.legend(labels=['Observed', 'Predicted'])
        title1 = c+'  out-of-sample: Observed vs predicted'
        if DYNAMIC_FORCASTING:
            title1 += ' (using dynamic forcasting)'
        else:
            title1 += ' (using training data only)'
        plt.title(title1)
        plt.xlabel('Date')
        plt.grid()
        plt.show()

    return res_stats_out



def train_time_series_with_folds(df1,  y_var = 'y_var', model_type = 'lgbm', horizon=30, TUNE = False):
    """function to tune and train the model"""
    X = df1.drop(y_var, axis=1)
    y = df1[y_var]
    
    # For non-numeric data, set them to the categorial data type: 
    for i in X.columns:
        if not is_numeric_dtype(X.loc[:,i]):
            X.loc[:,i] = X.loc[:,i].astype("category")        

    #take last week of the dataset for validation
    inx_day = np.unique(df1.index.get_level_values('Date'))[-horizon]
    
    X_train, X_test = X.iloc[df1.index.get_level_values('Date')<inx_day,:], X.iloc[df1.index.get_level_values('Date')>=inx_day,:]
    y_train, y_test = y.iloc[df1.index.get_level_values('Date')<inx_day], y.iloc[df1.index.get_level_values('Date')>=inx_day]
    
    # Use an LGBM ML model:    
    if model_type in ['lgbm','rf']:
        n_estimators = 20
                 
        params = {'subsample': 0.5, 'num_leaves': 10, 'max_depth': 5, 'learning_rate': 0.2} #'boosting_type' : 'dart'}
        if model_type == 'rf':
                params = params | {'boosting_type' : 'rf', 'bagging_freq' : 1, 'bagging_fraction' : 0.8  } 
                # {'subsample': 0.5, 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.01}
        clf = LGBMRegressor( random_state=42, n_estimators=n_estimators, **params)
        tuning_dict = { 
               #'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 'learning_rate': [0.01, 0.1, 0.3, 0.5],
                                'max_depth': [3, 5, -1], #'max_depth': [3, 5, 15, 20, 30],
                                'num_leaves': [5, 10, 31], #'num_leaves': [5, 10, 20, 30],
                                'subsample': [0.3, 0.5, 1] #'subsample': [0.1, 0.2, 0.8, 1]                  
            }
        # 'subsample': 0.8, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.01 
        #{'subsample': 1, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.01}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 5, 'learning_rate': 0.1}
    #create, train and do inference of the model
    if TUNE:
        # Tune hyperparameters and final model using cv cross-validation with n_iter parameter settings sampled from random search. Random search can cover a larger area of the paramter space with the same number of consider setting compared to e.g. grid search.
        rs = RandomizedSearchCV(clf, tuning_dict, 
            scoring= {'F1':  make_scorer(f1_score), 'balanced_accuracy': make_scorer(balanced_accuracy_score)}, #'f1', 'balanced_accuracy' Overview of scoring parameters: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                # default: accuracy_score for classification and sklearn.metrics.r2_score for regression
            refit= 'F1',
            cv=5, 
            return_train_score=False, 
            n_iter=20,
            verbose = True
        )
        print("\nTuning hyperparameters ..")
        rs.fit(X_train, y_train, eval_set = [[X_test, y_test]], 
               eval_metric = ['accuracy','auc','f1','logloss'], 
               callbacks=[log_evaluation(n_estimators), early_stopping(10)])    # 'f1_score','accuracy' # 
        
        print("\nTuned hyperparameters :(best score)     ",rs.best_score_)
        print("\nTuned hyperparameters :(best prameters) ",rs.best_params_)
        
        model = clf
        clf.set_params(**rs.best_params_)
    else:
        model = clf


    model.fit(X_train, y_train)


    return model, X_train, X_test, y_train, y_test

def plotseasonal(res, axes, title='' ):
    pd.DataFrame(res.observed).plot(ax=axes[0], legend=False, title= title)
    axes[0].set_ylabel('Observed')
    pd.DataFrame(res.trend).plot(ax=axes[1], legend=False)
    axes[1].set_ylabel('Trend')
    pd.DataFrame(res.seasonal).plot(ax=axes[2], legend=False)
    axes[2].set_ylabel('Seasonal')
    pd.DataFrame(res.resid).plot(ax=axes[3], legend=False)
    axes[3].set_ylabel('Residual')



def decompose_time_series(df, y_var='Revenue',  condition_variable = 'CCY', samples='all', period=5, decomposition_model_type='', output_folder_plots = '', title1='Decomposition', SAVE_OUTPUT = 0, PLOT = 1):
    """Seasonal decomposition using moving averages"""
    # set period=5 for daily data (there is no data for Saturday and Sunday)
    items = np.unique(df[condition_variable])
    if PLOT:
        fig, axs = plt.subplots(nrows=4, ncols=len(items), sharex=True, figsize = (24, 7))
    results = dict()
    #plt.rcParams.update({'figure.figsize': (10,10)})
    for ix, i in enumerate(items):
        inx_i = df.index[df[condition_variable]==i]
        if samples == 'all':
            #decomposing all time series timestamps
            result = seasonal_decompose(df.loc[inx_i,y_var].values, period=period, model=decomposition_model_type, extrapolate_trend='freq')
        else:
            #decomposing a sample of the time series
            result = seasonal_decompose(df.loc[inx_i,y_var].values[-samples:], period=period, model=decomposition_model_type, extrapolate_trend='freq')
                        
        if PLOT:
            # Plot
            #result.plot( ax = axs[ix]).suptitle(y_var+' in '+str(i))
            plotseasonal(result, axes= axs[:, ix], title =  y_var+' in '+str(i))

        results[i] = result

    # Saving plot to pdf and png file
    if PLOT and SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title1+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title1+'.png', dpi=100,bbox_inches="tight")   

    if PLOT:
        plt.tight_layout()
        plt.show()

    return results



def error_statistics(y_true, y_pred, df, PRINT = 0, ADDITIONAL_STATS=1, RND =3, colname='Error Statistics'):
    """
    Get error statistics of the true and predicted values
    :param y_true: (vector) true values.
    :param y_pred: (vector) predicted values.
    :param ADDITIONAL_STATS: (binary) indicates whether to show additional error statistics.
    :param RND: (int) indicates the nnumber of digits of the printed error statistics.
    """
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    #mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    mean_absolute_percentage_error =metrics.mean_absolute_percentage_error(y_true, y_pred)

    # 'Number of cases': len(y_true),
    res_stats =pd.DataFrame.from_dict({ 'Root mean squared error': round(np.sqrt(mse), RND),'Mean squared error ':round(mse, RND), 'Mean absolute error':round(mean_absolute_error, RND), 'Median absolute error':round(median_absolute_error, RND), 'Mean absolute prctg error': round(mean_absolute_percentage_error, RND)} , orient='index',columns=[colname]) 

    if 1:
        Total_observed_Profit  = sum( df.loc[y_true.index,"Profit"].values )
        Total_predicted_Profit = sum( y_pred) 
        Percentage_error_Total_Profit = (Total_predicted_Profit - Total_observed_Profit)/Total_observed_Profit*100
        res_stats.loc['Total observed Profit '] =  round(Total_observed_Profit, RND)
        res_stats.loc['Total predicted Profit'] =  round(Total_predicted_Profit, RND)
        s.loc['Percentage error total Profit'] =  round(Percentage_error_Total_Profit, RND)

    if ADDITIONAL_STATS:
        r2 = metrics.r2_score(y_true, y_pred)
        explained_variance = metrics.explained_variance_score(y_true, y_pred)
        correlation, _ = pearsonr(y_true, y_pred)

        res_stats.loc['Explained variance'] =  round(explained_variance, RND)
        #res_stats.loc['R2']                =  round(r2, RND)
        res_stats.loc['Correlation']       =  round(correlation,RND)

    if PRINT:
        print('\n-------------------------------------------------------------------------')
        print('\nNumber of cases           :', len(y_true))
        print('\nRoot mean squared error   :', round(np.sqrt(mse), RND))
        print('\nMean squared error        :', round(mse, RND))
        print('\nMean absolute error       :', round(mean_absolute_error, RND))
        print('\nMedian absolute error     :', round(median_absolute_error, RND))
        print('\nMean absolute prctg error :', round(mean_absolute_percentage_error, RND))
        print('\n-------------------------------------------------------------------------')        
        if ADDITIONAL_STATS:
            print('\n-------------------------------------------------------------------------')
            print('\nExplained variance        :', round(explained_variance, RND))
            #print('\nmean_squared_log_error: ', round(mean_squared_log_error,RND))
            print('\nR2                        :', round(r2, RND))
            print('\Correlation                :', round(correlation, RND))
        print('\n-------------------------------------------------------------------------')
     
    return res_stats



def get_difference_in_hours(start_timestamp, end_timestamp, PRINT = True):
    """
    Compute the difference in time of two timestamps in hours.
    
    Parameters
    ----------
    start_timestamp (timestamp or string in format '%Y/%m/%d %H:%M'): 
        The timestamp of the start time.
    end_timestamp (timestamp or string in format '%Y/%m/%d %H:%M'): 
        The timestamp of the ending time.
    PRINT (boolean):
        Indiacates whether the outcome should be printed.   

    Return 
    ----------
    (float): 
        The difference in time of two timestamps in hours
    """

    # convert start timestamp if it isn't a datetime object:
    start_timestamp = get_timestamp(start_timestamp)

    # convert end timestamp if it isn't a datetime object:
    end_timestamp = get_timestamp(end_timestamp)

    time_difference = (end_timestamp - start_timestamp).total_seconds()/3600

    if PRINT:
        print('\nThere are %.1f hours between %s and %s'%(time_difference,start_timestamp,end_timestamp))
    return time_difference



def get_timestamp(timestamp):
    """
    helper function to convert input to a datetime if it isn't a datetime object.
    """
    if not isinstance(timestamp, datetime.date):
        timestamp = datetime.datetime.strptime(timestamp,'%Y/%m/%d %H:%M')
    return timestamp



def get_difference_in_hours_during_nine_to_five(start_timestamp, end_timestamp, PRINT = True):
    """
    Compute the difference in time of two timestamps in hours between 09:00 – 17:00 and only on weekdays.
    
    Parameters
    ----------
    start_timestamp (timestamp or string in format '%Y/%m/%d %H:%M'): 
        The timestamp of the start time.
    end_timestamp (timestamp or string in format '%Y/%m/%d %H:%M'): 
        The timestamp of the ending time.
    PRINT (boolean):
        Indiacates whether the outcome should be printed.   

    Return 
    ----------
    (float): 
        The difference in time of two timestamps in hours
    """

    # get the timestamp difference:
    time_difference = get_difference_in_hours_during_nine_to_five_worker(start_timestamp, end_timestamp)

    if PRINT:
        print('\nThere are %.1f hours on weekdays from 9am to 5pm between %s and %s'%(time_difference,start_timestamp,end_timestamp))

    return time_difference    



def get_difference_in_hours_during_nine_to_five_worker(start_timestamp, end_timestamp):
    """
    Worker function within wrapper function get_difference_in_hours_during_nine_to_five.
    
    """

    # convert start timestamp if it isn't a datetime object:
    start_timestamp = get_timestamp(start_timestamp)

    # convert end timestamp if it isn't a datetime object:
    end_timestamp = get_timestamp(end_timestamp)

    # prepare the start day to be at the (next) relevant date time for the computation:
    start_timestamp = prepare_start_timestamp(start_timestamp)
  
    # prepare the end day to be at the (next) relevant date time for the computation:
    end_timestamp = prepare_end_timestamp(end_timestamp)


    # Result 1: no valid difference:
    if start_timestamp > end_timestamp:
        return 0.0


    # Result 2: same day:
    if (end_timestamp - start_timestamp).days==0:
        return (end_timestamp - start_timestamp).total_seconds()/3600
    

    # Result 3: at least one day difference:

    ## initizalize the time_difference:
    time_difference = 0.0

    ## add startday difference:
    time_difference += 17 - start_timestamp.hour - start_timestamp.minute/60
    start_timestamp += datetime.timedelta(1)

    ## add end_timestamp difference:
    time_difference +=  end_timestamp.hour + end_timestamp.minute/60 - 9
    end_timestamp -= datetime.timedelta(1)

    ## define weekend days:
    weekend = set([5, 6])

    ## add time of the remaining days between start and end:
    d = start_timestamp
    while d <= end_timestamp:
        if d.weekday() not in weekend:
            time_difference += 8 # 8 = 17 - 9 
        d += datetime.timedelta(days=1)

    return time_difference    



def prepare_start_timestamp(start_timestamp, weekend = set([5, 6])):
    """
    helper function to prepare the start day
    """
    # go to next day if too late:
    if start_timestamp.hour>=17:
        start_timestamp = start_timestamp.replace(hour=9, minute=0)
        start_timestamp += datetime.timedelta(1)

    # set to 9 if too early:
    if start_timestamp.hour<9:
        start_timestamp = start_timestamp.replace(hour=9, minute=0)

    # go to next day if weekend:
    while start_timestamp.weekday() in weekend:
        start_timestamp += datetime.timedelta(1)

    return start_timestamp



def prepare_end_timestamp(end_timestamp, weekend = set([5, 6])):
    """
    helper function to prepare the end day
    """
    # set to 17 if too late:
    if end_timestamp.hour>=17:
        end_timestamp = end_timestamp.replace(hour=17, minute=0)

    # go to previous day if too early:
    if end_timestamp.hour<9:
        end_timestamp = end_timestamp.replace(hour=17, minute=0)
        end_timestamp -= datetime.timedelta(1)

    # go to previous day if weekend:
    while end_timestamp.weekday() in weekend:
        end_timestamp -= datetime.timedelta(1)

    return end_timestamp
