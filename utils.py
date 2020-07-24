import pandas as pd
import numpy as np
import time
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, plot_roc_curve

# classifiers 
from sklearn import linear_model
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LassoCV

from print_utils import *

## Funcitons for EDA/Feature Engineering
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def integrate_all_files(main_df):
    files = list()

    for i in range(1,19):
        exp_number = '0' + str(i) if i < 10 else str(i)
        file = pd.read_csv("data/experiment_{}.csv".format(exp_number))
        row = main_df[main_df['No'] == i]
        
        #add experiment settings to features
        file['feedrate']=row.iloc[0]['feedrate']
        file['clamp_pressure']=row.iloc[0]['clamp_pressure']
        
        # Having label as 'tool_conidtion'
        
        file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
        files.append(file)
    df = pd.concat(files, ignore_index = True)
    for i in range(len(df["Machining_Process"])):
        if df["Machining_Process"][i] == "end":
            df["Machining_Process"][i] = "End"

    new_df = pd.DataFrame()

    pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,
    'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'Starting':11}
    new_df = df.copy()
    new_df['Machining_Process']=df['Machining_Process'].map(pro)

    return df, new_df

def shuffle_split_data(main_df):
    main_df = main_df.fillna(0)
    shuffled = shuffle(main_df)
    Y_data = shuffled["label"]
    X_data = shuffled.drop(['label'],axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.20, random_state= 42)

    return shuffled, [X_train, X_test, Y_train, Y_test]

## Functions for model
@st.cache(suppress_st_warning=True)
def model_insights(X_train, Y_train, X_test, Y_test):
    model_res = {}

    models = {
        "DummyClassifier": DummyClassifier(),
        "LogisticRegression": LogisticRegression(),
        "RandomForestClassifier": RandomForestClassifier(),
        "Perceptron": Perceptron(),
        "SGDClassifier" : SGDClassifier(),
        "DecisionTreeClassifier" : DecisionTreeClassifier(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "LinearSVC" : LinearSVC(),
        "GaussianNB": GaussianNB()
    }


    st.title("CLASSIFIERS")
    st.subheader("=======================================================")

    for key in models:

        st.subheader("Running a {} model on the data".format(key))
        print_model_info(key)
        time.sleep(1.5)
        model = models[key]
        model.fit(X_train, Y_train)

        ## Results
        model_pred = model.predict(X_test)
        test_acc_model=round(model.score(X_test, Y_test)*100,2)
        train_acc_model=round(model.score(X_train, Y_train)*100,2)

        st.markdown("**The train accuracy is {}**".format(train_acc_model))
        st.markdown("**The test accuracy is {}**".format(test_acc_model))
       
        st.markdown("#### Learning Curve")
        plt_first(model, "Leanring curves for {}".format(key), X_train, Y_train, n_jobs=6)

        model_res[key] = model
        st.subheader("=======================================================")
    
    return model_res

def plt_first(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    

    _, axes = plt.subplots(1, 1, figsize=(15, 15))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    st.pyplot()

def plt_second(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    

    _, axes = plt.subplots(1, 1, figsize=(15, 15))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot n_samples vs fit_times
    axes.grid()
    axes.plot(train_sizes, fit_times_mean, 'o-')
    axes.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("fit_times")
    axes.set_title("Scalability of the model")

    st.pyplot()

def plt_third(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    _, axes = plt.subplots(1, 1, figsize=(15, 15))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot fit_time vs score
    axes.grid()
    axes.plot(fit_times_mean, test_scores_mean, 'o-')
    axes.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes.set_xlabel("fit_times")
    axes.set_ylabel("Score")
    axes.set_title("Performance of the model")

    st.pyplot()

# Feature importance reference https://stackoverflow.com/questions/24255723/sklearn-logistic-regression-important-features?rq=1
def feature_importance_tree(importance):
    df_dict = pd.DataFrame()
    # summarize feature importance
    name_columns = ["X1_ActualPosition", "X1_ActualVelocity", "X1_ActualAcceleration", "X1_CommandPosition", 
                    "X1_CommandVelocity", "X1_CommandAcceleration", "X1_CurrentFeedback", "X1_DCBusVoltage", 
                    "X1_OutputCurrent", "X1_OutputVoltage", "X1_OutputPower", "Y1_ActualPosition", "Y1_ActualVelocity", 
                    "Y1_ActualAcceleration", "Y1_CommandPosition", "Y1_CommandVelocity", "Y1_CommandAcceleration", 
                    "Y1_CurrentFeedback", "Y1_DCBusVoltage", "Y1_OutputCurrent", "Y1_OutputVoltage", "Y1_OutputPower", 
                    "Z1_ActualPosition", "Z1_ActualVelocity", "Z1_ActualAcceleration", "Z1_CommandPosition", "Z1_CommandVelocity", 
                    "Z1_CommandAcceleration", "Z1_CurrentFeedback", "Z1_DCBusVoltage", "Z1_OutputCurrent", "Z1_OutputVoltage", 
                    "S1_ActualPosition", "S1_ActualVelocity", "S1_ActualAcceleration", "S1_CommandPosition", "S1_CommandVelocity", 
                    "S1_CommandAcceleration", "S1_CurrentFeedback", "S1_DCBusVoltage", "S1_OutputCurrent", "S1_OutputVoltage", 
                    "S1_OutputPower", "S1_SystemInertia", "M1_CURRENT_PROGRAM_NUMBER", "M1_sequence_number", "M1_CURRENT_FEEDRATE", 
                    "Machining_Process", "feedrate", "clamp_pressure"]
    df_dict["feature_name"] = name_columns
    df_dict["feature_importance"] = importance
    st.markdown("The model feature importance gives back numbers which are shown in the graph below, \
                    However we have also shown how these features are mapped to the names with the dataframe given below.")
    st.write(df_dict)
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    st.pyplot()

def feature_importance_reg(importance, X):
    feature_importance = abs(importance[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    featfig = plt.figure()
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=4)
    featax.set_xlabel('Relative Feature Importance')

    plt.tight_layout()
    st.pyplot()
