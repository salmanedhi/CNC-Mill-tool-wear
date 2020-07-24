import streamlit as st
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from print_utils import *
from utils import *

data = []
def main_app():

    train_res = pd.read_csv("data/train.csv")
    all_files = []
    res_model = {}

    st.sidebar.title("CNC Mining Tool Wear Detection")
    option = st.sidebar.radio("Select", ["Introduction","EDA/Feature Engineering", "Model", "Results"])

    if option == "Introduction":
        intro_info()
        print_intro()

    elif option == "EDA/Feature Engineering":
        print_insight_info()
        i = 0 
        files = []
        for dirname, _, filenames in os.walk('data'): 
            for filename in filenames:
                if "experiment" in filename:
                    i+=1
                    files.append(os.path.join(dirname, filename))
        
        st.sidebar.markdown("### Choose your actions from dropdown menu :")
        extract_button = st.sidebar.selectbox("Select excel file:" ,["train.csv Information", "experiments.csv Information", 
                                                "Data Processing for Model training"])
        st.subheader("=======================================================")

        if extract_button == "train.csv Information":
            st.title("Data Insights of train.csv file")
            st.subheader("Train.csv DataFrame: ")
            st.write(train_res)
            st.subheader("=======================================================")

            st.subheader("\n Columns: ")
            st.write(train_res.columns)
            st.subheader("=======================================================")

            st.markdown("### Label to predict: {}".format("tool_condition"))
            
            # checking for NaN columns
            st.subheader("=======================================================")
            st.subheader("\n \n NaN Check")
            st.markdown("### Columns with at least one NaN value : {} ".format(train_res.columns[train_res.isnull().any()].tolist()))
            st.markdown("### Columns with all NaN value : {} ".format(train_res.columns[train_res.isnull().all()].tolist()))
            st.markdown("### Number of NaN values: {}".format(train_res.isnull().sum().sum()))
            st.subheader("=======================================================")

            # Feature Counts
            st.markdown("### Feature Counts ")
            for i in range(1, len(train_res.columns)):
                st.write("####### " + train_res.columns[i] + " #######")
                st.write(train_res.iloc[:,i].value_counts())
            
            st.subheader("=======================================================")
            st.subheader("Pair plot of train.csv column/features")
            sb.pairplot(train_res, hue='tool_condition', vars=["feedrate","clamp_pressure"])
            st.pyplot()

        elif extract_button == "experiments.csv Information":
            st.title("Data Insights of experiments.csv files")
            file_analytics = st.sidebar.selectbox("Select an experiment file:", files)
            exp_file = pd.read_csv(file_analytics)
            index = files.index(file_analytics)

            st.subheader("DataFrame (head): ")
            st.write(exp_file.head())
            st.subheader("=======================================================")

            st.subheader("\nFeature columns and types:")
            st.write(exp_file.dtypes)
            st.subheader("=======================================================")

            # checking for NaN columns [df.isnull().sum().sum()]
            st.subheader("\n \n NaN Check")
            st.markdown("### Columns with at least one NaN value : {} ".format(exp_file.columns[exp_file.isnull().any()].tolist()))
            st.markdown("### Columns with all NaN value : {} ".format(exp_file.columns[exp_file.isnull().all()].tolist()))
            if len(exp_file.columns[exp_file.isnull().all()].tolist()) >= 1:
                for column in exp_file.columns[exp_file.isnull().all()].tolist():
                    exp_file.drop(column)
            st.subheader("=======================================================")

            # Integrating experiment files with labels and features from train.csv 
            files_i = list()
            exp_number = ''
            if index < 9:
                exp_number = '0' + str(index+1)
            elif index == 9:
                exp_number = str(index + 1)
            else:
                exp_number = str(index+1)
            file = pd.read_csv("data/experiment_{}.csv".format(exp_number))
            row = train_res[train_res['No'] == index+1]
                
            #add experiment settings to features
            file['feedrate']=row.iloc[0]['feedrate']
            file['clamp_pressure']=row.iloc[0]['clamp_pressure']
                
            # Having label as 'tool_conidtion'    
            file['label'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
            files_i.append(file)

            df = pd.concat(files_i, ignore_index = True)

            st.subheader("Integrated experiment DataFrame with label/clamp_presssure/feedrate")
            pro={'Layer 1 Up':1,'Repositioning':2,'Layer 2 Up':3,'Layer 2 Up':4,'Layer 1 Down':5,
                'End':6,'Layer 2 Down':7,'Layer 3 Down':8,'Prep':9,'end':10,'Starting':11}
            data=[df]
            for dataset in data:
                dataset['Machining_Process']=dataset['Machining_Process'].map(pro)
            
            st.write(df.head())
            st.subheader("=======================================================")
            st.subheader("Integrated DataFrame summary (experiment file + train_csv row)")
            st.write(df.describe())
            st.subheader("=======================================================")        
            # Highly Co related features 
            st.markdown("### Choose the number of corelated pairs you want to analyze")
            value_n = st.slider("Highest Corelated features", 1, len(df.columns[0]), 1)
            top_corr = get_top_abs_correlations(df, n=value_n)
            list_corr = []
            tuple_list = []
            for a, b in top_corr.index:
                list_corr.append(a)
                list_corr.append(b)
                tuple_list.append((a,b))
            list_corr = list(dict.fromkeys(list_corr))
            st.markdown("Top Highly Corelated features from the data (most corelated to least)")
            st.write(tuple_list)

            st.subheader("=======================================================")        
            button = st.sidebar.selectbox("Correlation and PairPlot for experiment_{} file".format(exp_number), 
                                          ["Select", "Correlation map", "Pair plot"])
            if button == "Select":
                st.subheader("Choose from 'Correlation and PairPlot for experiment_01 file' sidebar.")
            elif button == "Correlation map":
                st.subheader("Correlation betwen all variables of experiment_{} file".format(exp_number))
                plt.figure(figsize=(50, 40))
                p = sb.heatmap(df.corr())
                st.pyplot()
            else:
                st.subheader("Pair plot for the highly correlated feature columns")
                st.subheader("Label is {} which is {}".format(df["label"][0], "Worn" if df["label"][0] == 1 else "Unworn"))
                sb.pairplot(df.sample(int(len(df)/2)), hue='label', vars=list_corr[:7])
                st.pyplot()
            
            st.subheader("=======================================================")
            st.subheader("Click checkbox to")
            checkbox = st.checkbox("Show pairplot analysis for all experiment files")
            if checkbox:
                st.subheader("Exploratory pair plot analysis of all experiment files")
                st.subheader("Labels: 0 = {} and 1 = {}".format("Unworn", "Worn"))
                #Integrating all experiment files as one data corpus
                all_files, mapped_files = integrate_all_files(train_res)
                sb.pairplot(mapped_files.sample(int(len(mapped_files)/4)), hue='label', vars= list_corr[:7])
                st.pyplot()
            
        else:
            st.title("Data Processing for Model Training")
            st.markdown("For Model training, we integrate all the experimentation files and \
                        also introduce labels for the each experimentation observation, while considering each observation\
                        independent from the other (marking them worn or unworn)")
            all_files, mapped_files = integrate_all_files(train_res)
            shuffled, data = shuffle_split_data(all_files)

            st.markdown("An instance of the data will be like: ")
            st.write(mapped_files.iloc[0])

            st.markdown("You can scroll through the above row and look for the columns included. \
                        You can also look at how feed_rate, clamp_pressure and label has been \
                        integrated into the data augmentation for model training. ")

            st.subheader("=======================================================")
            st.markdown("### Length of the total dataset: {} rows and {} columns".format(len(mapped_files), len(mapped_files.columns)))

            st.markdown("#### Looking from the bar plot below we can see that the labels are somewhat balanced for worn and unworn.")
            sb.countplot(x='label', data=all_files)
            st.pyplot()

            st.subheader("=======================================================")
            st.markdown("### Count relationship between Machining Process and label(tool condition).")
            sb.catplot(x='label', col='Machining_Process', col_wrap=2, kind='count', data=all_files)
            st.pyplot()

            st.subheader("=======================================================")
            st.markdown("### Unshuffled data")
            st.write(mapped_files.head(7))

            st.subheader("=======================================================")
            st.markdown("### Shuffled data")
            st.write(shuffled.head(7))

            st.subheader("=======================================================")
            st.markdown("### Splitting into train and test set using sklearn's test_train_split()")
            st.markdown("#### Length of X_train: {}".format(len(data[0]))) 
            st.markdown("#### Length of X_test is: {}".format (len(data[1])))
            st.subheader("=======================================================")

    elif option == "Model":
        intro_model()
        st.title("Model Training")
        _, all_files = integrate_all_files(train_res)
        shuffled, data = shuffle_split_data(all_files)
        st.markdown("#### Click the button below to run the models on the data: ")
        button_models = st.button("Run Classifiers")
        if button_models:
            res_model = model_insights(data[0], data[2], data[1], data[3])

            st.subheader("Saved models..")
            st.write(res_model)
            with open('filename.pickle', 'wb') as handle:
                pickle.dump(res_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            handle.close()

    elif option == "Results":
        st.title("Results")
        _, all_files = integrate_all_files(train_res)
        shuffled, data = shuffle_split_data(all_files)
        with open('filename.pickle', 'rb') as handle:
                b = pickle.load(handle)
        
        handle.close()

        dropdown_arr = [v for v in b.keys()]
        st.subheader("You have to first run the Models section to be able to run results section, as the trianed models \
                     are then used for the results/insights.")
        drop_model = st.selectbox("Select the model you want insights about", dropdown_arr)
        st.write("## {}".format(drop_model))
        model = b[drop_model]

        # Accuracy and Confusion Matrix
        st.markdown("#### Accuracy is : {}%".format(100*accuracy_score(data[3], model.predict(data[1]))))
        metrics = precision_recall_fscore_support(data[3], model.predict(data[1]))
        st.markdown("#### Precision of class 0 is : {} and class 1 is : {}".format(round(metrics[0][0]*100, 2), round(metrics[0][1]*100, 2)))
        st.markdown("#### Recall of class 0 is : {} and class 1 is : {}".format(round(metrics[1][0]*100, 2), round(metrics[1][1]*100, 2)))
        st.markdown("#### F1-score of class 0 is : {} and class 1 is : {}".format(round(metrics[2][0]*100, 2), round(metrics[2][1]*100, 2)))

        st.subheader("=======================================================")
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, data[1], data[3])
        st.pyplot()

        st.subheader("=======================================================")
        st.subheader("ROC Curve")
        plot_roc_curve(model, data[1].to_numpy(), data[3].to_numpy().reshape(-1,1))
        st.pyplot()


        # Feature importance in the model
        if drop_model == "RandomForestClassifier" or drop_model == "DecisionTreeClassifier":
            st.subheader("=======================================================")
            st.subheader("Feature Importance in the trained model")
            importance = model.feature_importances_
            feature_importance_tree(importance)
        elif drop_model == "LogisticRegression" or drop_model == "Perceptron" or drop_model == "SGDClassifier" or drop_model == "LinearSVC":
            st.subheader("=======================================================")
            st.subheader("Feature Importance in the trained model")
            importance = model.coef_
            feature_importance_reg(importance, shuffled)
        
        # Accuracy, Performance, Scalability plots
        st.subheader("=======================================================")
        st.subheader("Scalibility of the model")
        plt_second(model, "Scalability of {}".format(drop_model), data[0], data[2], n_jobs=6)
        st.subheader("=======================================================")
        st.subheader("Performance of the model")
        plt_third(model, "Performance of {}".format(drop_model), data[0], data[2], n_jobs=6)


if __name__=="__main__":
    main_app()




