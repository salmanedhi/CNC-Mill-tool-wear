import streamlit as st

def print_insight_info():
    st.info("This section will give you an insight into the data, \
             how it looks, how is the data structured, what are the columns \
             what is the useful information etc")

def intro_info():
    st.info("This section will give a brief insight to the problem statement \
             and a brief outline for our approach, so that the audience is caught up to the project before \
            diving deeper into the analytics and model.")

def intro_model():
    st.info("This section will deal with trying different machine learning models for the \
                clasification task at hand which to predict whether the tool being used \
                in the CNC process is worn or not. We will be using multiple machine learning \
                models for our predictions and compare their performance to finally \
                present our results.")
    
def print_intro():
    st.markdown("### Introduction:")
    st.markdown("Cutting tool wear degrades the product quality in manufacturing processes. \
                Monitoring tool wear value online is therefore needed to prevent degradation in machining quality. \
                Moroever ensuring the tool wear condition in real time based on different features of the mining machine \
                can help real time diagnosis and quick response from the responsible authorities to maintain the \
                quality of the production units that are being mined by the mining tool.")
    
    st.markdown("### Goal:")
    st.markdown("Our goal is to develop an algorithm which given attributes like velocity, position, co ordinates and acceleration \
                 of the mining tool, can figure out whether the tool is worn our or not with good accuracy and confidence.")
    
    st.markdown("### Context:")
    st.markdown("A series of machining experiments were run on 2' x 2' x 1.5' wax blocks in a CNC milling machine in the System-level Manufacturing and Automation Research Testbed (SMART) at the University of Michigan. \
                Machining data was collected from a CNC machine for variations of tool condition, feed rate, and clamping pressure. Each experiment produced a finished wax part with an 'S' shape - S for smart manufacturing \
                - carved into the top face, as shown in test_artifact.jpg (included in the dataset).")
    
    st.markdown("### Inspiration:")
    st.markdown("The dataset can be used in classification studies such as: \
                \n\n 1. Tool wear detection --- Supervised binary classification could be performed for identification of worn and unworn cutting tools. Eight experiments were run with an unworn tool while ten were run with a worn tool (see tool_condition column for indication). \
                \n\n 2. Detection of inadequate clamping --- The data could be used to detect when a workpiece is not being held in the vise with sufficient pressure to pass visual inspection (see passedvisualinspection column for indication of visual flaws). \
                Experiments were run with pressures of 2.5, 3.0, and 4.0 bar. The data could also be used for detecting when conditions are critical enough to prevent the machining operation from completing (see machining_completed column for indication of when machining was preemptively stopped due to safety concerns).")
    
    st.markdown("### GUI Design:")
    st.markdown("GUI is divided into three sections:\
                \n\n 1. **EDA/Feature Engineering**:\
                    \n In this section we will dive deeper into the data and how it is structured. \
                \n\n 2. **Model**: \
                    \n In this section we will run multiple classifiers to get an insight on how different \
                    classifiers perform with the classification task at hand. \
                \n\n 3. **Results**: \
                    \n In this section we will analyze our models with different metrics and plots, to judge \
                    the models on the task.")

def print_model_info(key):
    if key == "DummyClassifier":
        st.write("Dummy classifier is used as a baseline for classification tasks to compare with other\
                    classifiers. it predicts by looking at the distribution of training set classes that is \
                        the class that maximises the class prior (most frequent).")
    
    elif key == "LogisticRegression":
        st.write("It is used to predict the probability of a certain class. \
                It is a statistical model that uses a logistic function to model \
                a binary dependent variable. In logistic regression the dependent (output) \
                variable is encoded in binary 1 (yes), 0(no). In other words, LR model predicts P(Y=1) as a function of X.")
    
    elif key == "RandomForestClassifier":
        st.write("Random forest classifiers come under the umbrella of ensemble learning. \
                They consruct multitude of decision trees during training and output the class that is the mode of the classes of individual trees. \
                They correct the problem of overfitting in decision trees.")
    
    elif key == "DecisionTreeClassifier":
        st.write("Decision Tree is a simple representation for classifying examples. \
                 It is a supervised machine learning algorithm where the predictor space is segment into a number of simple regions. They consist of : \
                \n 1. Nodes: where we test the value of certain attribute \
                \n 2. Edges: corresponds to the outcome of text and connects to the next node of leaf\
                \n 3. Lead nodes: terminal nodes that predict the outcome (class label).")

    elif key == "KNeighborsClassifier":
        st.write("It is a type of instance-based learning or lazy learning where the function is only approximated locally. it assumes that similar things exist in close proximity. \
                 The input consists of K nearest training samples while the output object is assigned to the class most common among its K nearest neighbors.\
                 No explicit training is done on KNN and it is sensitive to the local structure of the data.")
    
    elif key == "Perceptron":
        st.write("It is a supervised learning algorithm of binary classification. It is a kind of linear classifier that makes its predictions based on a linear function combining set of weights with the input (features). \
                  In the context of neural networks, perceptron is a single layer neuron using the unit step function. The value of unit step function is 0 for negative arguments and 1 for positive.")
    
    elif key == "GaussianNB":
        st.write("It is a supervised learning algorithm based on applying Bayes theorem with the assumptions of conditional indepedence \
                  between every pair of features given the value of class variable and the likelihood of features to be Gaussian.")
    
    elif key == "SGDClassifier":
        st.write("Stochastic Gradient Descent is a simple approach to fitting linear classifiers under convex loss functions such as Logistic Regression and Support Vector Machines. \
                 It is considered as an optimization technique. \
                 Its advantages are efficiency and ease of implementation while the disadvantages are that it is sensitive to feature scaling and requires a number of hyperparameters and iterations.")
    
