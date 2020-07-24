# CNC-Mining-tool-classification
CNC mining tool classification using streamlit

#### Organization of the code repository
1. Data folder contains all the experiment files and the train.csv file
2. **filename.pickle** has the models stored as a pickle dictionary for Result section in the GUI/Interface.
3. **print_utils.py** contains all the textual material that has been written on the website. 
4. **utils.py** contains all the helper functions that are required for the **data_analysis.py** file.
5. **data_analysis.py** file has the main code for the application. 

### data_analysis.py (main file)
- [x] main_app() is the main function of the application, where we have four sections described by st.sidebar.radio
- [x] Then we have four options. Introduction, EDA, Model and Results, which have further functionalities and functions which can be found in utils.py and will be discussed further. 
- [x] Introduction (line 21) has the intro_info and print_info functions which are in the print_utils.py functions describing what the section is all about. 
- [x] EDA/Feature Engineering (line 25) has the section code for the Exploratory data analysis and feature engineering, which has further functionalities in the sidebar (line 36)
- [x] In the model (line 210) section we run multiple models for benchmarks, using model_insights(line 218) function which is defined in utils.py
- [x] Results (line 227) has all the model results and insights which also uses functions from the utils.py file. 

#### If you want to run on local machine 
1. install streamlit (pip install streamlit)
2. install seaborn (pip install seaborn)
3. git clone repository
4. change directory to the repository (cd CNC-Mining-tool-classification)
5. execute command: streamlit run data_analysis.py 

#### To run on web
1. go to **cnctool.herokuapp.com**
2. THE GUI TAKES TIME TO PROCESS IN THE MODEL AND THE RESULTS SECTION BECAUSE EVERYTHING RUNS IN REAL TIME, and nothing is pre-loaded