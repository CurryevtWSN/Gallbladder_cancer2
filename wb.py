#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
import openpyxl
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Prediction system for gallbladder cancer distant metastasis:a retrospective cohort study based on machine learning')
st.title('Prediction system for gallbladder cancer distant metastasis:a retrospective cohort study based on machine learning')

#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')
Combined_Summary_Stage =  st.sidebar.selectbox("Combined Summary Stage",('Stage I', 'Stage II', "Stage III"),index=2)
T = st.sidebar.selectbox("T Recode", ('T0','T1','T2','T3','T4','TX'), index = 4)
Surgery_Recode = st.sidebar.selectbox("Surgery Recode", ('No',"Yes"), index = 1)
Age =  st.sidebar.slider("Age (year)", 5,95,value = 65, step = 1)
N = st.sidebar.selectbox("N Recode", ('N0','N1','N2','NX'), index = 2)
Median_household_income_inflation_adj_to_2021 = st.sidebar.selectbox("Median household income inflation(2021)", 
                                                                     ('<$50,000','$50,000-$60,000','$60,000-$70,000','>$70,000'), index = 2)
Radiation_recode = st.sidebar.selectbox("Radiation Recode", ('No',"Yes"), index = 1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Shi-Nan Wu, Xiamen university')
#传入数据
map = {'No':0,
       'Yes':1,
       'Stage I':0, 
       'Stage II':1, 
       "Stage III":2,
       'T0':0,
       'T1':1,
       'T2':2,
       'T3':3,
       'T4':4,
       'TX':5,
       'N0':0,
       'N1':1,
       'N2':2,
       'NX':3,
       '<$50,000':0,
       '$50,000-$60,000':1,
       '$60,000-$70,000':2,
       '>$70,000':3
}
N =map[N]
T = map[T]
Combined_Summary_Stage = map[Combined_Summary_Stage]
Surgery_Recode = map[Surgery_Recode]
Median_household_income_inflation_adj_to_2021 = map[Median_household_income_inflation_adj_to_2021]
Radiation_recode = map[Radiation_recode]

# 数据读取，特征标注
#%%load model
xgb_model = joblib.load('xgb_model.pkl')

#%%load data
hp_train = pd.read_excel("data.xlsx", sheet_name="Sheet1")
features = ["Combined_Summary_Stage","T","Surgery_Recode","Age","N","Median_household_income_inflation_adj_to_2021","Radiation_recode"]
target = ["M"]
y = np.array(hp_train[target])
sp = 0.5

is_t = (xgb_model.predict_proba(np.array([[Combined_Summary_Stage,T,Surgery_Recode,Age,N,Median_household_income_inflation_adj_to_2021,Radiation_recode]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[Combined_Summary_Stage,T,Surgery_Recode,Age,N,Median_household_income_inflation_adj_to_2021,Radiation_recode]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[Combined_Summary_Stage,T,Surgery_Recode,Age,N,Median_household_income_inflation_adj_to_2021,Radiation_recode]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB model")
    disp1 = plt.show()
    st.pyplot(disp1)

