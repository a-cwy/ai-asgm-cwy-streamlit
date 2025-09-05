import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

cb_labels = ['High', 'Low', 'Medium']
svm_labels = ['', 'Low', 'Medium', 'High']

#setup models
cb = CatBoostClassifier()
cb.load_model('/workspaces/ai-asgm-cwy-streamlit/models/catboost_tuned.cbm')
rf = RandomForestClassifier()
rf = joblib.load('/workspaces/ai-asgm-cwy-streamlit/models/rf_tuned.joblib')
sd_scl = StandardScaler()
sd_scl = joblib.load('/workspaces/ai-asgm-cwy-streamlit/models/scaler.joblib')
svm = SVC()
svm = joblib.load('/workspaces/ai-asgm-cwy-streamlit/models/base_ovr_73.joblib')

st.set_page_config(
    page_title='AI Dashboard',
    layout='wide'
)
st.title("Prediction of Online Gaming Behaviour")
st.write("Predict the Engagement Level of players using CatBoost, RandomForest and SVM models.")

sessions_per_week = st.number_input("Sessions played per week", min_value=0, max_value=50)
avg_session_duration_mins = st.number_input("Average session duration (mins)", min_value=0, max_value=600)
achievements_unlocked = st.number_input("Achievements unlocked", min_value=0, max_value=100)
player_level = st.number_input("Player level", min_value=0, max_value=100)

if st.button("Predict"):
    X = pd.DataFrame(
        {
            'SessionsPerWeek':[sessions_per_week],
            'AvgSessionDurationMinutes':[avg_session_duration_mins],
            'AchievementsUnlocked':[achievements_unlocked],
            'PlayerLevel':[player_level]
        }
    )
    rf_pred = rf.predict(X)
    cb_pred = cb.predict(X)
    svm_pred = svm.predict(sd_scl.transform(X))

    st.write(X)
    st.write(f'CatBoost prediction: {cb_labels[int(cb_pred)]}')
    st.write(f'RandomForest prediction: {rf_pred[0]}')
    st.write(f'Support Vector machine prediction: {svm_labels[int(svm_pred)]}')