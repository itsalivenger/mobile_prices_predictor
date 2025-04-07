import pandas as pd
import numpy as np
import streamlit as st
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, accuracy_score

# Importation des donnees
df = pd.read_csv('mobile_prices.csv')

# Separation des features continues a normaliser
continuous_Features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']
non_continuous = [feature for feature in df.columns if feature not in continuous_Features + ['price_range']]

# Normalisation
scaler = StandardScaler()
minMaxNormalizer = MinMaxScaler()

continuous_data = df[continuous_Features]
normalized_data = scaler.fit_transform(continuous_data)
minMaxed_scaled_data = minMaxNormalizer.fit_transform(continuous_data)

# datasets finaux
df_scaled = pd.DataFrame(normalized_data, columns=continuous_Features)
naive_scaled = pd.DataFrame(minMaxed_scaled_data, columns=continuous_Features)

finalData = pd.concat([df_scaled, df[non_continuous + ['price_range']]], axis=1)
naive_data = pd.concat([naive_scaled, df[non_continuous + ['price_range']]], axis=1)

dataX = finalData.drop('price_range', axis=1)
dataY = finalData['price_range']

naive_bayes_X = naive_data.drop('price_range', axis=1)
naive_bayes_Y = naive_data['price_range']

# Train/test split
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.2, random_state=123)
naive_trainX, naive_testX, naive_trainY, naive_testY = train_test_split(naive_bayes_X, naive_bayes_Y, test_size=0.2, random_state=123)

# interface streamlit
st.header("Mobile Price Classification - Model Comparison")
st.subheader("Model Comparison by Ali Ait Houssa")

model_choice = st.selectbox("Choose a model", ["Logistic Regression", "KNN", "Naive Bayes", "Decision Tree", "Random Forest"])

dossier_sauvegarde_models = "saved_models"
os.makedirs(dossier_sauvegarde_models, exist_ok=True)

input_data = {}
with st.expander("Enter Mobile Specs for Prediction"):
    for feature in continuous_Features + non_continuous:
        input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))

if model_choice == "KNN":
    k = st.slider("Select number of neighbors (k)", 1, 50, 5)
    model_path = f"{dossier_sauvegarde_models}/knn_k{k}.joblib"
    if os.path.exists(model_path):
        knn = joblib.load(model_path)
    else:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        knn.fit(trainX, trainY)
        joblib.dump(knn, model_path)
        
    pred = knn.predict(testX)
    score = accuracy_score(testY, pred)
    cv = cross_val_score(knn, dataX, dataY, cv=5)
    st.write("Accuracy:", score)
    st.text(classification_report(testY, pred))
    st.write("Cross-validation scores:", cv)
    st.write("Mean CV Score:", cv.mean())

    # Prediction
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df[continuous_Features])
    full_input = pd.concat([pd.DataFrame(input_scaled, columns=continuous_Features), input_df[non_continuous]], axis=1)
    prediction = knn.predict(full_input)
    st.success(f"Predicted Class: {prediction[0]}")

else:
    model_path = f"{dossier_sauvegarde_models}/{model_choice.replace(' ', '_').lower()}.joblib"

    if model_choice == "Logistic Regression":
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = LogisticRegression()
            model.fit(trainX, trainY)
            joblib.dump(model, model_path)
        pred = model.predict(testX)
        cv = cross_val_score(model, dataX, dataY, cv=5)

    elif model_choice == "Naive Bayes":
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = GaussianNB()
            model.fit(naive_trainX, naive_trainY)
            joblib.dump(model, model_path)
        pred = model.predict(naive_testX)
        cv = cross_val_score(model, naive_bayes_X, naive_bayes_Y, cv=5)
        testY = naive_testY

    elif model_choice == "Decision Tree":
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            model = DecisionTreeClassifier()
            model.fit(trainX, trainY)
            joblib.dump(model, model_path)
        pred = model.predict(testX)
        cv = cross_val_score(model, dataX, dataY, cv=5)

    elif model_choice == "Random Forest":
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            base_tree = DecisionTreeClassifier()
            model = BaggingClassifier(base_tree, n_estimators=10)
            model.fit(trainX, trainY)
            joblib.dump(model, model_path)
        pred = model.predict(testX)
        cv = cross_val_score(model, dataX, dataY, cv=5)

    st.write("Accuracy:", accuracy_score(testY, pred))
    st.text(classification_report(testY, pred))
    st.write("Cross-validation scores:", cv)
    st.write("Mean CV Score:", cv.mean())

    # Prediction
    input_df = pd.DataFrame([input_data])
    if model_choice == "Naive Bayes":
        input_scaled = minMaxNormalizer.transform(input_df[continuous_Features])
        full_input = pd.concat([pd.DataFrame(input_scaled, columns=continuous_Features), input_df[non_continuous]], axis=1)
        prediction = model.predict(full_input)
    else:
        input_scaled = scaler.transform(input_df[continuous_Features])
        full_input = pd.concat([pd.DataFrame(input_scaled, columns=continuous_Features), input_df[non_continuous]], axis=1)
        prediction = model.predict(full_input)

    st.success(f"Predicted Class: {prediction[0]}")
