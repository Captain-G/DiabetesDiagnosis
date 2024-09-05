from random import random
import streamlit as st
import pandas as pd
import numpy as np
import math
import pickle as pkl


def handle_outliers(data, columns):
    for column in columns:
        if column in data.columns:
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            low_b = q1 - 1.5 * iqr
            high_b = q3 + 1.5 * iqr

            outliers = (data[column] < low_b) | (data[column] > high_b)

            if outliers.any():
                data[column] = np.where(data[column] < low_b, low_b,
                                        np.where(data[column] > high_b, high_b, data[column]))
            else:
                pass

    return data


def logisticRegression(scaler, new_data, model_log_reg):
    st.write("-------------------------")
    st.write("Logistic Regression")
    st.write("-------------------------")
    new_data_scaled = scaler.transform(new_data)
    y_new_proba = model_log_reg.predict_proba(new_data_scaled)
    st.write(f"No Diabetes : {round(y_new_proba[0][0] * 100, 2)}% Chance")
    st.write(f"Diabetes : {round(y_new_proba[0][1] * 100, 2)}% Chance")


def randomForestClassifier(scaler, new_data, model_random_forest):
    st.write("-------------------------")
    st.write("Random Forest Classifier")
    st.write("-------------------------")
    new_data_scaled = scaler.transform(new_data)
    y_new_proba = model_random_forest.predict_proba(new_data_scaled)
    st.write(f"No Diabetes : {round(y_new_proba[0][0] * 100, 2)}%")
    st.write(f"Diabetes : {round(y_new_proba[0][1] * 100, 2)}%")


def supportVectorClassifier(scaler, new_data, model):
    st.write("-------------------------")
    st.write("Support Vector Classifier")
    st.write("-------------------------")
    new_data_scaled = scaler.transform(new_data)
    y_new_proba = model.predict_proba(new_data_scaled)
    st.write(f"No Diabetes : {round(y_new_proba[0][0] * 100, 2)}% Chance")
    st.write(f"Diabetes : {round(y_new_proba[0][1] * 100, 2)}% Chance")


def main():
    st.header("Diabetes Detection App")
    df = pd.read_csv("dataset/diabetes.csv")

    num_col = df.select_dtypes(include='number').columns
    df = handle_outliers(df, num_col)

    with open('models/diabetes_svc.pkl', 'rb') as file:
        model_svc = pkl.load(file)

    with open('models/diabetes_log_reg.pkl', 'rb') as file:
        model_log_reg = pkl.load(file)

    with open('models/diabetes_random_forest.pkl', 'rb') as file:
        model_random_forest = pkl.load(file)

    with open('models/scaler.pkl', 'rb') as file:
        scaler = pkl.load(file)

    pregnancies = st.slider("Enter the Number of Pregnancies that the Subject has had",
                            min_value=int(df["Pregnancies"].min()), max_value=int(df["Pregnancies"].max()), step=1,
                            value=math.ceil(df["Pregnancies"].mean()))
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Enter the Glucose concentration in the Blood", min_value=df["Glucose"].min(),
                                  max_value=df["Glucose"].max(), value=df["Glucose"].mean(), step=1.0)
    with col2:
        blood_pressure = st.number_input("Enter the Blood Pressure Measurement",
                                         min_value=int(df["BloodPressure"].min()),
                                         max_value=int(df["BloodPressure"].max()), step=1,
                                         value=int(math.ceil(df["BloodPressure"].mean())))
    skin_thickness = st.slider("Enter the Tricep Skinfold Thickness", min_value=int(df["SkinThickness"].min()),
                               max_value=int(df["SkinThickness"].max()), step=1,
                               value=math.ceil(df["SkinThickness"].mean()))
    insulin = st.slider("Enter the Serum Insulin Levels", min_value=int(df["Insulin"].min()),
                        max_value=int(df["Insulin"].max()), step=1, value=math.ceil(df["Insulin"].mean()))
    bmi = st.number_input("Enter the Body Mass Index", min_value=float(df["BMI"].min()),
                          max_value=float(df["BMI"].max()), step=1.0, value=float(df["BMI"].mean()))
    diabetes_pedigree_function = st.slider("Enter the Genetic Predisposition to Diabetes Function",
                                           min_value=df["DiabetesPedigreeFunction"].min(),
                                           max_value=df["DiabetesPedigreeFunction"].max(),
                                           value=df["DiabetesPedigreeFunction"].mean())
    age = st.slider("Enter the Age of the Subject", min_value=int(df["Age"].min()), max_value=int(df["Age"].max()),
                    step=1, value=math.ceil(df["Age"].mean()))

    models = st.multiselect("Select the Models that you want to use",
                            ["Logistic Regression", "Support Vector Classifier", "Random Forest Classifier"])

    if st.button("Show Prediction"):
        if len(models) == 0:
            st.warning("Select at least 1 Model")
        else:
            new_data = np.array(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            )

            if len(models) == 1:
                if "Support Vector Classifier" in models:
                    supportVectorClassifier(scaler, new_data, model_svc)
                if "Logistic Regression" in models:
                    logisticRegression(scaler, new_data, model_log_reg)
                if "Random Forest Classifier" in models:
                    randomForestClassifier(scaler, new_data, model_random_forest)



            elif len(models) == 2:
                col1, col2 = st.columns(2)
                with col1:
                    if "Support Vector Classifier" in models:
                        supportVectorClassifier(scaler, new_data, model_svc)
                        models.remove("Support Vector Classifier")
                    elif "Logistic Regression" in models:
                        logisticRegression(scaler, new_data, model_log_reg)
                        models.remove("Logistic Regression")
                    elif "Random Forest Classifier" in models:
                        randomForestClassifier(scaler, new_data, model_random_forest)
                        models.remove("Random Forest Classifier")
                with col2:
                    if "Support Vector Classifier" in models:
                        supportVectorClassifier(scaler, new_data, model_svc)
                        models.remove("Support Vector Classifier")
                    elif "Logistic Regression" in models:
                        logisticRegression(scaler, new_data, model_log_reg)
                        models.remove("Logistic Regression")
                    elif "Random Forest Classifier" in models:
                        randomForestClassifier(scaler, new_data, model_random_forest)
                        models.remove("Random Forest Classifier")


            elif len(models) == 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if "Support Vector Classifier" in models:
                        supportVectorClassifier(scaler, new_data, model_svc)
                        models.remove("Support Vector Classifier")
                    elif "Logistic Regression" in models:
                        logisticRegression(scaler, new_data, model_log_reg)
                        models.remove("Logistic Regression")
                    elif "Random Forest Classifier" in models:
                        randomForestClassifier(scaler, new_data, model_random_forest)
                        models.remove("Random Forest Classifier")
                with col2:
                    if "Support Vector Classifier" in models:
                        supportVectorClassifier(scaler, new_data, model_svc)
                        models.remove("Support Vector Classifier")
                    elif "Logistic Regression" in models:
                        logisticRegression(scaler, new_data, model_log_reg)
                        models.remove("Logistic Regression")
                    elif "Random Forest Classifier" in models:
                        randomForestClassifier(scaler, new_data, model_random_forest)
                        models.remove("Random Forest Classifier")

                with col3:
                    if "Support Vector Classifier" in models:
                        supportVectorClassifier(scaler, new_data, model_svc)
                        models.remove("Support Vector Classifier")
                    elif "Logistic Regression" in models:
                        logisticRegression(scaler, new_data, model_log_reg)
                        models.remove("Logistic Regression")
                    elif "Random Forest Classifier" in models:
                        randomForestClassifier(scaler, new_data, model_random_forest)
                        models.remove("Random Forest Classifier")


if __name__ == "__main__":
    main()
