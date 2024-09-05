import streamlit as st
import pandas as pd


def main():
    st.header("Model Performance Metrics")

    data = {
        'Model': ["Logistic Regression", "Random Forest Classifier", "Support Vector Classifier"],
        'Accuracy': ["81.82%", "79.87%", "80.52%"],
        'Recall': ["74.36%", "69.05%", "71.79%"],
        'F1 Score': ["67.44%", "65.17%", "65.12Z%"],
        'Average': ["74.54%", "71.36%", "72.48%"],
    }

    # Convert data into a DataFrame
    df = pd.DataFrame(data)

    st.table(df)

    st.write("""
    Accuracy: The overall correctness of the model's predictions was measured as the ratio
    of correctly predicted instances to the total instances.
    
    Recall: The ability of the model to correctly identify positive cases (patients with
    diabetes) was evaluated to understand how well the model performs in detecting the
    target condition.
    
    F1 Score: The harmonic mean of precision and recall was calculated to provide a
    balance between the two, especially useful in cases where the class distribution is
    imbalanced.
    """)

if __name__ == "__main__":
    main()