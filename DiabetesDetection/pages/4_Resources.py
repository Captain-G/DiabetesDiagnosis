import streamlit as st

def main():
    st.header("Project Resources")

    st.write("Dataset Link on Kaggle")
    st.write("https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset")

    st.write("-----------------------------------------------")

    st.write("Project Link on Github")
    st.write("https://github.com/Captain-G/DiabetesDiagnosis")

    st.write("--------------------------------------------")

    st.write("Download the Dataset")

    st.download_button(
        label="Download Dataset file",
        data="dataset/diabetes.csv",
        file_name='diabetes.csv',
        mime='text/csv',
    )

    st.write("-----------------------------------")


    st.write("Download the Trained Models")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Open the pickle file in binary mode
        with open("models/diabetes_log_reg.pkl", "rb") as file:
            log_reg_pkl = file.read()

        # Create a download button for the pickle file
        st.download_button(
            label="Download Logistic Regression Model",
            data=log_reg_pkl,
            file_name="model.pkl",
            mime="application/octet-stream"
        )
    with col2:
        # Open the pickle file in binary mode
        with open("models/diabetes_random_forest.pkl", "rb") as file:
            ran_forest_pkl = file.read()

        # Create a download button for the pickle file
        st.download_button(
            label="Download Random Forest Model",
            data=ran_forest_pkl,
            file_name="model.pkl",
            mime="application/octet-stream"
        )
    with col3:
        # Open the pickle file in binary mode
        with open("models/diabetes_svc.pkl", "rb") as file:
            svc_pkl = file.read()

        # Create a download button for the pickle file
        st.download_button(
            label="Download Support Vector Classifier Model",
            data=svc_pkl,
            file_name="model.pkl",
            mime="application/octet-stream"
        )


    st.write("-----------------------------------")

    st.write("Download the Report")

    st.download_button(
        label="Download Project Report",
        data="report/DiabetesCaseStudyReport",
        file_name='report.pdf',
        mime='text/pdf',
    )



if __name__ == "__main__":
    main()