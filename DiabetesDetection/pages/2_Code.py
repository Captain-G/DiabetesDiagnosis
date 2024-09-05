import streamlit as st


def main():
    st.header("Diabetes Detection App")
    st.subheader("Dataset Overview")
    st.write("""
    Source: The dataset was sourced from https://kaggle.com and originally comes from the
National Institute of Diabetes and Digestive and Kidney Diseases.

Link: https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset/data

Structure: The dataset consists of 768 rows and 9 columns.
    """)

    st.write("---------------------------")
    st.write("Importation of Libraries")

    st.code("""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score,recall_score, f1_score ,classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        
        import warnings
        
        warnings.filterwarnings('ignore')
    
    """)

    st.write("Importing the Dataset")
    st.code("""
    df = pd.read_csv("drive/MyDrive/datasets/diabetes.csv")
    df.head(10)
    """)

    st.write("Carried out Exploratory Data Analysis. Refer to the Jupyter Notebook in Resources/Github")

    st.write("There were outliers on majority of the feature columns, so I handled them by capping them out.")
    st.code("""
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
                print(f"{column}: has outliers")
                # Cap outliers
                data[column] = np.where(data[column] < low_b, low_b,
                                    np.where(data[column] > high_b, high_b, data[column]))
            else:
                print(f"{column}: no outliers")


    return data

num_col = df.select_dtypes(include='number').columns
df = handle_outliers(df, num_col)
    
    """)

    st.write("----------------------------")
    st.write("Data Preprocessing")
    st.write("Splitting the features and target data")
    st.code("""
    x=df.drop('Outcome',axis=1)
    y=df.Outcome
    """)

    st.write("Scaling the data using Sklearn's Standard Scalar")
    st.code("""
    scale=StandardScaler()
    x_scale=scale.fit_transform(x)
    """)

    st.write("Splitting the training and target data")
    st.code("""
    x_train , x_test ,y_train ,y_test=train_test_split(x_scale,y,test_size=0.2,random_state=0)
    """)

    st.write("--------------------")
    st.write("Model Building")
    st.code("""
    log_reg = LogisticRegression(class_weight='balanced')
    rf_clf=RandomForestClassifier()
    svc = SVC(probability=True)    
    """)

    st.write(
        "A Function that trains the three models: Support Vector Classifier, Logistic Regression and Random Forest Classifier, and gets their Performance Metrics")

    st.code("""
    def all(model):
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
    
        accuracy_test=accuracy_score(y_pred,y_test)*100
        accuracy_train=model.score(x_train,y_train)*100
        recall_result=recall_score(y_pred,y_test)*100
        f1_result=f1_score(y_pred,y_test)*100
    
        test_score.append(accuracy_test)
        train_score.append(accuracy_train)
        rec_score.append(recall_result)
        f_score.append(f1_result)
    
        print('Accuracy after train the model is :',accuracy_train)
        print('Accuracy after test the model is :',accuracy_test)
        print('Result recall score is :',recall_result)
        print('Result F1 score is :',f1_result)
    """)

    st.write(
        "The Logistic Regression Model worked just fine. However, the other well performed well on training but on testing the models with new unseen data, they were terrible. Reason being, the dataset was imbalanced. To address this, I used SMOTE.")
    st.code("""
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    
    # Train the logistic regression model on the resampled dataset
    log_reg.fit(x_train_resampled, y_train_resampled)
    
    new_data = np.array([[2, 197, 70, 45, 318.125, 30.5, 0.158, 53]])
    new_data_scaled = scale.transform(new_data)
    y_new_proba = log_reg.predict_proba(new_data_scaled)
    print(f"Probability of No Diabetes : {round(y_new_proba[0][0] * 100, 2)}%")
    print(f"Probability of Diabetes : {round(y_new_proba[0][1] * 100, 2)}%")
    
    """)

    st.write("Comparing the Models Performances against each other")
    st.code("""
    plt.figure(figsize=(15, 8))
    bar_width = 0.2
    xpos=np.arange(len(columns))
    bars1=plt.bar(xpos - 0.3, train_score, width=bar_width, label="train_score",color='lightblue')
    bars2=plt.bar(xpos - 0.1, test_score, width=bar_width, label="test_score",color='lightpink')
    bars3=bars=plt.bar(xpos + 0.1, rec_score, width=bar_width, label="recall_score",color='lightgreen')
    bars4=plt.bar(xpos + 0.3, f_score, width=bar_width, label="f1_score",color='plum')
    
    plt.xticks(xpos, columns)
    plt.legend()
    plt.xlabel("Models",fontsize=15)
    plt.ylabel("Scores",fontsize=15)
    plt.title("Model Performance Comparison",fontsize=30)
    plt.show()

    """)

    st.write("Feature Importance Extraction")
    st.code("""
    rf_clf.fit(x,y)
    feature_importances = rf_clf.feature_importances_
    
    # Create a DataFrame with the feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': x.columns,
        'Importance': feature_importances
    })
    plt.figure(figsize=(11, 8))
    
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)
    bars=sns.barplot(x='Importance',y='Feature',data=feature_importances_df,color='lightpink')
    bars.patches[0].set_hatch('/')
    plt.ylabel('Feature',fontsize=15)
    plt.xlabel('Importance',fontsize=15)
    plt.title('Feature Importances from Random Forest Model',fontsize=18)
    plt.xticks(rotation=90)
    plt.show()
    """)

    st.write("Model Evaluation using a Confusion Matrix")
    st.code("""
    def cm(model):
    y_pred=model.predict(x_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    print(model)
    plt.show()
    """)

    st.write("---------------------------")
    st.write("The full code can be seen on my github - Visit the Resources Page for Link")

    st.write("*Property of Peter Gachuki*")
    st.write("---------------------------")



if __name__ == "__main__":
    main()
