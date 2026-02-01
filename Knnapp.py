#### Data Generation
import numpy as np
import pandas as pd
import streamlit as st
# pd.set_option('display.max_rows',None)
# pd.reset_option('display.max_rows')
st.title("📘 Student Result Prediction App")
st.write("Predict result based on **Study Hours** and **IQ** using KNN")

np.random.seed(42)
n = 500
study_hours = np.round(np.random.uniform(1,9,n),2)
# study_hours
intelligent_qutient = np.random.randint(55,146,n)
# intelligent_qutient
df = pd.DataFrame({
    "study_hours":study_hours,
    "iq":intelligent_qutient,
})
df.head()
df['result'] = (df["study_hours"] * df["iq"] >= 400).astype(int) 
df.sample(10)
#### Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
df.hist(figsize=(12,10))
plt.show()
sns.scatterplot(x="study_hours",y="iq",hue="result",data=df)
plt.show()
#### Modeling
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
X = df[["study_hours","iq"]]
# X
y = df["result"]
# y
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train)
predictions = model.predict(x_test)
accuracy_score(y_test,predictions)
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))
#### Model to app
study_input = st.number_input(
    "Enter Study Hours",
    min_value=1.0,
    max_value=9.0,
    step=0.1
)

iq_input = st.number_input(
    "Enter IQ",
    min_value=55,
    max_value=145,
    step=1
)

if st.button("Predict Result"):
    user_data = np.array([[study_input, iq_input]])

    prediction = model.predict(user_data)

    if prediction == 1:
        st.success("🎉 Result: PASS")
        st.balloons()
    else:
        st.error("❌ Result: FAIL")

