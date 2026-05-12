import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Job Market Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("ds_salaries.csv")
    return df

df = load_data()

st.title("📊 Data Science Job Market Analytics")
st.markdown("Analysing global data science salaries and trends")

experience = st.sidebar.multiselect("Experience Level",
    options=df["experience_level"].unique(),
    default=df["experience_level"].unique())

df_filtered = df[df["experience_level"].isin(experience)]




avg_salary=round(df_filtered["salary_in_usd"].mean(),2)


totaljobs=len(df_filtered)

max_salary=df_filtered["job_title"].value_counts().index[0]

col1, col2, col3 = st.columns(3)

col1.metric("Average Salary(USD)", avg_salary)

col2.metric("Total Jobs", totaljobs)


col3.metric("Top Job Title", max_salary)

st.subheader("📊 Salary Trends")



avg_by_exp=df_filtered.groupby("experience_level")["salary_in_usd"].mean()
 
fig = px.bar(avg_by_exp, x=avg_by_exp.index, y="salary_in_usd", title="Average Salary by Experience Level",labels={"x": ""})
st.plotly_chart(fig, use_container_width=True)


salary_by_country=df_filtered.groupby("company_location")["salary_in_usd"].mean().sort_values(ascending=False).head(10)

fig = px.bar(salary_by_country, x=salary_by_country.index, y="salary_in_usd", title="Salary by Country (Top 10)",labels={"x": ""})
st.plotly_chart(fig, use_container_width=True)


top10_titles=df_filtered.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False).head(10)
fig = px.bar(top10_titles, x=top10_titles.index, y="salary_in_usd", title="Top 10 Job Titles",labels={"x": ""})
st.plotly_chart(fig, use_container_width=True)




#Linear Regression

ml_df = df[["job_title", "experience_level", "company_size", "remote_ratio", "salary_in_usd"]].dropna()


# Convert text to numbers
le_job = LabelEncoder()
le_exp = LabelEncoder()
le_size = LabelEncoder()

ml_df = ml_df.copy()
ml_df["job_title"] = le_job.fit_transform(ml_df["job_title"])
ml_df["experience_level"] = le_exp.fit_transform(ml_df["experience_level"])
ml_df["company_size"] = le_size.fit_transform(ml_df["company_size"])

# Train the model
X = ml_df[["job_title", "experience_level", "company_size", "remote_ratio"]]
y = ml_df["salary_in_usd"]

model = LinearRegression()
model.fit(X, y)


st.subheader("💰 Salary Predictor")

job_input = st.selectbox("Job Title", options=sorted(df["job_title"].unique()))
exp_input = st.selectbox("Experience Level", options=["EN", "MI", "SE", "EX"])
size_input = st.selectbox("Company Size", options=["S", "M", "L"])
remote_input = st.selectbox("Remote Ratio", options=[0, 50, 100])

if st.button("Predict Salary"):
    job_encoded = le_job.transform([job_input])[0]
    exp_encoded = le_exp.transform([exp_input])[0]
    size_encoded = le_size.transform([size_input])[0]
    
    prediction = model.predict([[job_encoded, exp_encoded, size_encoded, remote_input]])
    st.success(f"Predicted Salary: ${round(prediction[0], 2):,}")