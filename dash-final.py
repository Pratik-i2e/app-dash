import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import datetime
import os

# --- Set Groq API credentials ---
groq_api_key = "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"
groq_api_base = "https://api.groq.com/openai/v1"
model_name = "llama3-70b-8192"

llm = ChatOpenAI(
    model_name=model_name,
    temperature=0.0,
    openai_api_key=groq_api_key,
    openai_api_base=groq_api_base
)

# --- Load Data ---
@st.cache_data
def load_data():
    fte_df = pd.read_csv("AUTOMATION DATA(FTE Data) (1).csv")
    finance_df = pd.read_csv("AUTOMATION DATA(Finance Data) (1).csv")
    commitment_df = pd.read_csv("AUTOMATION DATA(Commitment) (1).csv")

    fte_df['DATE'] = pd.to_datetime(fte_df['YEARMONTH'].astype(str) + '01', format='%Y%m%d')
    finance_df['DATE'] = pd.to_datetime(finance_df['GL_INVOICE_MONTHYEAR'].astype(str) + '01', format='%Y%m%d')
    fte_df['Year'] = fte_df['DATE'].dt.year
    fte_df['Quarter'] = fte_df['DATE'].dt.quarter
    finance_df['Year'] = finance_df['DATE'].dt.year
    finance_df['Quarter'] = finance_df['DATE'].dt.quarter

    fte_df['FTE_RATE'] = pd.to_numeric(fte_df['FTE_RATE'], errors='coerce').fillna(0)
    fte_df['FTE'] = pd.to_numeric(fte_df['FTE'], errors='coerce').fillna(0)
    fte_df['fte_cost'] = fte_df['FTE'] * fte_df['FTE_RATE']
    finance_df['GL_TRANS_AMOUNT'] = pd.to_numeric(finance_df['GL_TRANS_AMOUNT'], errors='coerce').fillna(0)

    fte_agg = fte_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg(
        total_fte=('FTE', 'sum'),
        avg_fte_rate=('FTE_RATE', 'mean'),
        fte_cost=('fte_cost', 'sum')
    ).reset_index()

    finance_agg = finance_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg(
        total_spend=('GL_TRANS_AMOUNT', 'sum'),
        invoice_count=('GL_INVOICE_NUMBER', 'count')
    ).reset_index()

    commitment_agg = commitment_df.groupby(['PROJECT_NAME']).sum(numeric_only=True).reset_index()

    merged_df = pd.merge(fte_agg, finance_agg, on=['PROJECT_NAME', 'Year', 'Quarter'], how='outer')
    merged_df = pd.merge(merged_df, commitment_agg, on=['PROJECT_NAME'], how='outer')
    merged_df.fillna(0, inplace=True)
    return merged_df

def sarima_forecast(series, steps=4):
    if len(series) < 4:
        return [np.nan] * steps
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if (series <= 0).all():
        return [0.0] * steps
    try:
        log_series = np.log1p(series)
        model = SARIMAX(log_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 4),
                        enforce_stationarity=True, enforce_invertibility=True)
        model_fit = model.fit(disp=False)
        forecast_log = model_fit.forecast(steps=steps)
        forecast = np.expm1(forecast_log)
        forecast = np.where(np.isfinite(forecast), forecast, 0.0)
        forecast[forecast < 0] = 0
        return forecast.tolist()
    except Exception as e:
        return [np.nan] * steps

def forecast_all(df):
    metrics = ['total_fte', 'avg_fte_rate', 'fte_cost', 'total_spend', 'invoice_count']
    forecast_rows = []
    for project in df['PROJECT_NAME'].unique():
        group = df[df['PROJECT_NAME'] == project].sort_values(['Year', 'Quarter'])
        group['Period'] = group['Year'].astype(str) + 'Q' + group['Quarter'].astype(str)
        for metric in metrics:
            ts = group.set_index('Period')[metric].dropna()
            forecast = sarima_forecast(ts, 4)
            for i, val in enumerate(forecast):
                year = 2025
                quarter = i + 1
                forecast_rows.append({'PROJECT_NAME': project, 'Year': year, 'Quarter': quarter, metric: val})
    forecast_df = pd.DataFrame(forecast_rows)
    return forecast_df.pivot_table(index=['PROJECT_NAME', 'Year', 'Quarter'], values=metrics, aggfunc='sum', fill_value=0).reset_index()

# --- Streamlit UI ---
st.set_page_config(page_title="Project Financial Insights", layout="wide")
st.title("📊 Project Financial and FTE Insight Generator")

data = load_data()
forecast_df = forecast_all(data)
forecast_df['is_forecast'] = True
data['is_forecast'] = False
combined_df = pd.concat([data, forecast_df], ignore_index=True).fillna(0)
combined_df = combined_df.groupby(['PROJECT_NAME', 'Year', 'Quarter', 'is_forecast'], as_index=False).sum()
combined_df = combined_df.sort_values(['PROJECT_NAME', 'Year', 'Quarter'])

combined_df['spend_change_pct'] = combined_df.groupby('PROJECT_NAME')['total_spend'].pct_change().fillna(0)
combined_df['fte_change_pct'] = combined_df.groupby('PROJECT_NAME')['total_fte'].pct_change().fillna(0)
combined_df['risk_flag'] = combined_df.apply(lambda row: (
    "Overspending risk" if row['spend_change_pct'] > 0.10 else
    "FTE Overutilization" if row['fte_change_pct'] > 0.15 else
    "FTE Underutilization" if row['fte_change_pct'] < -0.15 else
    "Data anomaly" if row['total_spend'] == 0 or row['total_fte'] == 0 else
    "Normal"
), axis=1)

project_names = combined_df['PROJECT_NAME'].dropna().unique()
selected_project = st.selectbox("Select Project", sorted(project_names))
selected_year = st.selectbox("Select Year to Compare with 2024", ["2025"], index=0)

if st.button("Generate Summary"):
    with st.spinner("Generating financial insights..."):
        filtered = combined_df[
            (combined_df['PROJECT_NAME'] == selected_project) & 
            (combined_df['Year'].isin([2024, 2025]))
        ]
        csv_str = filtered.to_csv(index=False)
        prompt = ChatPromptTemplate.from_template("""
You are a senior financial analyst. Below is quarterly financial and FTE data for the selected project in 2024 and 2025.
This data includes forecasted values for 2025 where actuals are not yet available.

Compare:
- Q1 2024 vs Q1 2025
- Q2 2024 vs Q2 2025
- ... and so on up to Q4

If actual data for a 2025 quarter does not exist, use forecasted values (they are marked accordingly).
After comparing quarter wise provide detailed insights on:
- Budget risks or overspending
- FTE over- or under-utilization
- Anomalies and red flags
- Quarter-over-quarter trends to support business decisions

IMPORTANT: Mention whether 2025 data is actual or forecasted in each insight. Use numbers to validate and compare.

{csv_str}
""")
        chain = LLMChain(prompt=prompt, llm=llm)
        summary = chain.run(csv_str=csv_str)
        st.markdown("### 📌 Business Insights")
        st.markdown(summary)

        # Optional Visualization
        # st.line_chart(filtered.pivot(index=['Year','Quarter'], columns='is_forecast', values='total_spend'))
