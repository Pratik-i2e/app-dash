# import streamlit as st
# import pandas as pd
# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from langchain.prompts import ChatPromptTemplate
# from langchain.chains import LLMChain
# from langchain_community.chat_models import ChatOpenAI
# import datetime
# import os

# # --- Set Groq API credentials ---
# groq_api_key = st.secrets["SECRET_GROQ"]
# groq_api_base = "https://api.groq.com/openai/v1"
# model_name = "llama3-70b-8192"

# llm = ChatOpenAI(
#     model_name=model_name,
#     temperature=0.0,
#     openai_api_key=groq_api_key,
#     openai_api_base=groq_api_base
# )

# # --- Load Data ---
# @st.cache_data
# def load_data():
#     fte_df = pd.read_csv("AUTOMATION DATA(FTE Data) (1).csv")
#     finance_df = pd.read_csv("AUTOMATION DATA(Finance Data) (1).csv")
#     commitment_df = pd.read_csv("AUTOMATION DATA(Commitment) (1).csv")

#     fte_df['DATE'] = pd.to_datetime(fte_df['YEARMONTH'].astype(str) + '01', format='%Y%m%d')
#     finance_df['DATE'] = pd.to_datetime(finance_df['GL_INVOICE_MONTHYEAR'].astype(str) + '01', format='%Y%m%d')
#     fte_df['Year'] = fte_df['DATE'].dt.year
#     fte_df['Quarter'] = fte_df['DATE'].dt.quarter
#     finance_df['Year'] = finance_df['DATE'].dt.year
#     finance_df['Quarter'] = finance_df['DATE'].dt.quarter

#     fte_df['FTE_RATE'] = pd.to_numeric(fte_df['FTE_RATE'], errors='coerce').fillna(0)
#     fte_df['FTE'] = pd.to_numeric(fte_df['FTE'], errors='coerce').fillna(0)
#     fte_df['fte_cost'] = fte_df['FTE'] * fte_df['FTE_RATE']
#     finance_df['GL_TRANS_AMOUNT'] = pd.to_numeric(finance_df['GL_TRANS_AMOUNT'], errors='coerce').fillna(0)

#     fte_agg = fte_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg(
#         total_fte=('FTE', 'sum'),
#         avg_fte_rate=('FTE_RATE', 'mean'),
#         fte_cost=('fte_cost', 'sum')
#     ).reset_index()

#     finance_agg = finance_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg(
#         total_spend=('GL_TRANS_AMOUNT', 'sum'),
#         invoice_count=('GL_INVOICE_NUMBER', 'count')
#     ).reset_index()

#     commitment_agg = commitment_df.groupby(['PROJECT_NAME']).sum(numeric_only=True).reset_index()

#     merged_df = pd.merge(fte_agg, finance_agg, on=['PROJECT_NAME', 'Year', 'Quarter'], how='outer')
#     merged_df = pd.merge(merged_df, commitment_agg, on=['PROJECT_NAME'], how='outer')
#     merged_df.fillna(0, inplace=True)
#     return merged_df

# def sarima_forecast(series, steps=4):
#     if len(series) < 4:
#         return [np.nan] * steps
#     series = series.replace([np.inf, -np.inf], np.nan).dropna()
#     if (series <= 0).all():
#         return [0.0] * steps
#     try:
#         log_series = np.log1p(series)
#         model = SARIMAX(log_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 4),
#                         enforce_stationarity=True, enforce_invertibility=True)
#         model_fit = model.fit(disp=False)
#         forecast_log = model_fit.forecast(steps=steps)
#         forecast = np.expm1(forecast_log)
#         forecast = np.where(np.isfinite(forecast), forecast, 0.0)
#         forecast[forecast < 0] = 0
#         return forecast.tolist()
#     except Exception as e:
#         return [np.nan] * steps

# def forecast_all(df):
#     metrics = ['total_fte', 'avg_fte_rate', 'fte_cost', 'total_spend', 'invoice_count']
#     forecast_rows = []
#     for project in df['PROJECT_NAME'].unique():
#         group = df[df['PROJECT_NAME'] == project].sort_values(['Year', 'Quarter'])
#         group['Period'] = group['Year'].astype(str) + 'Q' + group['Quarter'].astype(str)
#         for metric in metrics:
#             ts = group.set_index('Period')[metric].dropna()
#             forecast = sarima_forecast(ts, 4)
#             for i, val in enumerate(forecast):
#                 year = 2025
#                 quarter = i + 1
#                 forecast_rows.append({'PROJECT_NAME': project, 'Year': year, 'Quarter': quarter, metric: val})
#     forecast_df = pd.DataFrame(forecast_rows)
#     return forecast_df.pivot_table(index=['PROJECT_NAME', 'Year', 'Quarter'], values=metrics, aggfunc='sum', fill_value=0).reset_index()

# # --- Streamlit UI ---
# st.set_page_config(page_title="Project Financial Insights", layout="wide")
# st.title("📊 Project Financial and FTE Insight Generator")

# data = load_data()
# forecast_df = forecast_all(data)
# forecast_df['is_forecast'] = True
# data['is_forecast'] = False
# combined_df = pd.concat([data, forecast_df], ignore_index=True).fillna(0)
# combined_df = combined_df.groupby(['PROJECT_NAME', 'Year', 'Quarter', 'is_forecast'], as_index=False).sum()
# combined_df = combined_df.sort_values(['PROJECT_NAME', 'Year', 'Quarter'])

# combined_df['spend_change_pct'] = combined_df.groupby('PROJECT_NAME')['total_spend'].pct_change().fillna(0)
# combined_df['fte_change_pct'] = combined_df.groupby('PROJECT_NAME')['total_fte'].pct_change().fillna(0)
# combined_df['risk_flag'] = combined_df.apply(lambda row: (
#     "Overspending risk" if row['spend_change_pct'] > 0.10 else
#     "FTE Overutilization" if row['fte_change_pct'] > 0.15 else
#     "FTE Underutilization" if row['fte_change_pct'] < -0.15 else
#     "Data anomaly" if row['total_spend'] == 0 or row['total_fte'] == 0 else
#     "Normal"
# ), axis=1)

# project_names = combined_df['PROJECT_NAME'].dropna().unique()
# selected_project = st.selectbox("Select Project", sorted(project_names))
# selected_year = st.selectbox("Select Year to Compare with 2024", ["2025"], index=0)

# if st.button("Generate Summary"):
#     with st.spinner("Generating financial insights..."):
#         filtered = combined_df[
#             (combined_df['PROJECT_NAME'] == selected_project) & 
#             (combined_df['Year'].isin([2024, 2025]))
#         ]
#         csv_str = filtered.to_csv(index=False)
#         prompt = ChatPromptTemplate.from_template("""
# You are a senior financial analyst. Below is quarterly financial and FTE data for the selected project in 2024 and 2025.
# This data includes forecasted values for 2025 where actuals are not yet available.

# Compare:
# - Q1 2024 vs Q1 2025
# - Q2 2024 vs Q2 2025
# - ... and so on up to Q4

# If actual data for a 2025 quarter does not exist, use forecasted values (they are marked accordingly).
# After comparing quarter wise provide detailed insights on:
# - Budget risks or overspending
# - FTE over- or under-utilization
# - Anomalies and red flags
# - Quarter-over-quarter trends to support business decisions

# IMPORTANT: Mention whether 2025 data is actual or forecasted in each insight. Use numbers to validate and compare.

# {csv_str}
# """)
#         chain = LLMChain(prompt=prompt, llm=llm)
#         summary = chain.run(csv_str=csv_str)
#         st.markdown("### 📌 Business Insights")
#         st.markdown(summary)

#         # Optional Visualization
#         # st.line_chart(filtered.pivot(index=['Year','Quarter'], columns='is_forecast', values='total_spend'))


import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import pandas as pd
import os

from crewai import Agent, Task, Crew

# Set Groq API Key from secrets
os.environ["GROQ_API_KEY"] = st.secrets["SECRET_GROQ"]

st.set_page_config(page_title="📊 Financial & FTE Analyzer", layout="wide")
st.title("📈 Project Financial Forecast & FTE Summary")

# Upload files
fte_file = 1

# Helper function for CrewAI summary
def generate_summary_with_crewai(csv_str, project_names):
    analyst_agent = Agent(
        role="Financial Analyst",
        goal="Compare quarterly financial and FTE data between 2024 and 2025",
        backstory="An experienced analyst specializing in financial and workforce trends",
        llm="groq/llama-3.3-70b-versatile"
    )

    insight_agent = Agent(
        role="Business Insight Generator",
        goal="Generate strategic, executive-level insights from financial and FTE comparisons",
        backstory="An executive summary expert providing decision support",
        llm="groq/llama-3.3-70b-versatile"
    )

    qa_agent = Agent(
        role="QA Reviewer",
        goal="Ensure the final insights are accurate, clear, and decision-useful",
        backstory="A sharp editor and analyst who ensures quality, clarity, and correctness in reports",
        llm="groq/llama-3.3-70b-versatile"
    )

    task1 = Task(
        description=f"""
You are analyzing a CSV of quarterly project data (filtered and cleaned below).

Only use the exact PROJECT_NAME values provided in the CSV — do not invent or generalize names like "Alpha", "Beta", or "Project A".

Compare:
- Q1 2024 vs Q1 2025
- Q2 2024 vs Q2 2025
- Q3 2024 vs Q3 2025
- Q4 2024 vs Q4 2025

For each quarter:
- Group results by real project names from the data
- Compare total_spend, total_fte, invoice_count, and fte_cost
- Mention whether 2025 values are forecasted or actual (via `is_forecast` column)
- Highlight any major percentage changes or zero values as potential anomalies

🔒 Use only these project names: {', '.join(project_names)}

Structure your report like:
Quarter: Q1 Comparison
- ProjectName1: analysis...
- ProjectName2: analysis...
...
""",
        agent=analyst_agent,
        expected_output="Clean structured comparison of all quarters and projects using real names and actual vs forecast status",
        input=csv_str
    )

    task2 = Task(
        description="""
Use the analyst's output to generate an executive summary:
- Highlight budget risks, overspending, anomalies, and FTE inefficiencies
- Clearly state whether values for 2025 are actual or forecasted
- Provide strategic recommendations to decision-makers
""",
        agent=insight_agent,
        expected_output="Polished executive summary including risks, anomalies, and strategic recommendations",
        input=task1.output
    )

    task3 = Task(
        description="""
Review the insights generated by the Insight Generator.
Ensure each quarter (Q1–Q4) is clearly separated.
Verify each project is discussed in its respective quarter.
Check that forecasted vs actual status is mentioned.
Refine language, improve structure, remove redundancy, and ensure clarity for executives.
""",
        agent=qa_agent,
        expected_output="QA-approved final executive summary",
        input=task2.output
    )

    crew = Crew(
        agents=[analyst_agent, insight_agent, qa_agent],
        tasks=[task1, task2, task3],
        verbose=True
    )

    return crew.kickoff()

# Proceed after all files are uploaded
if fte_file:
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
    # Combine key data from all sources (simplified example)
    fte_agg = fte_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg({
        'FTE Count': 'sum',
        'FTE Cost': 'sum'
    }).reset_index()

    finance_agg = finance_df.groupby(['PROJECT_NAME', 'Year', 'Quarter']).agg({
        'Spend Amount': 'sum',
        'Invoice Count': 'count'
    }).reset_index()

    combined_df = pd.merge(fte_agg, finance_agg, on=['PROJECT_NAME', 'Year', 'Quarter'], how='outer')
    combined_df.fillna(0, inplace=True)
    combined_df['is_forecast'] = combined_df['Year'].apply(lambda y: y == 2025)

    st.subheader("📂 Preview Combined Project Data")
    st.dataframe(combined_df)

    selected_project = st.selectbox("Select Project for Analysis", combined_df['PROJECT_NAME'].unique())

    if st.button("Generate Summary"):
        with st.spinner("Generating business insights with CrewAI..."):
            filtered = combined_df[
                (combined_df['PROJECT_NAME'] == selected_project) &
                (combined_df['Year'].isin([2024, 2025]))
            ]
            csv_str = filtered.to_csv(index=False)
            summary = generate_summary_with_crewai(csv_str, [selected_project])
            st.markdown("### 📌 Business Insights")
            st.markdown(summary)



