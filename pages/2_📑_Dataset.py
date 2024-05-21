import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")
st.title(":bar_chart: Dataset")
root_path = Path(__file__).parent.parent # pages < root
df = pd.read_excel(root_path / "data" / "Data_Telco_Customer_Churn.xlsx", index_col=0)

tab1, tab2, tab3 = st.tabs(["Demo", "Service","Status"])
with tab1:
    Demo = pd.read_excel(df, sheet_name = 'Telco_Demo')
    st.dataframe(Demo.head(20))
with tab2:
    Services = pd.read_excel(df, sheet_name = 'Telco_Services')
    st.dataframe(Services.head(20))
with tab3:
    Status = pd.read_excel(df, sheet_name = 'Telco_Status')
    st.dataframe(Status.head(20))


