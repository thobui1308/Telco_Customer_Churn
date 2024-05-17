import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.set_page_config(layout="wide")
st.title(":bar_chart: Dataset")
root_path = Path(__file__).parent.parent # pages < root
df = pd.read_excel(root_path / "data" / "data.xlsx", index_col=0)
df_profile = ProfileReport(df ,explorative=True)
st_profile_report(df_profile)

