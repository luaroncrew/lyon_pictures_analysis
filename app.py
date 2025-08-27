import streamlit as st
from data_exploration_tab import data_exploration
from null_analysis_tab import null_analysis


(
    data_exploration_tab,
    null_analysis_tab,
) = st.tabs([
    "Data Exploration",
    "Null Analysis",
])

data_exploration(data_exploration_tab)
null_analysis(null_analysis_tab)
