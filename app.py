import streamlit as st
from data_exploration_tab import data_exploration


(
    data_exploration_tab,
) = st.tabs([
    "Data Exploration",
])

data_exploration(data_exploration_tab)
