import streamlit as st
import pandas as pd
from data_exploration_tab import data_exploration

initial_data = pd.read_csv("initial_data.csv")

(
    data_exploration_tab,
) = st.tabs([
    "Data Exploration",
])

# make a data exploration tab with the ability to observe the map with filters

# data transformations:
# add new column with the data in a normal format


data_exploration(data_exploration_tab)