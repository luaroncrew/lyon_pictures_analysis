import streamlit as st
from data_exploration_tab import data_exploration
from linear_regression_tab import linear_regression


(
    data_exploration_tab,
    linear_regression_tab,
) = st.tabs([
    "Data Exploration",
    "Linear Regression",
])

data_exploration(data_exploration_tab)
linear_regression(linear_regression_tab)
