import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def linear_regression(tab):
    with tab:
        st.header("Linear Regression Example")
        
        # Controls for data generation
        st.subheader("Data Generation Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_samples = st.slider(
                "Number of Samples",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Number of data points to generate"
            )
        
        with col2:
            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=1.0,
                help="Standard deviation of noise added to the data"
            )
        
        with col3:
            random_seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
                help="Seed for reproducible random data generation"
            )
        
        # Generate synthetic data
        logger.info(f"Generating {n_samples} samples with noise level {noise_level}")
        np.random.seed(random_seed)
        
        # Create feature variable (X)
        X = np.random.uniform(-10, 10, n_samples)
        
        # Define true linear relationship: y = 2.5x + 5
        true_slope = 2.5
        true_intercept = 5
        
        # Generate target variable with noise
        y = true_slope * X + true_intercept + np.random.normal(0, noise_level, n_samples)
        
        # Create DataFrame for easier handling
        data = pd.DataFrame({
            'X': X,
            'y': y
        })
        
        # Display true relationship
        st.subheader("True Relationship")
        st.write(f"**True equation:** y = {true_slope}x + {true_intercept}")
        
        # Train-test split
        st.subheader("Model Training")
        
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        ) / 100
        
        X_train, X_test, y_train, y_test = train_test_split(
            X.reshape(-1, 1), y, test_size=test_size, random_state=random_seed
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        logger.info(f"Model trained - Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
        
        # Display model parameters
        st.subheader("Learned Model")
        st.write(f"**Learned equation:** y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
        
        # Display metrics
        st.subheader("Model Performance")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        metric_col1.metric("Train MSE", f"{train_mse:.2f}")
        metric_col2.metric("Test MSE", f"{test_mse:.2f}")
        metric_col3.metric("Train R²", f"{train_r2:.4f}")
        metric_col4.metric("Test R²", f"{test_r2:.4f}")
        
        # Create visualization
        st.subheader("Data Visualization")
        
        # Create scatter plot with regression line
        fig = go.Figure()
        
        # Add training data
        fig.add_trace(go.Scatter(
            x=X_train.flatten(),
            y=y_train,
            mode='markers',
            name='Training Data',
            marker=dict(color='blue', size=8, opacity=0.6)
        ))
        
        # Add test data
        fig.add_trace(go.Scatter(
            x=X_test.flatten(),
            y=y_test,
            mode='markers',
            name='Test Data',
            marker=dict(color='red', size=8, opacity=0.6)
        ))
        
        # Add regression line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = model.predict(X_line)
        
        fig.add_trace(go.Scatter(
            x=X_line.flatten(),
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='green', width=3)
        ))
        
        # Add true line for comparison
        y_true_line = true_slope * X_line.flatten() + true_intercept
        
        fig.add_trace(go.Scatter(
            x=X_line.flatten(),
            y=y_true_line,
            mode='lines',
            name='True Line',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Linear Regression: Data and Fitted Line",
            xaxis_title="X (Feature)",
            yaxis_title="y (Target)",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("Residuals Analysis")
        
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test
        
        fig_residuals = go.Figure()
        
        # Add training residuals
        fig_residuals.add_trace(go.Scatter(
            x=y_pred_train,
            y=residuals_train,
            mode='markers',
            name='Training Residuals',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add test residuals
        fig_residuals.add_trace(go.Scatter(
            x=y_pred_test,
            y=residuals_test,
            mode='markers',
            name='Test Residuals',
            marker=dict(color='red', size=6, opacity=0.6)
        ))
        
        # Add horizontal line at y=0
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="gray")
        
        fig_residuals.update_layout(
            title="Residuals vs Predicted Values",
            xaxis_title="Predicted Values",
            yaxis_title="Residuals",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Display sample of data
        st.subheader("Data Sample")
        st.write("First 10 data points:")
        display_data = pd.DataFrame({
            'X': X[:10],
            'y (actual)': y[:10],
            'y (predicted)': model.predict(X[:10].reshape(-1, 1))
        })
        st.dataframe(display_data, use_container_width=True)