"""
Streamlit tab for null value analysis and data quality visualization.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data_validator import DataValidator, load_and_validate
import logging

logger = logging.getLogger(__name__)


def null_analysis(tab):
    """Main function for null analysis tab."""
    
    with tab:
        st.header("Null Value Analysis & Data Quality")
        st.markdown("Analyze missing data patterns and data quality issues in your dataset.")
        
        try:
            df = pd.read_csv("initial_data.csv")
            validator = DataValidator(df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns):,}")
            with col3:
                patterns = validator.get_null_patterns()
                st.metric("Data Completeness", f"{patterns['completeness_rate']:.1f}%")
            
            st.subheader("Null Value Overview")
            
            null_analysis = validator.analyze_nulls()
            
            null_data = []
            for col, stats in null_analysis.items():
                if stats['has_nulls']:
                    null_data.append({
                        'Column': col,
                        'Null Count': stats['null_count'],
                        'Null %': stats['null_percentage'],
                        'Data Type': stats['data_type']
                    })
            
            if null_data:
                null_df = pd.DataFrame(null_data).sort_values('Null %', ascending=False)
                
                fig_bar = px.bar(
                    null_df, 
                    x='Column', 
                    y='Null %',
                    title='Null Percentage by Column',
                    labels={'Null %': 'Null Percentage (%)'},
                    color='Null %',
                    color_continuous_scale='Reds',
                    text='Null %'
                )
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_bar.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                st.subheader("Detailed Null Statistics")
                st.dataframe(
                    null_df.style.background_gradient(subset=['Null %'], cmap='Reds'),
                    use_container_width=True
                )
            else:
                st.success("No null values found in the dataset!")
            
            st.subheader("Null Patterns Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Row-wise Analysis")
                st.info(f"Rows with any null: {patterns['rows_with_any_null']:,}")
                st.info(f"Complete rows: {patterns['complete_rows']:,}")
                st.info(f"Max nulls in a single row: {patterns['max_nulls_in_row']}")
                st.info(f"Average nulls per row: {patterns['avg_nulls_per_row']:.2f}")
            
            with col2:
                row_completeness = pd.DataFrame({
                    'Status': ['Complete', 'Has Nulls'],
                    'Count': [patterns['complete_rows'], patterns['rows_with_any_null']]
                })
                
                fig_pie = px.pie(
                    row_completeness, 
                    values='Count', 
                    names='Status',
                    title='Row Completeness Distribution',
                    color_discrete_map={'Complete': '#2ecc71', 'Has Nulls': '#e74c3c'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            if patterns['columns_always_together_null']:
                st.subheader("Correlated Null Patterns")
                st.markdown("These columns tend to have null values together:")
                
                corr_data = []
                for col1, col2, corr in patterns['columns_always_together_null']:
                    corr_data.append({
                        'Column 1': col1,
                        'Column 2': col2,
                        'Correlation': corr
                    })
                
                if corr_data:
                    corr_df = pd.DataFrame(corr_data)
                    st.dataframe(corr_df, use_container_width=True)
            
            st.subheader("Handling Strategies")
            st.markdown("Recommended strategies for handling null values:")
            
            strategies = validator.suggest_handling_strategy()
            
            strategy_groups = {}
            for col, strategy in strategies.items():
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(col)
            
            for strategy, columns in strategy_groups.items():
                if "No action needed" not in strategy:
                    with st.expander(f"{strategy} ({len(columns)} columns)"):
                        for col in columns:
                            stats = null_analysis[col]
                            st.write(f"**{col}**")
                            st.write(f"- Null count: {stats['null_count']:,} ({stats['null_percentage']:.1f}%)")
                            st.write(f"- Data type: {stats['data_type']}")
                            st.write("---")
            
            st.subheader("Data Quality Report")
            
            with st.expander("View Full Report"):
                report = validator.get_summary_report()
                st.text(report)
            
            if st.button("Export Report to CSV"):
                export_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Null Count': stats['null_count'],
                        'Null Percentage': stats['null_percentage'],
                        'Data Type': stats['data_type'],
                        'Unique Values': stats['unique_values'],
                        'Suggested Strategy': strategies.get(col, 'N/A')
                    }
                    for col, stats in null_analysis.items()
                ])
                
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="Download Null Analysis Report",
                    data=csv,
                    file_name="null_analysis_report.csv",
                    mime="text/csv"
                )
                
        except FileNotFoundError:
            st.error("initial_data.csv not found. Please ensure the data file exists.")
            st.info("Run the pre-processing step first if you haven't already.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in null analysis tab: {str(e)}")


def create_null_heatmap(df: pd.DataFrame):
    """Create a heatmap visualization of null values."""
    null_mask = df.isnull().astype(int)
    
    if null_mask.sum().sum() == 0:
        return None
    
    sample_size = min(100, len(null_mask))
    null_sample = null_mask.sample(n=sample_size, random_state=42)
    
    fig = go.Figure(data=go.Heatmap(
        z=null_sample.T,
        x=null_sample.index,
        y=null_sample.columns,
        colorscale=[[0, 'green'], [1, 'red']],
        showscale=False,
        hovertemplate='Row: %{x}<br>Column: %{y}<br>Is Null: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Null Value Heatmap (Sample of {sample_size} rows)',
        xaxis_title='Row Index',
        yaxis_title='Column',
        height=400
    )
    
    return fig