import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

initial_data = pd.read_csv("initial_data.csv")


def data_exploration(tab):
    st.dataframe(initial_data)

    # Limit data to head to prevent performance issues
    limited_data = initial_data.head(30000)

    # Extract coordinates for clustering
    map_data = pd.DataFrame()
    map_data['lat'] = limited_data[' lat']
    map_data['lon'] = limited_data[' long']

    with tab:
        # Add controls for DBSCAN parameters
        st.subheader("DBSCAN Clustering Parameters")
        col1, col2, col3 = st.columns(3)

        with col1:
            eps = st.slider("Epsilon (neighborhood size)",
                            min_value=0.001,
                            max_value=0.5,
                            value=0.3,
                            step=0.001,
                            help="Maximum distance between two points to be considered neighbors")

        with col2:
            min_samples = st.slider("Minimum Samples",
                                    min_value=2,
                                    max_value=300,
                                    value=5,
                                    step=1,
                                    help="Minimum number of points to form a dense region")

        with col3:
            algorithm = st.selectbox(
                "Algorithm",
                options=['auto', 'ball_tree', 'kd_tree', 'brute'],
                index=0,
                help="Algorithm used to compute the nearest neighbors"
            )

        # Apply DBSCAN clustering
        coords = map_data[['lat', 'lon']].values
        coords_scaled = StandardScaler().fit_transform(coords)

        # Apply DBSCAN with user-selected parameters
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm=algorithm)
        map_data['cluster'] = dbscan.fit_predict(coords_scaled)

        # Count clusters and noise points
        n_clusters = len(set(map_data['cluster'])) - (1 if -1 in map_data['cluster'] else 0)
        n_noise = list(map_data['cluster']).count(-1)

        # Create metrics to show clustering results
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Number of Clusters", n_clusters)
        metric_col2.metric("Noise Points", n_noise)

        # Create color mapping for clusters
        colors = px.colors.qualitative.Plotly
        color_map = {-1: "#808080"}  # Gray for noise points
        for i in range(n_clusters):
            color_map[i] = colors[i % len(colors)]

        # Convert cluster numbers to colors
        map_data['color'] = map_data['cluster'].map(color_map)

        # Display map with cluster colors
        st.subheader("Geographical Clustering")
        st.map(
            data=map_data,
            use_container_width=True,
            color='color',
            size=2
        )

        # Optional: Show breakdown of clusters in a bar chart
        if n_clusters > 0:
            st.subheader("Cluster Distribution")
            cluster_counts = map_data['cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Count']

            # Sort by cluster number, keeping -1 (noise) at the end
            cluster_counts = cluster_counts.sort_values('Cluster',
                                                        key=lambda x: [i == -1 for i in x],
                                                        kind='stable')

            fig = px.bar(cluster_counts, x='Cluster', y='Count',
                         title='Points per Cluster (-1 = noise)',
                         color='Cluster',
                         color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)