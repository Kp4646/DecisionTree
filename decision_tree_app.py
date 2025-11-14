import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --- Configuration and Helpers ---

# Set Streamlit page configuration
st.set_page_config(
    page_title="Interactive Decision Tree",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global constant for sample size (200 for visual clarity)
N_SAMPLES = 200

# 1. Initialize session state for controls and data
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
# 'run_viz' controls the visibility of the detailed TREE STRUCTURE
if 'run_viz' not in st.session_state:
    st.session_state.run_viz = False
if 'split_feature' not in st.session_state:
    st.session_state.split_feature = 'Feature 1'
if 'scan_value' not in st.session_state:
    st.session_state.scan_value = 0.0


def set_run_viz():
    """Sets the state to run the detailed visualization (Tree Structure)."""
    st.session_state.run_viz = True


def change_seed():
    """Updates the random seed to generate new data and resets tree visualization."""
    st.session_state.random_seed = np.random.randint(10000)
    st.session_state.run_viz = False  # Reset tree view on new data generation


@st.cache_data
def generate_data(dataset_name, noise, n_samples, seed):
    """Generates synthetic data based on user selection."""
    if dataset_name == 'Moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    elif dataset_name == 'Circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=seed)
    elif dataset_name == 'Blobs':
        X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=seed)

    # Scale X coordinates for consistent visualization space
    X_min, X_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    return X, y, X_min, X_max, y_min, y_max


# --- Core ML/Visualization Logic ---

@st.cache_resource
def get_full_model(X, y, max_depth):
    """Trains the Decision Tree Classifier and returns the model (cached)."""
    try:
        # Train the full model based on user max_depth
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X, y)
        return model
    except ValueError as e:
        st.error(f"Error training full model: {e}")
        st.warning("Try reducing the Max Depth or clicking 'Generate New Data'.")
        return None


def plot_decision_boundary(model, X, y, X_min, X_max, y_min, y_max, max_depth, scan_feature, scan_value, h=0.02):
    """
    Plots the decision boundary using Plotly, including the interactive scan line.
    This now uses the full model defined by the sidebar's max_depth.
    """
    if model is None:
        return go.Figure()

    # Create meshgrid for prediction space
    xx, yy = np.meshgrid(np.arange(X_min, X_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on the meshgrid (Decision Boundary)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create Plotly Figure
    fig = go.Figure()

    # Add Decision Boundary (Contour/Heatmap)
    fig.add_trace(go.Contour(
        x=np.unique(xx),
        y=np.unique(yy),
        z=Z,
        showscale=False,
        colorscale=['rgba(255, 0, 0, 0.2)', 'rgba(0, 0, 255, 0.2)'],  # Light red and blue
        opacity=0.5,
        name='Decision Boundary',
        hoverinfo='none',
        line_width=0  # Hide contour lines
    ))

    # Add Data Points (Scatter Plot)
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=y,
            colorscale=[[0, 'red'], [1, 'blue']],
            opacity=0.8,
            line=dict(width=1, color='DarkSlateGrey')
        ),
        name='Data Points'
    ))

    # --- Add Interactive Scan Line ---
    scan_shape = {}
    if scan_feature == 'Feature 1':
        # Vertical line for Feature 1 (X-axis) split
        scan_shape = dict(
            type="line",
            x0=scan_value, x1=scan_value,
            y0=y_min, y1=y_max,
            line=dict(color="Red", width=3, dash="dashdot")
        )
    elif scan_feature == 'Feature 2':
        # Horizontal line for Feature 2 (Y-axis) split
        scan_shape = dict(
            type="line",
            x0=X_min, x1=X_max,
            y0=scan_value, y1=scan_value,
            line=dict(color="Red", width=3, dash="dashdot")
        )

    # Update layout for aesthetics
    fig.update_layout(
        title=f"Decision Boundary (Max Depth={max_depth})",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        template="plotly_dark",  # Nice dark theme
        margin=dict(l=20, r=20, t=50, b=20),
        height=600,
        showlegend=False,
        shapes=[scan_shape] if scan_shape else []
    )

    return fig


def plot_tree_structure(model):
    """
    Plots the full decision tree structure.
    """
    if model is None:
        return None

    plt.style.use('dark_background')  # Match the dark theme of the Plotly chart

    # Set figsize dynamically based on the full depth to prevent node overlap
    full_depth = model.get_depth()
    figsize_x = max(18, full_depth * 3.5)
    figsize_y = max(8, full_depth * 2)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # Use only core, robust arguments for maximum Scikit-learn compatibility
    plot_tree(
        model,
        feature_names=['Feature 1', 'Feature 2'],
        class_names=['Class 0', 'Class 1'],
        filled=True,
        rounded=True,
        ax=ax,
        fontsize=10,
    )
    ax.set_title("Decision Tree Structure", fontsize=16)

    # Adjust layout to prevent overlapping text
    plt.tight_layout()
    return fig


# --- Streamlit App Layout ---

def main():
    """The main Streamlit application function."""

    st.title("🌲 Interactive Decision Tree Visualizer")

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Decision Tree Configuration")

        # Dataset Selection
        dataset_name = st.selectbox(
            "Dataset",
            ('Moons', 'Circles', 'Blobs'),
            index=0,
            key='dataset',
            help="Choose a synthetic dataset with different separability."
        )

        # Noise Control
        noise_value = st.slider(
            "Noise (Complexity)",
            min_value=0.0,
            max_value=0.50,
            value=0.1,
            step=0.01,
            format="%.2f",
            key='noise',
            help="Higher noise makes the data more scattered and harder to separate."
        )

        # Max Depth Control (Defines the model's potential maximum depth)
        max_depth = st.slider(
            "Max Tree Depth (Model Limit)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key='max_depth',
            help="Controls the maximum depth the tree is allowed to reach."
        )

        st.markdown("---")

        # Button to show/hide the full tree structure
        st.button("🚀 Show Tree Structure", on_click=set_run_viz, type="primary")

        # GENERATE NEW DATA Button - Triggers script rerun and resets tree view
        st.button("🔄 Generate New Data", on_click=change_seed)

        st.markdown(f"<p style='font-size: small; color: gray;'>Current Seed: {st.session_state.random_seed}</p>",
                    unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            f"""
            **System Info**
            - **Data Points:** {N_SAMPLES}
            """,
            unsafe_allow_html=True
        )

    # --- Main Visualization Logic (Runs on every change) ---

    current_seed = st.session_state.random_seed

    # 2. Generate Data
    with st.spinner(f"Generating {dataset_name} data..."):
        X, y, X_min, X_max, y_min, y_max = generate_data(
            dataset_name=st.session_state.dataset,
            noise=st.session_state.noise,
            n_samples=N_SAMPLES,
            seed=current_seed
        )

    # 3. Train Full Model
    with st.spinner(f"Training Decision Tree (Max Depth: {st.session_state.max_depth})..."):
        full_model = get_full_model(X, y, st.session_state.max_depth)

    if full_model:
        st.header("Results")

        # --- Interactive Scan Controls ---
        st.subheader("Interactive Split Scan")

        # Determine the range for the scan slider based on the data
        scan_range_x = (X_min, X_max)
        scan_range_y = (y_min, y_max)

        col_feat, col_scan = st.columns([1, 2])

        with col_feat:
            st.session_state.split_feature = st.radio(
                "Scan Feature",
                ('Feature 1', 'Feature 2'),
                horizontal=True,
                key='split_feature_radio',
                help="Select which feature's axis the scanning line moves along."
            )

        with col_scan:
            current_range = scan_range_x if st.session_state.split_feature == 'Feature 1' else scan_range_y

            # Ensure the scan value is within the new range if the feature changed
            if st.session_state.scan_value < current_range[0] or st.session_state.scan_value > current_range[1]:
                st.session_state.scan_value = (current_range[0] + current_range[1]) / 2

            st.session_state.scan_value = st.slider(
                "Split Position",
                min_value=current_range[0],
                max_value=current_range[1],
                value=st.session_state.scan_value,
                step=0.01,
                format="%.2f",
                key='scan_slider',
                help="Move the line to see how a potential split at this position would divide the data."
            )

        # 5. Plot Decision Boundary (Always runs using the full model)
        st.subheader("Decision Boundary & Scan Line")
        st.markdown("**(Boundary updates in real-time as you adjust parameters)**")

        fig_boundary = plot_decision_boundary(
            full_model, X, y, X_min, X_max, y_min, y_max, st.session_state.max_depth,
            st.session_state.split_feature, st.session_state.scan_value
        )
        st.plotly_chart(fig_boundary, use_container_width=True)

        st.markdown("---")

        # 6. Display Tree Structure (Only runs on button click)
        if st.session_state.run_viz:
            st.subheader("Decision Tree Structure")

            # Use columns to center a potentially large tree plot
            tree_col1, tree_col2, tree_col3 = st.columns([0.1, 0.8, 0.1])
            with tree_col2:
                fig_tree = plot_tree_structure(full_model)
                if fig_tree:
                    st.pyplot(fig_tree, use_container_width=True)
                else:
                    st.warning("Could not render tree structure.")
        else:
            st.info("Click **'Show Tree Structure'** above to visualize the flowchart representation of the model.")

    else:
        st.error("Model training failed. Please check parameters.")


# Run the app
if __name__ == '__main__':
    main()