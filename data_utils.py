import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os # <--- ADD THIS LINE

# This function might already exist, ensure it's here
def load_data(file_path):
    """Loads data from a CSV file or creates an empty DataFrame if not found."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure 'date' column is datetime type upon loading
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    return pd.DataFrame()


## --- Corrected Plotting Functions ---

def plot_mood_trend(df, column_name, title):
    """
    Generates a Plotly line chart for a given metric over time.

    Args:
        df (pd.DataFrame): DataFrame containing the data, with a 'date' column.
        column_name (str): The name of the column to plot (e.g., 'mood', 'sleep', 'anxiety', 'energy').
        title (str): The title for the plot.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if df.empty or column_name not in df.columns or 'date' not in df.columns:
        # Return an empty figure or a message if data is not suitable
        return px.scatter(title=f"No data available for {title}")

    # Ensure 'date' column is datetime type for proper plotting
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_sorted = df.sort_values('date')

    fig = px.line(df_sorted, x='date', y=column_name, title=title)
    fig.update_layout(xaxis_title="Date", yaxis_title=column_name.replace('_', ' ').title())
    fig.update_traces(mode='lines+markers') # Show both lines and markers for clarity

    return fig


def plot_burnout_distribution(df):
    """
    Generates a Plotly bar chart for burnout level distribution.

    Args:
        df (pd.DataFrame): DataFrame containing the data, with a 'burnout' column.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if df.empty or 'burnout' not in df.columns or df['burnout'].isnull().all():
        return px.bar(title="No burnout data available for distribution.")

    # Map burnout numerical levels to descriptive labels for better visualization
    burnout_labels = {
        0: "0 - Not at all",
        1: "1 - Mild tiredness",
        2: "2 - Drained",
        3: "3 - Full-on burnout"
    }
    df['burnout_label'] = df['burnout'].map(burnout_labels)

    # Sort categories to ensure correct order on the plot
    category_order = [burnout_labels[i] for i in sorted(burnout_labels.keys())]

    fig = px.bar(
        df['burnout_label'].value_counts().reindex(category_order).reset_index(),
        x='burnout_label',
        y='count',
        title="Burnout Level Distribution",
        labels={'burnout_label': 'Burnout Level', 'count': 'Number of Entries'},
        color='burnout_label', # Color bars by level
        color_discrete_map={
            "0 - Not at all": "green",
            "1 - Mild tiredness": "lightgreen",
            "2 - Drained": "orange",
            "3 - Full-on burnout": "red"
        }
    )
    fig.update_layout(xaxis_title="Burnout Level", yaxis_title="Number of Entries")
    return fig


def plot_average_by_phase(df, column_name, title):
    """
    Generates a Plotly bar chart for average metric by menstrual cycle phase.

    Args:
        df (pd.DataFrame): DataFrame with 'cycle_phase' and the metric column.
        column_name (str): The name of the column to average (e.g., 'mood', 'energy').
        title (str): The title for the plot.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if df.empty or 'cycle_phase' not in df.columns or column_name not in df.columns:
        return px.bar(title=f"No data available for {title}")

    # Ensure 'cycle_phase' is treated as categorical for correct ordering
    # Define a specific order for cycle phases for consistent plotting
    phase_order = ['Period', 'Follicular', 'Ovulatory', 'Luteal', 'Unknown']
    df['cycle_phase'] = pd.Categorical(df['cycle_phase'], categories=phase_order, ordered=True)

    avg_df = df.groupby('cycle_phase')[column_name].mean().reset_index()
    avg_df = avg_df.sort_values(by='cycle_phase') # Sort by the defined categorical order

    fig = px.bar(avg_df, x='cycle_phase', y=column_name, title=title,
                 labels={'cycle_phase': 'Cycle Phase', column_name: f'Average {column_name.replace("_", " ").title()}'},
                 color='cycle_phase') # Color bars by phase

    fig.update_layout(xaxis_title="Cycle Phase", yaxis_title=f"Average {column_name.replace('_', ' ').title()}")
    return fig


def plot_phase_distribution(df):
    """
    Generates a Plotly pie chart for menstrual cycle phase distribution.

    Args:
        df (pd.DataFrame): DataFrame with 'cycle_phase' column.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object.
    """
    if df.empty or 'cycle_phase' not in df.columns or df['cycle_phase'].isnull().all():
        return px.pie(title="No cycle phase data available for distribution.")

    phase_counts = df['cycle_phase'].value_counts().reset_index()
    phase_counts.columns = ['cycle_phase', 'count']

    fig = px.pie(phase_counts, values='count', names='cycle_phase', title='Distribution of Logged Cycle Phases')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig