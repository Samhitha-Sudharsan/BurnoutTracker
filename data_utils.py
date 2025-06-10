# data_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns # Add this import for better visualizations

def load_data(path="cleaned_data.csv"):
    # Ensure 'date' column is parsed as datetime objects upon loading
    # and that 'menstrual_phase' and 'cycle_day_of_phase' are loaded correctly
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date']) # Ensure date is datetime for plotting
    return df

def plot_mood_trend(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    df.sort_values('date', inplace=True)
    ax.plot(df['date'], df['mood'], marker='o', linestyle='-', color='skyblue', label='Mood')
    ax.set_title("Mood Over Time", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mood Rating", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

def plot_burnout_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    # Order burnout levels if they are numeric or map them for consistent order
    burnout_order = {0: 'Not at all', 1: 'Mild tiredness', 2: 'Drained', 3: 'Full-on burnout'}
    # Ensure 'burnout' is treated correctly as categorical or numeric if it's already mapped
    # If burnout is numeric (0-3), convert to string labels for better bar plot labels
    if pd.api.types.is_numeric_dtype(df['burnout']):
        # Map back to string labels for plotting if desired, or plot as numeric
        # For simplicity, we'll use value_counts on the numeric values
        burnout_counts = df['burnout'].value_counts().sort_index()
        burnout_counts.index = burnout_counts.index.map(burnout_order) # Map numeric to descriptive labels
        burnout_counts.plot(kind='bar', ax=ax, color='lightcoral')
    else: # If 'burnout' is still categorical strings like 'not at all', 'drained' etc.
        df['burnout'].value_counts().plot(kind='bar', ax=ax, color='lightcoral')

    ax.set_title("Burnout Frequency", fontsize=16)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Burnout Level", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# --- New Functions for Menstrual Phase Analysis ---

def plot_average_by_phase(df, column_name, title):
    """
    Plots the average of a given numerical column by menstrual phase.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure 'menstrual_phase' is treated as a category with a defined order for plotting
    phase_order = ["Menstrual (Period)", "Follicular", "Ovulation", "Luteal", "Luteal (PMS likely)", "N/A", "Unknown/Irregular Cycle", "Not Started"]
    # Filter out "N/A" and "Not Started" if they contain no meaningful data for averages
    plot_df = df[df['menstrual_phase'].isin(["Menstrual (Period)", "Follicular", "Ovulation", "Luteal", "Luteal (PMS likely)"])]
    
    if not plot_df.empty:
        # Group by phase and calculate the mean, then sort according to phase_order
        avg_by_phase = plot_df.groupby('menstrual_phase')[column_name].mean().reindex(phase_order)
        # Drop NaN values that resulted from reindexing phases that might not exist in the data
        avg_by_phase = avg_by_phase.dropna() 
        
        if not avg_by_phase.empty:
            sns.barplot(x=avg_by_phase.index, y=avg_by_phase.values, ax=ax, palette='viridis')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Menstrual Phase", fontsize=12)
            ax.set_ylabel(f"Average {column_name.replace('_', ' ').title()}", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write(f"No data available to plot {title} for recognized menstrual phases.")
    else:
        st.write(f"No data available to plot {title} for recognized menstrual phases.")


def plot_phase_distribution(df):
    """
    Plots the distribution (count) of entries across different menstrual phases.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure 'menstrual_phase' is treated as a category with a defined order
    phase_order = ["Menstrual (Period)", "Follicular", "Ovulation", "Luteal", "Luteal (PMS likely)", "N/A", "Unknown/Irregular Cycle", "Not Started"]
    
    # Use value_counts to get the counts for each phase, then reindex to sort
    phase_counts = df['menstrual_phase'].value_counts().reindex(phase_order, fill_value=0)
    # Filter out phases with 0 counts if they are not important to display
    phase_counts = phase_counts[phase_counts > 0]

    if not phase_counts.empty:
        sns.barplot(x=phase_counts.index, y=phase_counts.values, ax=ax, palette='mako')
        ax.set_title("Data Entries Distribution by Menstrual Phase", fontsize=16)
        ax.set_xlabel("Menstrual Phase", fontsize=12)
        ax.set_ylabel("Number of Entries", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No data available to plot menstrual phase distribution.")