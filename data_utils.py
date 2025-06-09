# data_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def load_data(path="cleaned_data.csv"):
    return pd.read_csv(path)

def plot_mood_trend(df):
    fig, ax = plt.subplots()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    ax.plot(df['date'], df['mood'], marker='o', label='Mood')
    ax.set_title("Mood Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mood")
    st.pyplot(fig)

def plot_burnout_distribution(df):
    fig, ax = plt.subplots()
    df['burnout'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Burnout Frequency")
    ax.set_ylabel("Count")
    st.pyplot(fig)
