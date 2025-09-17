import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# --- Configuration and Helper Functions ---

def configure_gemini():
    """Loads the API key from .env and configures the Gemini API."""
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Display an error in the Streamlit app instead of exiting
        st.error("Error: GOOGLE_API_KEY not found. Please set it in your .env file.")
        st.stop()
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
        st.stop()

def get_chart_intent_from_gemini(question, df_columns):
    """Asks Gemini to classify the user's question into a chart type."""
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    You are a data analysis assistant. Based on the user's question about a dataset 
    with columns [{", ".join(df_columns)}], return exactly one of the following keywords:
    - 'grade_split': For questions about the proportion or split of grades.
    - 'avg_scores_by_location': For questions comparing average scores by location.
    - 'score_distribution': For questions about the distribution or frequency of scores.
    - 'unknown': If the question does not match any of the above.

    User Question: "{question}"
    Keyword:
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip().replace("`", "")
    except Exception as e:
        st.error(f"Error communicating with Gemini API: {e}")
        return "unknown"

def generate_chart(intent, df):
    """Generates a chart based on the intent and returns the matplotlib figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if intent == "grade_split":
        ax.set_title("Split of Students in A & B Grades")
        grade_counts = df['Grade'].value_counts()
        ax.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=90)
    
    elif intent == "avg_scores_by_location":
        ax.set_title("Average Overall Score by Location")
        avg_scores = df.groupby('Location')['Overall AVG'].mean().sort_values(ascending=False)
        sns.barplot(x=avg_scores.index, y=avg_scores.values, ax=ax)
        ax.set_xlabel("Location")
        ax.set_ylabel("Average Score")

    elif intent == "score_distribution":
        ax.set_title("Distribution of Overall Averages")
        sns.histplot(df['Overall AVG'], kde=True, bins=20, ax=ax)
        ax.set_xlabel("Overall Average")
        ax.set_ylabel("Number of Players")
    else:
        return None

    plt.tight_layout()
    return fig

# --- Streamlit Web App ---

# Configure the page
st.set_page_config(page_title="Badminton Performance Analyzer", layout="wide")
st.title("🏸 AI-Powered Badminton Performance Analyzer")

# Configure the Gemini API at the start
configure_gemini()

# Load data from Streamlit Secrets
try:
    # Access the secret string
    csv_string = st.secrets["csv_data"]
    # Use io.StringIO to treat the string as a file and specify the tab separator
    df = pd.read_csv(io.StringIO(csv_string), sep='\t')
    
   

except Exception as e:
    st.error(f"Fatal Error: Could not load data from Streamlit Secrets. Details: {e}")
    st.stop()

# User input
user_question = st.text_input(
    "Ask a question about the data:", 
    "e.g., Show me the grade distribution"
)

if st.button("Generate Chart"):
    if user_question:
        with st.spinner("Analyzing your question with Gemini..."):
            intent = get_chart_intent_from_gemini(user_question, df.columns)
        
        st.write(f"🔍 **Gemini Interpreted Intent:** `{intent}`")

        if intent != "unknown":
            with st.spinner("Creating your chart..."):
                chart_figure = generate_chart(intent, df)
                st.pyplot(chart_figure)
        else:
            st.warning("Sorry, I couldn't determine a specific chart for your request.")
            st.info("Please try asking about: grade splits, average scores by location, or score distribution.")
    else:
        st.warning("Please enter a question.")