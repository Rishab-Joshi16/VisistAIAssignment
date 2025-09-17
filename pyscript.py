import os
import io
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st

# --- NEW: Helper function to add labels to bar charts ---
def add_labels_to_bars(ax):
    """Adds data labels to the bars in a matplotlib Axes object."""
    for p in ax.patches:
        width = p.get_width()
        height = p.get_height()
        x, y = p.get_xy()
        
        # Determine the position for the label
        # For horizontal bars
        if width > height: 
            ax.annotate(f'{width:.2f}', (x + width + 0.02 * width, y + height / 2), ha='left', va='center')
        # For vertical bars
        else:
            ax.annotate(f'{height:.0f}', (x + width / 2, y + height + 0.02 * height), ha='center', va='bottom')


# --- Configuration and Helper Functions ---

def configure_gemini():
    """Loads the API key from .env and configures the Gemini API."""
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
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
    - 'top_10_players': For questions about the best performing or top 10 players.
    - 'skill_heatmap': For questions about the correlation or relationship between skills.
    - 'radar_profile_by_grade': For questions comparing the average skills of different grades.
    - 'player_quadrant': For questions that ask to categorize or group players by attack skills (FA and BA).
    - 'skill_distribution_boxplot': For questions about comparing the distribution of each skill.
    - 'skill_vs_average_scatter': For questions about the relationship between a specific skill (like Forehand Attack) and the overall average.
    - 'player_count_by_location': For questions about the number of players in each location.
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

def generate_chart(intent, df, df_unpivoted):
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
        add_labels_to_bars(ax) # --- NEW ---

    elif intent == "score_distribution":
        ax.set_title("Distribution of Overall Averages")
        sns.histplot(df['Overall AVG'], kde=True, bins=20, ax=ax)
        ax.set_xlabel("Overall Average")
        ax.set_ylabel("Number of Players")

    elif intent == "top_10_players":
        ax.set_title("Top 10 Players by Overall Average")
        top_10 = df.nlargest(10, 'Overall AVG').sort_values('Overall AVG', ascending=True)
        ax.barh(top_10['Student'], top_10['Overall AVG'])
        ax.set_xlabel("Overall Average Score")
        ax.set_ylabel("Student")
        add_labels_to_bars(ax) # --- NEW ---

    elif intent == "skill_heatmap":
        ax.set_title("Correlation Heatmap of Skills")
        skill_cols = ['FG', 'BG', 'FA', 'BA', 'BS', 'Overall AVG']
        corr = df[skill_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

    elif intent == "radar_profile_by_grade":
        plt.close(fig) 
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        ax.set_title("Average Skill Profile: Grade A vs. Grade B")
        skills = ['FG', 'BG', 'FA', 'BA', 'BS']
        grade_a_means = df[df['Grade'] == 'A'][skills].mean().values
        grade_b_means = df[df['Grade'] == 'B'][skills].mean().values
        angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
        angles += angles[:1]
        values_a = np.concatenate((grade_a_means, [grade_a_means[0]]))
        plot_a, = ax.plot(angles, values_a, 'o-', label='Grade A')
        ax.fill(angles, values_a, alpha=0.1)
        values_b = np.concatenate((grade_b_means, [grade_b_means[0]]))
        plot_b, = ax.plot(angles, values_b, 'o-', label='Grade B')
        ax.fill(angles, values_b, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(skills)
        ax.legend()
        # --- NEW: Add labels to radar points ---
        for i, (angle, value) in enumerate(zip(angles[:-1], grade_a_means)):
            ax.text(angle, value + 5, f'{value:.1f}', ha='center', va='center', color=plot_a.get_color())
        for i, (angle, value) in enumerate(zip(angles[:-1], grade_b_means)):
            ax.text(angle, value + 5, f'{value:.1f}', ha='center', va='center', color=plot_b.get_color())


    elif intent == "player_quadrant":
        ax.set_title("Player Performance Quadrant (Attack Skills)")
        sns.scatterplot(data=df, x='FA', y='BA', hue='Grade', ax=ax, alpha=0.7)
        avg_fa = df['FA'].mean()
        avg_ba = df['BA'].mean()
        ax.axvline(avg_fa, color='r', linestyle='--', lw=1)
        ax.axhline(avg_ba, color='r', linestyle='--', lw=1)
        ax.set_xlabel("Forehand Attack (FA) Score")
        ax.set_ylabel("Backhand Attack (BA) Score")
        ax.legend(title="Grade")
    
    elif intent == "skill_distribution_boxplot":
        ax.set_title("Distribution of Player Scores by Skill")
        sns.boxplot(data=df_unpivoted, x='Skill', y='Score', ax=ax)
        ax.set_xlabel("Skill")
        ax.set_ylabel("Score")

    elif intent == "skill_vs_average_scatter":
        ax.set_title("Forehand Attack Score vs. Overall Average")
        sns.regplot(data=df, x='FA', y='Overall AVG', ax=ax, line_kws={"color": "red"})
        ax.set_xlabel("Forehand Attack (FA) Score")
        ax.set_ylabel("Overall Average")

    elif intent == "player_count_by_location":
        ax.set_title("Number of Players by Location")
        sns.countplot(data=df, x='Location', order=df['Location'].value_counts().index, ax=ax)
        ax.set_xlabel("Location")
        ax.set_ylabel("Number of Players")
        add_labels_to_bars(ax) # --- NEW ---

    else:
        return None

    plt.tight_layout()
    return fig

# --- Streamlit Web App ---
st.set_page_config(page_title="Badminton Performance Analyzer", layout="wide")
st.title("🏸 AI-Powered Badminton Performance Analyzer")

configure_gemini()

try:
    csv_string = st.secrets["csv_data"]
    df = pd.read_csv(io.StringIO(csv_string), sep='\t')
    
    skill_cols = ['FG', 'BG', 'FA', 'BA', 'BS']
    df_unpivoted = df.melt(id_vars=['Student', 'Location', 'Grade', 'Overall AVG'], 
                           value_vars=skill_cols, 
                           var_name='Skill', 
                           value_name='Score')

except Exception as e:
    st.error(f"Fatal Error: Could not load data from Streamlit Secrets. Details: {e}")
    st.stop()

st.sidebar.header("Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question here:", 
    "Show me the top 10 players",
    height=100
)

st.sidebar.subheader("Example Questions:")
st.sidebar.markdown("""
- *How many players are in each location?*
- *Who are the top 10 players?*
- *Compare the skills of Grade A vs. Grade B players.*
- *Categorize players by their attack skills.*
- *Show me a heatmap of skill correlations.*
- *Compare the skill distributions.*
""")

if st.sidebar.button("Generate Chart"):
    if user_question:
        with st.spinner("Analyzing your question with Gemini..."):
            intent = get_chart_intent_from_gemini(user_question, df.columns)
        
        st.write(f"🔍 **Gemini Interpreted Intent:** `{intent}`")

        if intent != "unknown":
            with st.spinner("Creating your chart..."):
                chart_figure = generate_chart(intent, df, df_unpivoted)
                st.pyplot(chart_figure)
        else:
            st.warning("Sorry, I couldn't determine a specific chart for your request.")
    else:
        st.warning("Please enter a question.")

st.subheader("Raw Data")
st.dataframe(df)