# AI-Powered Badminton Performance Analyzer 🏸


This project is an interactive web application as part of assignment built with Streamlit that allows users to analyze a badminton player performance dataset. Users can select from a list of frequently asked questions, and the app uses Google's Gemini API to understand the request and generate a relevant data visualization on the fly.

## Demo


*The application features a clean sidebar with FAQ-style buttons for generating charts.*

---

## Features

-   **AI-Powered Intent Recognition**: Utilizes the Google Gemini API to interpret user requests and map them to the correct visualization.
-   **Interactive FAQ Interface**: A simple, user-friendly sidebar with clickable questions, making data analysis intuitive and accessible.
-   **Dynamic Chart Generation**: Creates a wide variety of charts—including bar charts, pie charts, heatmaps, and radar charts—using Matplotlib and Seaborn.
-   **Secure by Design**: Keeps the dataset and API keys private. Data is loaded from Streamlit's secure secrets manager for deployment, and API keys are handled via environment variables.
-   **Data-Driven Insights**: Provides analysis on player performance, skill correlations, location-based averages, and more.

---

## Technology Stack

-   **Backend**: Python
-   **Web Framework**: Streamlit
-   **AI Model**: Google Gemini API (`gemini-1.5-flash-latest`)
-   **Data Libraries**: pandas, NumPy
-   **Plotting**: Matplotlib, Seaborn

---