import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import plotly.express as px
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename='logs/streamlit_app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set page config
st.set_page_config(
    page_title="IT Ticket Classification",
    page_icon="🎫",
    layout="wide"
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    try:
        model = xgb.XGBoost()
        model.load_model('models/ticket_classifier.json')
        vectorizer = joblib.load('data/processed/tfidf_vectorizer.joblib')
        label_encoder = joblib.load('models/label_encoder.joblib')
        return model, vectorizer, label_encoder
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        st.error("Error loading models. Please ensure all model files are present.")
        return None, None, None

# Preprocess input text
def preprocess_text(subject, description, vectorizer):
    text = f"{subject} {description}"
    return vectorizer.transform([text])

# Main function
def main():
    st.title("IT Ticket Classification System 🎫")
    st.markdown("""
    This system helps classify IT support tickets based on their content.
    Enter the ticket details below to get a predicted classification.
    """)

    # Load models
    model, vectorizer, label_encoder = load_models()
    
    if not all([model, vectorizer, label_encoder]):
        st.stop()

    # Create input form
    with st.form("ticket_form"):
        subject = st.text_input("Ticket Subject")
        description = st.text_area("Ticket Description")
        priority = st.selectbox(
            "Priority Level",
            ["Critical", "High", "Medium", "Low", "Very Low"]
        )
        ticket_type = st.selectbox(
            "Ticket Type",
            ["Incident", "Service Request", "Change Request", "Problem"]
        )

        submitted = st.form_submit_button("Classify Ticket")

    if submitted:
        try:
            # Process inputs
            priority_features = {
                'priority_critical': 1 if priority == "Critical" else 0,
                'priority_high': 1 if priority == "High" else 0,
                'priority_medium': 1 if priority == "Medium" else 0,
                'priority_low': 1 if priority == "Low" else 0,
                'priority_very_low': 1 if priority == "Very Low" else 0
            }

            # Get TF-IDF features
            tfidf_features = preprocess_text(subject, description, vectorizer)
            
            # Combine features
            features = np.hstack([
                np.array([list(priority_features.values())]),
                pd.get_dummies([ticket_type], columns=['type']).values,
                tfidf_features.toarray()
            ])

            # Make prediction
            prediction = model.predict(features)
            predicted_queue = label_encoder.inverse_transform(prediction)[0]

            # Display results
            st.success(f"Predicted Queue: {predicted_queue}")

            # Log the prediction
            logging.info(f"Prediction made - Subject: {subject[:50]}... Queue: {predicted_queue}")

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            st.error("An error occurred during classification. Please try again.")

    # Add visualizations
    if 'show_stats' not in st.session_state:
        st.session_state.show_stats = False

    if st.button("Show Statistics"):
        st.session_state.show_stats = not st.session_state.show_stats

    if st.session_state.show_stats:
        try:
            # Load historical data
            df = pd.read_csv('data/processed/processed_tickets.csv')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Queue distribution
                queue_counts = df['queue'].value_counts().head(10)
                fig1 = px.bar(
                    x=queue_counts.index,
                    y=queue_counts.values,
                    title="Top 10 Most Common Ticket Queues",
                    labels={'x': 'Queue', 'y': 'Number of Tickets'}
                )
                st.plotly_chart(fig1)

            with col2:
                # Priority distribution
                priority_counts = df['priority'].value_counts()
                fig2 = px.pie(
                    values=priority_counts.values,
                    names=priority_counts.index,
                    title="Ticket Priority Distribution"
                )
                st.plotly_chart(fig2)

        except Exception as e:
            logging.error(f"Error loading statistics: {str(e)}")
            st.error("Error loading statistics. Please try again later.")

if __name__ == "__main__":
    main()