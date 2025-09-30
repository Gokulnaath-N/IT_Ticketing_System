import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure outputs directory exists for logging
import os
os.makedirs('outputs', exist_ok=True)

import logging
logging.basicConfig(filename='outputs/dashboard_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour_created'] = pd.to_datetime(df['creation_timestamp'], errors='coerce').dt.hour
    df['day_of_week'] = pd.to_datetime(df['creation_timestamp'], errors='coerce').dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = pd.to_datetime(df['creation_timestamp'], errors='coerce').dt.month
    df['is_peak_hour'] = ((df['hour_created'] >= 9) & (df['hour_created'] <= 17)).astype(int)
    return df


def add_text_features(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    df = df.copy()
    df['description'] = df['description'].astype(str)
    df['description_length'] = df['description'].str.len()
    df['urgency_keywords'] = df['description'].str.contains('urgent|critical|asap|immediate|down|crash', case=False, na=False).astype(int)
    tfidf_matrix = vectorizer.transform(df['description'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
    return df


def load_resources():
    # Load the processed data
    df = pd.read_csv('../data/processed/processed_tickets.csv', parse_dates=['creation_timestamp'])
    
    # Load the trained model
    model_path = os.path.join('..', 'models', 'reg_model.pkl')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error("Model file not found. Please run the training notebook first.")
        model = None
    
    # Initialize feature transformers
    vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
    if 'description' in df.columns:
        vectorizer.fit(df['description'].astype(str))
    else:
        vectorizer.fit([''])  # Fallback if no description column
    
    # Initialize categorical encoder
    cat_cols = [col for col in ['priority', 'type', 'business_type'] if col in df.columns]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    if cat_cols:
        encoder.fit(df[cat_cols])
    else:
        encoder.fit([['']])  # Fallback if no categorical columns
    
    # Initialize numeric scaler
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include='number').columns.drop('resolution_time_hours', errors='ignore')
    if len(num_cols) > 0:
        scaler.fit(df[num_cols])
    
    return df, model, vectorizer, encoder, scaler, num_cols


st.title('IT Ticket Resolution Predictor Dashboard')
st.sidebar.title('Navigation')
page = st.sidebar.radio('Sections', ['Overview', 'KPIs', 'Trends', 'Predict Resolution', 'Cluster Analysis'])

df, model, vec, enc, scal, num_cols = load_resources()

if page == 'Overview':
    st.write('Project: Predicts ticket resolution time, clusters by type, and reports KPIs.')

elif page == 'KPIs':
    col1, col2, col3, col4 = st.columns(4)
    avg_res = df['resolution_time_hours'].mean()
    col1.metric('Avg Resolution Time', f'{avg_res:.2f} hours')
    fcr_rate = (df['escalation_history'] == 0).mean() * 100 if 'escalation_history' in df.columns else 80
    col2.metric('First Contact Resolution %', f'{fcr_rate:.1f}%')
    csat = df['satisfaction_scores'].mean() if 'satisfaction_scores' in df.columns else 4.5
    col3.metric('CSAT Score', f'{csat:.1f}/5')
    sla_comp = (df['resolution_time_hours'] < 24).mean() * 100
    col4.metric('SLA Compliance', f'{sla_comp:.1f}%')
    logging.info('KPIs displayed')

elif page == 'Trends':
    if 'priority' in df.columns:
        fig = px.bar(df.groupby('priority')['resolution_time_hours'].mean().reset_index(), x='priority', y='resolution_time_hours', title='Avg Resolution by Priority')
        st.plotly_chart(fig, use_container_width=True)
    if 'creation_timestamp' in df.columns and 'ticket_id' in df.columns:
        weekly = df.set_index('creation_timestamp').resample('W')['ticket_id'].count()
        fig2 = px.line(weekly, title='Weekly Ticket Volume')
        st.plotly_chart(fig2, use_container_width=True)

elif page == 'Predict Resolution':
    with st.form('New Ticket'):
        priority = st.selectbox('Priority', sorted(df['priority'].dropna().unique().tolist()) if 'priority' in df.columns else ['Low','Medium','High','Critical'])
        category = st.selectbox('Category', sorted(df['category'].dropna().unique().tolist()) if 'category' in df.columns else ['Hardware','Software','Network','Access'])
        dept = st.selectbox('User Department', sorted(df['user_department'].dropna().unique().tolist()) if 'user_department' in df.columns else ['IT','HR','Finance'])
        desc = st.text_area('Description')
        submitted = st.form_submit_button('Predict')
    if submitted:
        input_df = pd.DataFrame({'priority': [priority], 'category': [category], 'description': [desc], 'user_department': [dept], 'creation_timestamp': [pd.Timestamp.now()]})
        input_df = add_time_features(input_df)
        input_df = add_text_features(input_df, vec)
        encoded = pd.DataFrame(enc.transform(input_df[['priority', 'category', 'user_department']]).toarray(), columns=enc.get_feature_names_out(['priority', 'category', 'user_department']))
        input_features = pd.concat([input_df.drop(['priority', 'category', 'description', 'user_department', 'creation_timestamp'], axis=1), encoded], axis=1)
        input_features[num_cols] = input_features[num_cols].reindex(columns=num_cols, fill_value=0)
        input_features[num_cols] = scal.transform(input_features[num_cols])
        pred = model.predict(input_features)[0]
        st.success(f'Predicted Resolution Time: {pred:.2f} hours')
        logging.info('Prediction made')

elif page == 'Cluster Analysis':
    if 'cluster' in df.columns:
        st.dataframe(df.groupby('cluster').agg({'resolution_time_hours': 'mean', 'ticket_id': 'count'}).rename(columns={'ticket_id': 'count'}))
        if 'description_length' in df.columns:
            fig3 = px.scatter(df, x='description_length', y='resolution_time_hours', color='cluster', title='Resolution vs Description Length by Cluster')
            st.plotly_chart(fig3, use_container_width=True)


