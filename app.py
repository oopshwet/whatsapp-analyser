import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import emoji
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
import numpy as np
import google.generativeai as genai
import time

# -------------------------------------------------
# ✅ Streamlit & Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="💬 WhatsApp Chat Analyzer", layout="wide")
st.title("💬 WhatsApp Chat Analyzer — Full Dashboard")
st.write("Upload your exported WhatsApp chat text file (without media) to explore in-depth insights!")

# -------------------------------------------------
# ✅ Load Gemini API Key (Streamlit Cloud safe)
# -------------------------------------------------
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.sidebar.success("✅ Gemini API connected successfully")
except Exception as e:
    st.sidebar.error("❌ Gemini API key not found! Please add it in `.streamlit/secrets.toml`")
    st.stop()

# -------------------------------------------------
# 📦 NLTK Setup
# -------------------------------------------------
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------------------------------
# 📁 File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your WhatsApp chat file (.txt)", type=["txt"])

# -------------------------------------------------
# 🔧 Preprocessing
# -------------------------------------------------
def preprocess(data):
    pattern = r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\u202f?(?:AM|PM|am|pm)\s-\s'
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    df['message_date'] = df['message_date'].str.replace('\u202f', ' ', regex=False)
    df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %I:%M %p - ', errors='coerce')
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users, messages = [], []
    for message in df['user_message']:
        entry = re.split(r'([^:]+):\s', message, maxsplit=1)
        if len(entry) >= 3:
            users.append(entry[1])
            messages.append(entry[2])
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    df['period'] = df['hour'].apply(lambda h: f'{h}-{(h+1)%24}' if not pd.isna(h) else None)
    return df

# -------------------------------------------------
# 🔍 Helper Functions
# -------------------------------------------------
def extract_emojis(s):
    return [c for c in s if c in emoji.EMOJI_DATA]

def most_common_words(messages):
    words = []
    stop_words = stopwords.words('english')
    for msg in messages:
        for word in str(msg).lower().split():
            if word.isalpha() and word not in stop_words:
                words.append(word)
    return Counter(words).most_common(15)

def estimate_response_time(df, user):
    user_msgs = df[df['user'] == user].sort_values('date')
    if len(user_msgs) < 2:
        return None
    diffs = user_msgs['date'].diff().dropna().dt.total_seconds() / 60
    return np.mean(diffs)

sia = SentimentIntensityAnalyzer()

# -------------------------------------------------
# 📊 Main App
# -------------------------------------------------
if uploaded_file is not None:
    data = uploaded_file.read().decode("utf-8")
    df = preprocess(data)
    st.success("✅ Chat processed successfully!")

    # Sidebar
    st.sidebar.header("Filters")
    users = sorted(df['user'].unique().tolist())
    selected_user = st.sidebar.selectbox("Select a user", ["Overall"] + users)

    if selected_user != "Overall":
        df = df[df['user'] == selected_user]

    st.subheader(f"📊 Analysis for: {selected_user}")

    # ---------- Basic Stats ----------
    total_messages = df.shape[0]
    total_words = sum(len(str(m).split()) for m in df['message'])
    media_msgs = df[df['message'].str.contains('<Media omitted>', na=False)].shape[0]
    links = df['message'].str.contains('http|https', case=False, na=False).sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", total_messages)
    col2.metric("Total Words", total_words)
    col3.metric("Media Shared", media_msgs)
    col4.metric("Links Shared", links)

    # ---------- Most Active Users ----------
    if selected_user == "Overall":
        st.subheader("👥 Most Active Users")
        top_users = df['user'].value_counts().head(10)
        fig, ax = plt.subplots()
        sns.barplot(x=top_users.values, y=top_users.index, ax=ax)
        st.pyplot(fig)

    # ---------- Daily Timeline ----------
    st.subheader("📅 Daily Timeline")
    daily_timeline = df.groupby('only_date').count()['message']
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(daily_timeline.index, daily_timeline.values)
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # ---------- Monthly Timeline ----------
    st.subheader("🗓️ Monthly Timeline")
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time_list = [f"{m}-{y}" for m, y in zip(timeline['month'], timeline['year'])]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(time_list, timeline['message'])
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # ---------- Activity Heatmap ----------
    st.subheader("🔥 Activity Heatmap (Day vs Hour)")
    activity_heatmap = df.pivot_table(index='day_name', columns='hour', values='message', aggfunc='count').fillna(0)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(activity_heatmap, cmap='mako', ax=ax)
    st.pyplot(fig)

    # ---------- Sentiment ----------
    st.subheader("🧠 Sentiment Analysis")
    df['sentiment'] = df['message'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
    sentiment_counts = df['sentiment_label'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig)

    # ---------- Word Cloud ----------
    st.subheader("☁️ Word Cloud")
    text = " ".join(df['message'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # ---------- Emoji ----------
    st.subheader("😂 Emoji Analysis")
    emojis = df['message'].apply(extract_emojis)
    emoji_list = [em for sublist in emojis for em in sublist]
    if emoji_list:
        emoji_counts = Counter(emoji_list).most_common(10)
        emoji_df = pd.DataFrame(emoji_counts, columns=['emoji', 'count'])
        fig, ax = plt.subplots()
        sns.barplot(x='count', y='emoji', data=emoji_df, ax=ax)
        st.pyplot(fig)
    else:
        st.info("No emojis found in chat.")

    # ---------- Common Words ----------
    st.subheader("🔤 Most Common Words")
    common_words = most_common_words(df['message'])
    cw_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
    fig, ax = plt.subplots()
    sns.barplot(x='Count', y='Word', data=cw_df, ax=ax)
    st.pyplot(fig)

    # ---------- Response Time ----------
    st.subheader("⏱️ Average Response Time (minutes)")
    response_times = {u: estimate_response_time(df, u) for u in df['user'].unique() if u != "group_notification"}
    if response_times:
        rt_df = pd.DataFrame(response_times.items(), columns=['User', 'Avg Response (min)']).dropna()
        fig, ax = plt.subplots()
        sns.barplot(x='Avg Response (min)', y='User', data=rt_df.sort_values('Avg Response (min)'), ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough data to estimate response times.")

    # ---------- Night Owl ----------
    st.subheader("🌙 Night Owl vs Early Bird Behavior")
    df['time_of_day'] = df['hour'].apply(lambda h: 'Night (10PM–5AM)' if h >= 22 or h < 5 else 'Day (5AM–10PM)')
    tod_counts = df['time_of_day'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(tod_counts.values, labels=tod_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)

    # ---------- AI Summary (Gemini) ----------
    st.subheader("🧩 AI Chat Summary & Insights (Gemini)")

    def summarize_in_chunks(text, chunk_size=5000):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        summaries = []
        model = genai.GenerativeModel("gemini-1.5-flash")
        for i, chunk in enumerate(chunks):
            with st.spinner(f"🧠 Summarizing part {i+1}/{len(chunks)}..."):
                try:
                    response = model.generate_content(f"Summarize this WhatsApp chat part:\n{chunk}")
                    summaries.append(response.text)
                    time.sleep(1)
                except Exception as e:
                    summaries.append(f"(Error in part {i+1}: {e})")
        return " ".join(summaries)

    if st.button("✨ Generate Chat Summary using Gemini"):
        with st.spinner("Analyzing your chat using Gemini..."):
            chat_text = "\n".join(df[df['user'] != 'group_notification']['message'].dropna().astype(str).tolist())
            chat_text = chat_text[:30000]
            partial_summary = summarize_in_chunks(chat_text)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"""
            Combine these summaries into one cohesive overview.
            Highlight tone, topics, and emotional trends:
            {partial_summary}
            """)
            st.success("✅ Summary generated successfully!")
            st.write(response.text)
