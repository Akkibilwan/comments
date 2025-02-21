import streamlit as st
import googleapiclient.discovery
from openai import OpenAI
import pandas as pd
import re

# Configure Streamlit Page
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

# Load API Keys from Secrets
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Initialize APIs
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def extract_video_id(url):
    """Extract YouTube Video ID from URL."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|watch\?v=|/v/|/e/|watch\?.*?&v=)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id, max_results=50):
    """Fetch YouTube comments."""
    try:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        ).execute()
        
        comments = [
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in response.get('items', [])
        ]
        return comments if comments else None
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return None

def analyze_comments(comments_text):
    """Analyze comments with OpenAI."""
    if not comments_text:
        return "No comments available for analysis."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": 
                "Analyze these YouTube comments and provide:\n"
                "1. Main discussion topics\n"
                "2. Common sentiments\n"
                "3. Any frequent questions or feedback\n\n"
                f"Comments:\n{comments_text}"}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing comments: {e}")
        return None

# Streamlit UI
st.title("📊 YouTube Comment Analyzer")
video_url = st.text_input("🔗 Enter YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)

    if video_id:
        with st.spinner("Fetching comments..."):
            comments = get_comments(video_id)

        if comments:
            st.subheader("🔹 AI Analysis of Comments")
            with st.spinner("Analyzing comments..."):
                analysis_result = analyze_comments("\n".join(comments[:50]))
                st.markdown(analysis_result)
        else:
            st.warning("⚠️ No comments found for this video.")
    else:
        st.warning("⚠️ Invalid YouTube URL. Please enter a correct one.")
