import streamlit as st
import googleapiclient.discovery
from openai import OpenAI
import pandas as pd
import re

# Configure page
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

# Attempt to retrieve API keys from Streamlit Secrets
YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# Sidebar: Fallback for API keys if not in secrets.toml
with st.sidebar:
    st.title("API Configuration")
    if not YOUTUBE_API_KEY:
        YOUTUBE_API_KEY = st.text_input("Enter YouTube API Key", type="password")
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = st.text_input("Enter OpenAI API Key", type="password")

# Ensure API keys are available
if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
    st.warning("Please enter both API keys to proceed.")
    st.stop()

# Initialize APIs
youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
    match = re.search(r"(?:v=|youtu\.be/|embed/|watch\?v=|/v/|/e/|watch\?.*?&v=)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None

def get_comments(video_id, max_results=100):
    """Fetch YouTube comments."""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        
        comments = [
            {
                'text': item['snippet']['topLevelComment']['snippet']['textDisplay'],
                'likes': item['snippet']['topLevelComment']['snippet']['likeCount'],
                'author': item['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                'published_at': item['snippet']['topLevelComment']['snippet']['publishedAt']
            }
            for item in response.get('items', [])
        ]
        return pd.DataFrame(comments) if comments else None
    except Exception as e:
        st.error(f"Error fetching comments: {e}")
        return None

def analyze_comments(comments_text):
    """Analyze comments with OpenAI."""
    if not comments_text:
        return "No comments available for analysis."
    
    try:
        prompt = (
            "Analyze these YouTube comments and provide:\n"
            "1. Main topics/themes discussed\n"
            "2. Common sentiments\n"
            "3. Frequent questions or requests\n"
            "4. Notable feedback or suggestions\n\n"
            f"Comments:\n{comments_text}"
        )

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing comments: {e}")
        return None

# Main app
st.title("YouTube Comment Analyzer")
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)
    
    if video_id:
        with st.spinner("Fetching comments..."):
            df = get_comments(video_id)
        
        if df is not None and not df.empty:
            # Display top liked and latest comments
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 20 Most Liked Comments")
                st.dataframe(df.nlargest(20, 'likes')[['author', 'text', 'likes']], hide_index=True)

            with col2:
                st.subheader("Latest Comments")
                st.dataframe(df.sort_values('published_at', ascending=False).head(20)[['author', 'text', 'published_at']], hide_index=True)

            # Analyze comments
            st.subheader("Comment Analysis")
            with st.spinner("Analyzing comments..."):
                analysis_text = "\n".join(df['text'].head(50).tolist())
                analysis_result = analyze_comments(analysis_text)
                st.markdown(analysis_result)
        else:
            st.warning("No comments found for this video.")
    else:
        st.warning("Invalid YouTube video URL. Please enter a valid URL.")
