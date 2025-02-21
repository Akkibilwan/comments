import streamlit as st
import googleapiclient.discovery
from openai import OpenAI
import pandas as pd
from collections import defaultdict

# Configure page
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")

# Initialize session state for API keys
if 'youtube_api_key' not in st.session_state:
    st.session_state['youtube_api_key'] = ''
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return url.split('v=')[1][:11]
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1][:11]
    return url[:11]

def get_comments(youtube, video_id, max_results=100):
    """Fetch comments from YouTube video"""
    try:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        
        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'text': comment['textDisplay'],
                'likes': comment['likeCount'],
                'author': comment['authorDisplayName'],
                'published_at': comment['publishedAt']
            })
        
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return None

def analyze_comments_with_openai(comments_text, client):
    """Analyze comments using OpenAI API"""
    try:
        prompt = f"""Analyze these YouTube comments and provide:
        1. Main topics/themes discussed
        2. Common sentiments
        3. Frequent questions or requests
        4. Notable feedback or suggestions
        
        Comments:
        {comments_text}"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing comments: {str(e)}")
        return None

# Sidebar for API keys
with st.sidebar:
    st.title("API Configuration")
    youtube_api_key = st.text_input("YouTube API Key", type="password", key="youtube_key_input")
    openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key_input")
    
    if youtube_api_key:
        st.session_state['youtube_api_key'] = youtube_api_key
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key

# Main app
st.title("YouTube Comment Analyzer")

# Video URL input
video_url = st.text_input("Enter YouTube Video URL")

if video_url and st.session_state['youtube_api_key'] and st.session_state['openai_api_key']:
    # Initialize APIs
    youtube = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=st.session_state['youtube_api_key']
    )
    openai_client = OpenAI(api_key=st.session_state['openai_api_key'])
    
    # Get video ID and fetch comments
    video_id = extract_video_id(video_url)
    
    with st.spinner("Fetching comments..."):
        df = get_comments(youtube, video_id)
        
    if df is not None and not df.empty:
        # Display stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 20 Most Liked Comments")
            top_liked = df.nlargest(20, 'likes')[['author', 'text', 'likes']]
            st.dataframe(top_liked, hide_index=True)
        
        with col2:
            st.subheader("Latest Comments")
            latest = df.sort_values('published_at', ascending=False).head(20)[['author', 'text', 'published_at']]
            st.dataframe(latest, hide_index=True)
        
        # Analyze comments with OpenAI
        st.subheader("Comment Analysis")
        with st.spinner("Analyzing comments..."):
            # Prepare comments for analysis
            analysis_text = "\n".join(df['text'].head(50).tolist())  # Analyze top 50 comments
            analysis = analyze_comments_with_openai(analysis_text, openai_client)
            
            if analysis:
                st.markdown(analysis)
    
elif not st.session_state['youtube_api_key'] or not st.session_state['openai_api_key']:
    st.warning("Please enter both API keys in the sidebar to proceed.")
