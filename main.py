import os
import re
from collections import Counter
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Basic Stopwords for Top Words Extraction
STOPWORDS = set([
    "the", "and", "to", "i", "a", "of", "in", "it", "is", "that", "this", 
    "for", "on", "you", "my", "with", "but", "so", "was", "are", "have", "be", 
    "they", "as", "at", "not", "we", "like", "just", "video", "how", "what", "can"
])

def extract_video_id(url):
    """Extracts the YouTube video ID from a standard or shortened URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def fetch_comments(video_id, max_results=100):
    """Fetches top-level comments using the YouTube API."""
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing. Please add it to your .env file.")
        
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    
    response = request.execute()
    
    comments = []
    for item in response.get("items", []):
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)
        
    return comments

def preprocess_text(text):
    """Basic preprocessing: lowercase and remove basic punctuation for word counts."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def analyze_comments(comments):
    """Runs VADER sentiment analysis on the raw comments."""
    analyzer = SentimentIntensityAnalyzer()
    
    results = {"positive": 0, "neutral": 0, "negative": 0}
    all_words = []
    
    for comment in comments:
        # 1. Sentiment
        score = analyzer.polarity_scores(comment)
        compound = score['compound']
        
        if compound >= 0.05:
            results["positive"] += 1
        elif compound <= -0.05:
            results["negative"] += 1
        else:
            results["neutral"] += 1
            
        # 2. Extract words
        clean_comment = preprocess_text(comment)
        words = clean_comment.split()
        good_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        all_words.extend(good_words)
        
    return results, all_words

def main():
    print("=== YouTube Comment Sentiment Analyzer MVP ===")
    
    url = input("Enter a YouTube URL: ")
    video_id = extract_video_id(url)
    
    if not video_id:
        print("Invalid YouTube URL!")
        return
        
    print(f"\n[+] Extracted Video ID: {video_id}")
    try:
        print("[+] Fetching comments (up to 100)...")
        comments = fetch_comments(video_id)
        print(f"[+] Fetched {len(comments)} comments.")
        
        if not comments:
            print("No comments found or video comments are disabled.")
            return
            
        sentiment_counts, all_words = analyze_comments(comments)
        
        # Calculate percentages
        total = len(comments)
        pct_pos = (sentiment_counts['positive'] / total) * 100
        pct_neu = (sentiment_counts['neutral'] / total) * 100
        pct_neg = (sentiment_counts['negative'] / total) * 100
        
        # Extract Top 5 words
        top_words = Counter(all_words).most_common(5)
        
        print("\n=== RESULTS ===")
        print(f"Positive: {pct_pos:.1f}%")
        print(f"Neutral:  {pct_neu:.1f}%")
        print(f"Negative: {pct_neg:.1f}%")
        
        print("\nTop Words associated:")
        for word, count in top_words:
            print(f"  - {word}: {count} times")
            
    except Exception as e:
        print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    main()
