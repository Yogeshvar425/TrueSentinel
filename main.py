import os
import re
import csv
from collections import Counter
from googleapiclient.discovery import build
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Basic Stopwords for Topic Extraction
STOPWORDS = set([
    "the", "and", "to", "i", "a", "of", "in", "it", "is", "that", "this", 
    "for", "on", "you", "my", "with", "but", "so", "was", "are", "have", "be", 
    "they", "as", "at", "not", "we", "like", "just", "video", "how", "what", "can",
    "all", "your", "if", "will", "out", "about", "up", "from", "do", "who", "an",
    "me", "this", "there", "some", "more", "their", "them", "when", "would"
])

def extract_video_id(url):
    """Extracts the YouTube video ID from a standard URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def fetch_comments(video_id, max_results=100):
    """Fetches top-level comments using the YouTube Data API."""
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing. Add it to your .env file.")
        
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

def is_spam(comment):
    """LEVEL 2 FILTER: Detects and flags spam/bot comments."""
    comment_lower = comment.lower()
    
    # 1. Block URLs
    if "http://" in comment_lower or "https://" in comment_lower or "www." in comment_lower:
        return True
        
    # 2. Block Crypto / Investment Keywords
    spam_keywords = ["whatsapp", "telegram", "crypto", "bitcoin", "invest", "binance", "profit", "mentor", "scam"]
    for keyword in spam_keywords:
        if keyword in comment_lower:
            return True
            
    # 3. Block Repetitive Gibberish (e.g., 'llllll' or 'aaaaaaa')
    if re.search(r'(.)\1{10,}', comment_lower):
        return True
        
    return False

def extract_topics(comments, n_topics=3, n_words=4):
    """LEVEL 2 TOPIC MODELING: Uses Machine Learning (NMF) to find core subjects."""
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(STOPWORDS))
    try:
        tfidf = vectorizer.fit_transform(comments)
        nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(nmf.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append(", ".join(top_words))
        return topics
    except ValueError:
        return [] # Returns empty if not enough data to extract topics

def analyze_emotions(comments):
    """LEVEL 2 EMOTION AI: Uses HuggingFace DistilRoBERTa for complex emotion classification."""
    print("\n[~] Loading RoBERTa AI Emotion Model...")
    print("    (This may take a minute or two on the first run as the model downloads).")
    
    # We request all 7 probabilities (top_k=7) so we can dig past "neutral"
    emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=7)
    
    results = Counter()
    detailed_logs = []
    
    MAX_CHARS = 512 # Keep text within model limits
    
    print("[~] Classifying emotions...")
    for comment in comments:
        safe_comment = comment[:MAX_CHARS] # Truncate heavily long comments
        try:
            # Returns a list of the 7 emotions sorted by probability
            prediction = emotion_pipeline(safe_comment)[0]
            
            top_emotion = prediction[0]['label']
            confidence_score = prediction[0]['score']
            
            # >> SMART NEUTRAL DE-PRIORITIZATION FACTOR 
            # If the AI thinks it's neutral, but isn't absolutely certain (e.g., < 0.85% sure)
            # we dig down and grab whatever the second-highest emotion was.
            if top_emotion == 'neutral' and confidence_score < 0.85:
                second_emotion = prediction[1]['label']
                second_score = prediction[1]['score']
                
                # As long as the secondary emotion has at least *some* signal, promote it!
                if second_score >= 0.05:
                    top_emotion = second_emotion
                    confidence_score = second_score
            
            results[top_emotion] += 1
            detailed_logs.append({
                "emotion": top_emotion,
                "confidence": f"{confidence_score:.2f}",
                "comment": comment.replace('\n', ' ')
            })
        except Exception:
            # Safely catch weird encoding errors
            continue
            
    return results, detailed_logs

def main():
    print("=== YouTube Emotion & Topic Analyzer (Level 2) ===")
    
    url = input("\nEnter a YouTube URL: ")
    video_id = extract_video_id(url)
    
    if not video_id:
        print("Invalid YouTube URL!")
        return
        
    print(f"\n[+] Extracted Video ID: {video_id}")
    try:
        print("[+] Fetching comments...")
        raw_comments = fetch_comments(video_id, max_results=100)
        print(f"[+] Fetched {len(raw_comments)} raw comments.")
        
        if not raw_comments:
            print("No comments found or video comments are disabled.")
            return
            
        # >> FEATURE 1: SPAM FILTER
        clean_comments = [c for c in raw_comments if not is_spam(c)]
        spam_count = len(raw_comments) - len(clean_comments)
        print(f"[+] Filtering Step: Removed {spam_count} spam/bot comments.")
        
        if not clean_comments:
            print("No valid comments remaining after bot filtering.")
            return
            
        # >> FEATURE 2: TOPIC EXTRACTION
        print("[+] Extracting main topics discussed...")
        topics = extract_topics(clean_comments)
        
        # >> FEATURE 3: EMOTION AI
        emotion_counts, detailed_results = analyze_emotions(clean_comments)
        
        # Output Parsing
        total_analyzed = sum(emotion_counts.values())
        
        print("\n" + "="*40)
        print("           LEVEL 2 RESULTS           ")
        print("="*40)
        
        print("\n[ TOPICS IDENTIFIED IN CHAT ]")
        if topics:
            for i, topic in enumerate(topics, 1):
                print(f"  Topic {i}: [{topic}]")
        else:
            print("  Not enough varied text to extract clear topics.")
            
        print("\n[ TRUE EMOTION BREAKDOWN ]")
        for emotion, count in emotion_counts.most_common():
            pct = (count / total_analyzed) * 100
            print(f"  - {emotion.capitalize():<10}: {pct:5.1f}% ({count} comments)")
            
        print("\n[+] Saving detailed analysis to 'analysis_results.csv' for verification...")
        with open('analysis_results.csv', 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['emotion', 'confidence', 'comment'])
            writer.writeheader()
            writer.writerows(detailed_results)
        print("  -> Done! Open 'analysis_results.csv' in Excel to see how the AI graded each comment.")
            
    except Exception as e:
        print(f"\n[!] Error: {e}")

if __name__ == "__main__":
    main()
