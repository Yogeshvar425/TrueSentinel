"""
TrueSentiment — API Server
=========================
FastAPI backend that loads custom-trained models and serves real-time
sentiment analysis with cumulative impact metrics tracking.
"""

import os
import re
import json
import time
import logging
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from dotenv import load_dotenv
import pickle

# ─── Config ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TrueSentiment.Server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATS_PATH = os.path.join(MODELS_DIR, "cumulative_stats.json")

load_dotenv(os.path.join(BASE_DIR, ".env"))
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ─── Text Preprocessing (must match train.py exactly) ───────────
def preprocess_text(text: str) -> str:
    """Normalize text for consistent vectorization between training and inference."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z\s\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─── App + Lifespan ─────────────────────────────────────────────
app = FastAPI(
    title="TrueSentiment API",
    description="Real-time YouTube comment sentiment analysis with custom ML models",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ──────────────────────────────────────────────
log.info("Loading ML models...")
sentiment_model = None
vectorizer = None
training_metrics = None
server_start_time = time.time()

try:
    with open(os.path.join(MODELS_DIR, "logistic_model.pkl"), "rb") as f:
        sentiment_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    log.info("  ✓ Models loaded successfully")
except FileNotFoundError:
    log.warning("  ✗ Model files not found. Run train.py first.")

try:
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        training_metrics = json.load(f)
    log.info("  ✓ Training metrics loaded")
except FileNotFoundError:
    log.warning("  ✗ metrics.json not found. Training metrics will be unavailable.")

# ─── Cumulative Stats (persisted across restarts) ───────────────
def load_stats() -> dict:
    defaults = {
        "total_analyses": 0,
        "total_comments_processed": 0,
        "total_spam_blocked": 0,
        "total_videos_analyzed": 0,
        "first_used": datetime.now().isoformat(),
        "last_used": None
    }
    try:
        with open(STATS_PATH, "r") as f:
            saved = json.load(f)
            defaults.update(saved)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults

def save_stats(stats: dict):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

cumulative_stats = load_stats()

# ─── Shared Utilities ───────────────────────────────────────────
STOPWORDS = set([
    "the", "and", "to", "i", "a", "of", "in", "it", "is", "that", "this",
    "for", "on", "you", "my", "with", "but", "so", "was", "are", "have", "be",
    "they", "as", "at", "not", "we", "like", "just", "video", "how", "what", "can",
    "all", "your", "if", "will", "out", "about", "up", "from", "do", "who", "an",
    "me", "this", "there", "some", "more", "their", "them", "when", "would"
])

SPAM_KEYWORDS = [
    "whatsapp", "telegram", "crypto", "bitcoin", "invest",
    "binance", "profit", "mentor", "scam", "check my", "sub to"
]

def extract_video_id(url: str) -> str | None:
    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

def fetch_comments(video_id: str, max_results: int = 100) -> list[str]:
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY is missing.")
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id,
        maxResults=max_results, textFormat="plainText"
    )
    response = request.execute()
    return [
        item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        for item in response.get("items", [])
    ]

def is_spam(comment: str) -> bool:
    lower = comment.lower()
    if re.search(r'https?://|www\.', lower):
        return True
    if any(kw in lower for kw in SPAM_KEYWORDS):
        return True
    if re.search(r'(.)\1{10,}', lower):
        return True
    return False

def extract_topics(comments: list[str], n_topics: int = 3, n_words: int = 4) -> list[str]:
    topic_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words=list(STOPWORDS))
    try:
        tfidf = topic_vectorizer.fit_transform(comments)
        nmf = NMF(n_components=n_topics, random_state=42).fit(tfidf)
        feature_names = topic_vectorizer.get_feature_names_out()
        return [
            ", ".join(feature_names[i] for i in topic.argsort()[:-n_words - 1:-1])
            for topic in nmf.components_
        ]
    except ValueError:
        return []

def predict_sentiment(comments: list[str]) -> tuple[Counter, list[dict]]:
    results = Counter()
    detailed = []

    if not comments or not sentiment_model or not vectorizer:
        return results, detailed

    # Preprocess EXACTLY as in training
    processed = [preprocess_text(c) for c in comments]

    try:
        X = vectorizer.transform(processed)
        predictions = sentiment_model.predict(X)
        probabilities = sentiment_model.predict_proba(X)

        for i, comment in enumerate(comments):
            label = "positive" if predictions[i] == 1 else "negative"
            confidence = float(max(probabilities[i]))
            results[label] += 1
            detailed.append({
                "emotion": label,
                "confidence": round(confidence, 3),
                "comment": comment.replace("\n", " ")[:300]
            })
    except Exception as e:
        log.error("Prediction error: %s", e)

    return results, detailed

# ─── API Models ─────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    url: str

# ─── Routes ─────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """Returns server health, model status, training metrics, and cumulative impact stats."""
    uptime_seconds = int(time.time() - server_start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "status": "healthy" if sentiment_model else "degraded",
        "uptime": f"{hours}h {minutes}m {seconds}s",
        "model_loaded": sentiment_model is not None,
        "training_metrics": training_metrics,
        "cumulative_stats": cumulative_stats
    }


@app.post("/api/analyze")
async def analyze_video(req: AnalyzeRequest):
    video_id = extract_video_id(req.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    if not sentiment_model:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    start_time = time.time()

    try:
        raw_comments = fetch_comments(video_id, max_results=100)
        clean_comments = [c for c in raw_comments if not is_spam(c)]
        spam_count = len(raw_comments) - len(clean_comments)

        if not clean_comments:
            raise HTTPException(status_code=400, detail="No valid comments found after spam filtering.")

        topics = extract_topics(clean_comments)
        emotion_counts, detailed_results = predict_sentiment(clean_comments)

        total = sum(emotion_counts.values())
        emotions_response = [
            {"name": e, "count": c, "percentage": round((c / total) * 100, 1)}
            for e, c in emotion_counts.most_common()
        ]

        inference_time_ms = round((time.time() - start_time) * 1000, 1)

        # ── Update cumulative impact stats ──
        cumulative_stats["total_analyses"] += 1
        cumulative_stats["total_comments_processed"] += len(clean_comments)
        cumulative_stats["total_spam_blocked"] += spam_count
        cumulative_stats["total_videos_analyzed"] += 1
        cumulative_stats["last_used"] = datetime.now().isoformat()
        save_stats(cumulative_stats)

        # Build response with REAL metrics
        response = {
            "video_id": video_id,
            "total_analyzed": len(clean_comments),
            "spam_filtered": spam_count,
            "inference_time_ms": inference_time_ms,
            "topics": topics,
            "emotions": emotions_response,
            "detailed_comments": detailed_results,
            "cumulative_stats": {
                "total_analyses": cumulative_stats["total_analyses"],
                "total_comments_processed": cumulative_stats["total_comments_processed"],
                "total_spam_blocked": cumulative_stats["total_spam_blocked"],
            }
        }

        # Attach real training metrics if available
        if training_metrics:
            response["training_metrics"] = {
                "logistic_regression": {
                    "accuracy": training_metrics["logistic_regression"]["accuracy"],
                    "f1_score": training_metrics["logistic_regression"]["f1_score"],
                    "precision": training_metrics["logistic_regression"]["precision"],
                    "recall": training_metrics["logistic_regression"]["recall"],
                    "cv_accuracy": training_metrics["logistic_regression"]["cv_accuracy"],
                },
                "svm": {
                    "accuracy": training_metrics["svm"]["accuracy"],
                    "f1_score": training_metrics["svm"]["f1_score"],
                    "precision": training_metrics["svm"]["precision"],
                    "recall": training_metrics["svm"]["recall"],
                    "cv_accuracy": training_metrics["svm"]["cv_accuracy"],
                },
                "dataset": training_metrics["dataset"],
                "trained_at": training_metrics["trained_at"]
            }
        else:
            response["training_metrics"] = None

        log.info("Analysis complete: %d comments in %.1fms (video: %s)",
                 len(clean_comments), inference_time_ms, video_id)

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("Analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Static Files ───────────────────────────────────────────────
static_dir = os.path.join(BASE_DIR, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# ─── Entrypoint ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info("Starting TrueSentiment server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
