"""
TrueSentinel — Model Training Pipeline
=======================================
Harvests live YouTube comments, labels via lexicon rules, engineers TF-IDF features,
trains Logistic Regression + SVM, performs Stratified K-Fold cross-validation,
and exports real evaluation metrics to models/metrics.json.
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import json
import logging
from datetime import datetime
from googleapiclient.discovery import build
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from dotenv import load_dotenv

# ─── Config ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("TrueSentinel.Train")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# ─── Text Preprocessing (shared logic) ──────────────────────────
def preprocess_text(text: str) -> str:
    """Normalize text for consistent vectorization between training and inference."""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)          # Strip URLs
    text = re.sub(r'[^a-z\s\'\-]', ' ', text)             # Keep only letters, apostrophes, hyphens
    text = re.sub(r'\s+', ' ', text).strip()               # Collapse whitespace
    return text

# ─── Spam Detection ─────────────────────────────────────────────
SPAM_KEYWORDS = [
    "whatsapp", "telegram", "crypto", "bitcoin", "invest",
    "binance", "profit", "mentor", "scam", "check my", "sub to"
]

def is_spam(comment: str) -> bool:
    """Flags spam/bot comments using URL detection, keyword matching, and gibberish detection."""
    lower = comment.lower()
    if re.search(r'https?://|www\.', lower):
        return True
    if any(kw in lower for kw in SPAM_KEYWORDS):
        return True
    if re.search(r'(.)\1{10,}', lower):
        return True
    return False

# ─── Lexicon Labeling ───────────────────────────────────────────
POSITIVE_KEYWORDS = [
    'good', 'great', 'awesome', 'excellent', 'amazing', 'love', 'best',
    'cool', 'masterpiece', 'perfect', 'beautiful', 'excited', 'hype',
    'wow', 'nice', '10/10', 'fantastic', 'brilliant', 'wonderful',
    'outstanding', 'incredible', 'superb', 'recommend', 'favorite'
]
NEGATIVE_KEYWORDS = [
    'bad', 'terrible', 'awful', 'worst', 'hate', 'trash', 'garbage',
    'sad', 'angry', 'boring', 'disappointing', 'ruined', 'woke', 'sucks',
    'fake', 'annoying', 'shitty', 'shit', 'crap', 'horrible', 'pathetic',
    'overrated', 'cringe', 'waste', 'dislike', 'disgusting', 'mediocre'
]

def label_comment(comment: str) -> int | None:
    """Assigns sentiment label via lexicon matching with double-negative handling."""
    lower = comment.lower()

    pos = sum(1 for w in POSITIVE_KEYWORDS if w in lower)
    neg = sum(1 for w in NEGATIVE_KEYWORDS if w in lower)

    # Double-negative / slang patches
    if "not bad" in lower or "no complaints" in lower:
        pos += 5; neg = 0
    if "not good" in lower or "not great" in lower:
        neg += 5; pos = 0

    if pos > neg:
        return 1   # Positive
    elif neg > pos:
        return 0   # Negative
    return None     # Ambiguous — skip


# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════
def main():
    log.info("═" * 60)
    log.info("  TrueSentinel — Training Pipeline v2.0")
    log.info("═" * 60)

    if not YOUTUBE_API_KEY:
        log.error("YOUTUBE_API_KEY is missing in .env file.")
        return

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    # ── Phase 1: Data Harvesting ──────────────────────────────
    training_videos = [
        "dQw4w9WgXcQ",   # Rick Astley — Never Gonna Give You Up (Mixed/Fun)
        "jNQXAC9IVRw",   # Me at the zoo (First YouTube Video)
        "YbJOTdZBX1g",   # YouTube Rewind 2018 (Extremely Negative)
        "kJQP7kiw5Fk",   # Despacito (High engagement)
        "9bZkp7q19f0",   # Gangnam Style (Global viral)
        "RgKAFK5djSk",   # See You Again (Emotional, mixed)
    ]

    log.info("Phase 1: Harvesting training data from %d YouTube videos...", len(training_videos))
    all_comments = []

    for vid in training_videos:
        try:
            request = youtube.commentThreads().list(
                part="snippet", videoId=vid, maxResults=100, textFormat="plainText"
            )
            response = request.execute()
            fetched = [
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in response.get("items", [])
            ]
            all_comments.extend(fetched)
            log.info("  ✓ Video %s → %d comments", vid, len(fetched))
        except Exception as e:
            log.warning("  ✗ Video %s failed: %s", vid, e)

    # ── Pre-labeled training corpus (bypasses lexicon labeler) ──
    # These are injected directly into the labeled dataset (Phase 3) to ensure
    # the TF-IDF vocabulary is broad enough to generalize to unseen comments.
    # Without this, the lexicon labeler only captures ~50 comments from YouTube,
    # producing a model that defaults to "positive" on any unknown text.
    PRELABELED_CORPUS = [
        # ── POSITIVE (label=1) ─────────────────────────────────────
        # Enthusiastic / direct praise
        ("This is absolutely amazing, I love every second of it", 1),
        ("Best video I have ever watched on YouTube, hands down", 1),
        ("Incredible work, keep it up, you deserve more subscribers", 1),
        ("This made my entire day better, thank you so much", 1),
        ("Pure gold content right here, everyone needs to see this", 1),
        ("I have watched this at least ten times and it never gets old", 1),
        ("The production quality is insane, so well done", 1),
        ("This deserves way more views than it has right now", 1),
        ("Honestly one of the most creative things I have seen online", 1),
        ("You are so talented, this is genuinely impressive", 1),
        # Casual / slang positive
        ("Bro this slaps so hard, instant like from me", 1),
        ("Fire content as always, never disappoints", 1),
        ("W video, absolute banger, lets gooo", 1),
        ("No cap this is elite tier content", 1),
        ("This goes crazy, you snapped on this one", 1),
        ("Yooo this is actually insane, im speechless", 1),
        ("Goated video fr fr, shared this with all my friends", 1),
        ("Peak content, this channel is underrated honestly", 1),
        ("Dude this is nuts, how is this free", 1),
        ("Lowkey the best thing on my feed today", 1),
        # Emotional positive
        ("This actually brought tears to my eyes, so beautiful", 1),
        ("I smiled the entire time watching this, so wholesome", 1),
        ("This song helped me through some really tough times", 1),
        ("The message in this video is so powerful and inspiring", 1),
        ("My heart is so full after watching this, thank you", 1),
        ("This gives me so much hope for humanity", 1),
        ("I am so glad I found this channel, it changed my perspective", 1),
        ("This hit me right in the feels, crying happy tears", 1),
        # Technical / specific praise
        ("The editing in this video is next level professional", 1),
        ("Sound design is chef kiss, you really nailed the atmosphere", 1),
        ("The research that went into this is clearly extensive", 1),
        ("Your explanation is the clearest I have found on this topic", 1),
        ("Finally someone explains this properly, thank you", 1),
        ("The visuals are stunning and the pacing is perfect", 1),
        ("Camera work and lighting are on point in this", 1),
        ("This tutorial actually worked, unlike all the other ones", 1),
        # Agreement / support
        ("Completely agree with everything said here", 1),
        ("This is exactly what I was looking for", 1),
        ("Subscribed immediately after watching this", 1),
        ("Take my like and subscribe, you earned it", 1),
        ("Adding this to my favorites playlist right now", 1),
        ("Came back to watch this again, still just as good", 1),
        ("Sharing this everywhere because people need to see it", 1),
        ("You have a new fan, this is quality content", 1),
        # Humor positive
        ("I cant stop laughing, this is comedy gold", 1),
        ("This is hilarious, my stomach hurts from laughing", 1),
        ("Funniest thing I have seen all week, well done", 1),
        ("The humor in this is so clever and well timed", 1),
        # Short positive
        ("Love this so much", 1),
        ("Perfection", 1),
        ("Simply beautiful", 1),
        ("Masterpiece of a video", 1),
        ("Absolutely brilliant", 1),
        ("So good it hurts", 1),
        ("This is everything", 1),
        ("Chef kiss to whoever made this", 1),
        ("Respect for this content", 1),
        ("Outstanding work", 1),

        # ── NEGATIVE (label=0) ─────────────────────────────────────
        # Direct criticism
        ("This is terrible, I can not believe someone made this", 0),
        ("Worst video I have seen in a long time, total waste of time", 0),
        ("I regret clicking on this, complete garbage", 0),
        ("How does this have so many views, it is awful", 0),
        ("This is painfully bad, had to stop watching halfway through", 0),
        ("Absolute trash content, disliked and reported", 0),
        ("I want my ten minutes back, this was horrible", 0),
        ("This is so bad it actually made me angry", 0),
        ("What a letdown, expected much better from this channel", 0),
        ("Unwatchable, the quality is embarrassingly low", 0),
        # Casual / slang negative
        ("Bro this is mid at best, actually no it is just bad", 0),
        ("L video, huge miss, what happened to this channel", 0),
        ("This aint it chief, delete this", 0),
        ("Cringe level is off the charts with this one", 0),
        ("Who asked for this content literally nobody", 0),
        ("This fell off hard, used to be a good channel", 0),
        ("Hard pass on this one, not worth anyone time", 0),
        ("This is giving clown energy, so embarrassing", 0),
        ("Nah fam this is garbage tier", 0),
        ("Yikes this is rough, big thumbs down", 0),
        # Emotional negative
        ("This made me genuinely sad and not in a good way", 0),
        ("I feel depressed after watching this, so negative", 0),
        ("The negativity in this video is overwhelming and toxic", 0),
        ("This ruined my mood completely, wish I had not watched", 0),
        ("So disappointing, I had such high hopes for this", 0),
        ("This broke my heart but not in an artistic way just bad", 0),
        ("I feel worse about the world after seeing this", 0),
        ("This content is harmful and should not be promoted", 0),
        # Technical / specific criticism
        ("The audio quality is atrocious, sounds like it was recorded on a potato", 0),
        ("Editing is choppy and the transitions are jarring", 0),
        ("Zero research went into this, full of misinformation", 0),
        ("The explanation is confusing and completely wrong", 0),
        ("Clickbait title with no actual substance in the video", 0),
        ("The pacing is terrible, drags on forever about nothing", 0),
        ("Production value of a middle school project honestly", 0),
        ("This tutorial does not work at all, wasted my afternoon", 0),
        # Disagreement / rejection
        ("Completely disagree with everything in this video", 0),
        ("Unsubscribing after this, the channel went downhill", 0),
        ("Would give zero stars if I could, truly awful", 0),
        ("Do not waste your time watching this I am warning you", 0),
        ("Removing this from my recommendations immediately", 0),
        ("How is this trending when there is so much better content", 0),
        ("The like to dislike ratio says it all about this video", 0),
        ("Not watching anything from this creator ever again", 0),
        # Boring / unimpressed
        ("So boring I almost fell asleep watching this", 0),
        ("Incredibly dull and unoriginal content", 0),
        ("Nothing new here, same recycled content as everyone else", 0),
        ("The most generic and uninspired video I have ever seen", 0),
        ("Mediocre at best and that is being generous", 0),
        ("Could not even finish it, lost interest after one minute", 0),
        ("This is the definition of low effort content", 0),
        ("Bland, forgettable, and completely pointless", 0),
        # Annoyed / frustrated
        ("Stop with the clickbait titles already, it is so annoying", 0),
        ("The constant self promotion in this is unbearable", 0),
        ("Another terrible video with an obnoxious thumbnail", 0),
        ("This creator only cares about money not quality", 0),
        ("Sick and tired of this type of low quality content", 0),
        ("Please just stop making videos if this is the quality", 0),
        # Short negative
        ("Absolute garbage", 0),
        ("Terrible just terrible", 0),
        ("What a waste", 0),
        ("Disgusting content", 0),
        ("Hard cringe", 0),
        ("Not funny at all", 0),
        ("Total disappointment", 0),
        ("Pathetic effort", 0),
        ("Horrendous quality", 0),
        ("Truly awful stuff", 0),
        ("Unwatchable trash", 0),
        ("Below average in every way", 0),
    ]

    # Add pre-labeled corpus directly to labeled_data (bypass lexicon labeler)
    prelabeled_data = [{"comment": c, "sentiment": s} for c, s in PRELABELED_CORPUS]
    log.info("  Pre-labeled training corpus: %d samples (%d pos, %d neg)",
             len(PRELABELED_CORPUS),
             sum(1 for _, s in PRELABELED_CORPUS if s == 1),
             sum(1 for _, s in PRELABELED_CORPUS if s == 0))


    total_raw = len(all_comments)
    log.info("Total raw comments harvested: %d", total_raw)

    if total_raw == 0:
        log.error("No comments fetched. Check API key or network.")
        return

    # ── Phase 2: Preprocessing + Spam Filtering ───────────────
    log.info("Phase 2: Preprocessing & spam filtering...")
    clean_comments = [c for c in all_comments if not is_spam(c)]
    spam_count = total_raw - len(clean_comments)
    log.info("  Spam/bot comments removed: %d", spam_count)

    # ── Phase 3: Lexicon Labeling (YouTube comments) ────────────
    log.info("Phase 3: Generating ground-truth labels (lexicon rules)...")
    labeled_data = []
    for comment in clean_comments:
        label = label_comment(comment)
        if label is not None and len(comment) > 10:
            labeled_data.append({"comment": comment, "sentiment": label})

    yt_pos = sum(1 for d in labeled_data if d["sentiment"] == 1)
    yt_neg = sum(1 for d in labeled_data if d["sentiment"] == 0)
    log.info("  YouTube lexicon-labeled: %d (%d pos, %d neg)", len(labeled_data), yt_pos, yt_neg)

    # Merge with pre-labeled corpus
    labeled_data.extend(prelabeled_data)

    df = pd.DataFrame(labeled_data)
    pos_count = len(df[df["sentiment"] == 1])
    neg_count = len(df[df["sentiment"] == 0])
    log.info("  Labeled dataset: %d total (%d positive, %d negative)", len(df), pos_count, neg_count)

    if len(df) < 20:
        log.error("Insufficient labeled data (%d). Need at least 20.", len(df))
        return

    # ── Phase 4: Feature Engineering ──────────────────────────
    log.info("Phase 4: TF-IDF vectorization (unigrams + bigrams)...")
    X_text = df["comment"].apply(preprocess_text)
    y = df["sentiment"].astype(int)

    # Keeping "not" by NOT using stop_words='english' — "not bad" would become "bad"
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(X_text)
    log.info("  Feature matrix shape: %s", X.shape)

    # ── Phase 5: Train/Test Split ─────────────────────────────
    log.info("Phase 5: Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("  Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])

    # ── Phase 6: Model Training ───────────────────────────────
    log.info("Phase 6: Training models...")

    log.info("  → Logistic Regression (class_weight='balanced')...")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    log.info("  → SVM (linear kernel, class_weight='balanced')...")
    svm_model = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
    svm_model.fit(X_train, y_train)

    # ── Phase 7: Evaluation (REAL metrics) ────────────────────
    log.info("Phase 7: Evaluating on held-out test set...")

    def evaluate_model(model, name, X_test, y_test):
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        cm = confusion_matrix(y_test, preds).tolist()

        log.info("  %s → Accuracy: %.2f%% | F1: %.2f%% | Precision: %.2f%% | Recall: %.2f%%",
                 name, acc * 100, f1 * 100, prec * 100, rec * 100)

        return {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm
        }

    lr_metrics = evaluate_model(lr_model, "Logistic Regression", X_test, y_test)
    svm_metrics = evaluate_model(svm_model, "SVM", X_test, y_test)

    # ── Phase 8: Cross-Validation ─────────────────────────────
    log.info("Phase 8: 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_cv = cross_validate(
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        X, y, cv=cv, scoring=["accuracy", "f1"], return_train_score=False
    )
    svm_cv = cross_validate(
        SVC(kernel="linear", class_weight="balanced", random_state=42),
        X, y, cv=cv, scoring=["accuracy", "f1"], return_train_score=False
    )

    lr_cv_acc = lr_cv["test_accuracy"].mean() * 100
    lr_cv_f1 = lr_cv["test_f1"].mean() * 100
    svm_cv_acc = svm_cv["test_accuracy"].mean() * 100
    svm_cv_f1 = svm_cv["test_f1"].mean() * 100

    log.info("  LR  CV → Accuracy: %.2f%% ± %.2f | F1: %.2f%% ± %.2f",
             lr_cv_acc, lr_cv["test_accuracy"].std() * 100,
             lr_cv_f1, lr_cv["test_f1"].std() * 100)
    log.info("  SVM CV → Accuracy: %.2f%% ± %.2f | F1: %.2f%% ± %.2f",
             svm_cv_acc, svm_cv["test_accuracy"].std() * 100,
             svm_cv_f1, svm_cv["test_f1"].std() * 100)

    # ── Phase 9: Export Models + Metrics ──────────────────────
    log.info("Phase 9: Exporting models and metrics...")

    with open(os.path.join(MODELS_DIR, "logistic_model.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    metrics = {
        "trained_at": datetime.now().isoformat(),
        "dataset": {
            "total_raw_comments": total_raw,
            "spam_filtered": spam_count,
            "labeled_samples": len(df),
            "positive_samples": pos_count,
            "negative_samples": neg_count,
            "feature_dimensions": X.shape[1],
            "source_videos": len(training_videos)
        },
        "logistic_regression": {
            **lr_metrics,
            "cv_accuracy": round(lr_cv_acc, 2),
            "cv_accuracy_std": round(lr_cv["test_accuracy"].std() * 100, 2),
            "cv_f1": round(lr_cv_f1, 2),
            "cv_f1_std": round(lr_cv["test_f1"].std() * 100, 2),
        },
        "svm": {
            **svm_metrics,
            "cv_accuracy": round(svm_cv_acc, 2),
            "cv_accuracy_std": round(svm_cv["test_accuracy"].std() * 100, 2),
            "cv_f1": round(svm_cv_f1, 2),
            "cv_f1_std": round(svm_cv["test_f1"].std() * 100, 2),
        }
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("  ✓ Metrics saved to models/metrics.json")

    # Save training CSV
    df.to_csv(os.path.join(MODELS_DIR, "training_data.csv"), index=False)
    log.info("  ✓ Training data saved to models/training_data.csv")

    log.info("═" * 60)
    log.info("  Training complete. Models ready for deployment.")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
