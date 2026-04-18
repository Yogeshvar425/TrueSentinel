# YouTube Comment Sentiment Analyzer MVP

A simple Python application that fetches comments from a given YouTube video and analyzes their overall sentiment using VADER sentiment analysis. It also extracts the most commonly used words overall.

## Setup

1. Ensure you have Python installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your YouTube Data API v3 Key:
   ```env
   YOUTUBE_API_KEY=your_api_key_here
   ```

## Usage

Run the program from your terminal:
```bash
python main.py
```

When prompted, paste the URL of the YouTube video you want to analyze. The application will fetch up to 100 top-level comments and generate a sentiment breakdown (Positive, Neutral, Negative) along with the top 5 most frequently associated words.

## Dependencies

- `google-api-python-client`: For fetching comments via the YouTube API.
- `vaderSentiment`: For generating sentiment scores and analyzing text.
- `python-dotenv`: For managing the environment variables (like API keys) securely.
