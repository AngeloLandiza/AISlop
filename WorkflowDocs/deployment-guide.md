# YouTube Shorts Automation: Deployment & Advanced Guide

## Part 1: Getting Free API Keys (Step-by-Step)

### 1.1 Cohere API (Story Generation)

**Why Cohere?** Free tier: 1,000 API calls/month, excellent for emotional storytelling

**Steps:**
1. Go to https://dashboard.cohere.io/
2. Click "Sign Up"
3. Verify email
4. Copy your **API Key**
5. Add to `config.yaml`:

```yaml
APIs:
  cohere:
    api_key: "YOUR_COHERE_KEY_HERE"
```

**Monthly allowance:** 1,000 stories/month = ~30-33 per day (perfect for 3 uploads/day)

---

### 1.2 Google Cloud (Voice Synthesis + YouTube)

**Why Google?** Free tier includes: 1M characters/month for TTS + 1M free YouTube API calls

**Steps:**

1. Go to https://cloud.google.com/
2. Click "Try Free" → Create account
3. Get $300 credit for 90 days
4. Create new project:
   - Click project selector (top left)
   - Click "New Project"
   - Name it "YouTube-Shorts-Automation"
   - Click "Create"

5. Enable APIs:
   - Go to "APIs & Services"
   - Click "Enable APIs and Services"
   - Search for "YouTube Data API v3"
   - Click "Enable"
   - Repeat for "Cloud Text-to-Speech API"

6. Create OAuth 2.0 credentials:
   - Go to "Credentials"
   - Click "Create Credentials" → "OAuth 2.0 Client IDs"
   - Choose "Desktop application"
   - Click "Create"
   - Download JSON file
   - Rename to `client_secret.json`

7. Create Service Account for TTS:
   - Go to "Service Accounts"
   - Click "Create Service Account"
   - Name: "youtube-shorts-automation"
   - Click "Create and Continue"
   - Grant role: "Editor" (for this project)
   - Click "Continue"
   - Create JSON key
   - Rename to `google-credentials.json`

**Free Tier Details:**
- Text-to-Speech: **1 million characters/month**
- YouTube API: **Unlimited** (quota: 10K units/day, plenty for uploads)
- Cloud Storage: **5 GB free** (for temporary video files)

---

### 1.3 ElevenLabs API (Alternative Voice Synthesis)

**Why ElevenLabs?** More natural AI voices, optional free tier

**Steps:**
1. Go to https://elevenlabs.io/
2. Sign up with email
3. Free tier: Limited characters/month but available
4. Copy your **API Key**

**Optional** - only use if you want better voices than Google TTS

---

## Part 2: Deploying Locally (Your Computer)

### 2.1 Initial Setup

```bash
# Clone project or create new directory
mkdir youtube-shorts-automation
cd youtube-shorts-automation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html

# Create required directories
mkdir videos logs
```

### 2.2 First Run (Authentication)

```bash
# First time: authenticate with YouTube
python main.py --auth

# This opens browser, authorize access, then returns to terminal
# Credentials saved automatically in youtube_token.pickle

# Test single pipeline
python main.py --once

# If successful, you should see:
# [1/3] Generating story... ✓
# [2/3] Creating video... ✓
# [3/3] Uploading to YouTube... ✓
```

### 2.3 Continuous Operation

```bash
# Start automated scheduler
python main.py --schedule

# Runs at:
# - 08:00 AM
# - 02:00 PM
# - 08:00 PM
# Every day, completely hands-free

# Stop with Ctrl+C
```

---

## Part 3: Deploying to Cloud (Google Cloud Run)

### 3.1 Why Cloud? 

- ✅ 24/7 uptime (never stops uploading)
- ✅ Free tier available ($300 credit)
- ✅ No need to keep personal computer running
- ✅ Automatically scales if needed

### 3.2 Deploy to Google Cloud Run

```bash
# 1. Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# 2. Authenticate
gcloud auth login

# 3. Set project
gcloud config set project YOUR_PROJECT_ID

# 4. Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run automation
CMD ["python", "main.py", "--schedule"]
EOF

# 5. Create requirements.txt
pip freeze > requirements.txt

# 6. Build and deploy
gcloud run deploy youtube-shorts \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600 \
  --set-env-vars COHERE_API_KEY=your_key,GOOGLE_CREDENTIALS_FILE=google-credentials.json

# 7. Done! Service runs continuously at no cost
```

### 3.3 Monitor Cloud Deployment

```bash
# View logs
gcloud run logs read youtube-shorts --limit 50

# View service status
gcloud run services describe youtube-shorts

# View pricing/usage
# Go to: https://console.cloud.google.com/billing
```

---

## Part 4: Troubleshooting Guide

### Problem 1: "Cohere API Error: Insufficient credits"

**Cause:** Used all 1,000 free calls

**Solution:**
```python
# Switch to Claude API in story_generator.py
# Use: generate_story_claude(api_key) instead
# Free trial gives $5 credit (~1,500 calls)

# After that, use local Llama 2 via Replicate
# generate_story_llama_local(replicate_api_key)
```

---

### Problem 2: "YouTube authentication failed"

**Cause:** Credentials expired or misconfigured

**Solution:**
```bash
# Delete old credentials
rm youtube_token.pickle

# Re-authenticate
python main.py --auth

# If still fails, check:
# 1. client_secret.json exists
# 2. YouTube Data API enabled in Google Cloud
# 3. Correct project ID set
```

---

### Problem 3: "Video composition fails" / "FFmpeg not found"

**Cause:** FFmpeg not installed or not in PATH

**Solution:**
```bash
# Verify FFmpeg installed
ffmpeg -version

# If not found:
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# Windows: 
# - Download from https://ffmpeg.org/download.html
# - Add to system PATH
# - Restart terminal
```

---

### Problem 4: "Google Cloud TTS returns empty audio"

**Cause:** Text too long or contains unsupported characters

**Solution:**
```python
# In story_generator.py, clean text:
def clean_text_for_tts(text):
    """Remove problematic characters"""
    # Remove special characters
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    text = text.replace('"', '"')
    
    # Limit to 5000 characters (API limit per call)
    if len(text) > 5000:
        text = text[:5000]
    
    return text
```

---

### Problem 5: "Captions out of sync with audio"

**Cause:** Word timing extraction failed

**Solution:**
```python
# In video_generator.py, improve timing:
def extract_word_timings(self, audio_file, text):
    # Use more reliable onset detection
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, n_fft=2048, hop_length=512
    )
    onset_frames = librosa.onset.onset_detect(
        onset_env=onset_env,
        backtrack=True,
        units='frames'
    )
    
    # Smooth timing
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_times = np.convolve(onset_times, [0.25, 0.5, 0.25], mode='same')
    
    return onset_times
```

---

### Problem 6: "YouTube upload throttled"

**Cause:** Too many uploads in short time

**Solution:**
```python
# Add rate limiting in main.py
import time
from random import uniform

def upload_with_backoff(video_file, question):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = upload_to_youtube(video_file, question)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff: 2^attempt seconds + random
                wait_time = (2 ** attempt) + uniform(0, 1)
                print(f"Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

## Part 5: Optimization Strategies

### 5.1 Batch Processing

Generate multiple stories ahead of time to save API calls:

```python
def pre_generate_stories(count=50):
    """Generate batch of stories once per week"""
    stories = generate_batch_stories(count=count)
    
    # Save to JSON
    with open('stories_queue.json', 'w') as f:
        json.dump(stories, f)
    
    print(f"✓ Generated {count} stories for week")

# Schedule once per week
schedule.every().monday.at("00:00").do(pre_generate_stories, count=50)
```

**Benefit:** 
- Use 1,000 Cohere calls efficiently
- Generate 50 stories in bulk
- Process videos one-by-one throughout week

---

### 5.2 Video Caching

Cache background videos locally:

```python
def cache_background_videos(count=20):
    """Download stock videos once, reuse infinitely"""
    
    # Use Pixabay API to find videos
    pixabay_videos = [
        "subway_parkour_01.mp4",
        "minecraft_animation_02.mp4",
        # ... 20 more
    ]
    
    for video_url in pixabay_videos:
        download_video(video_url, f"videos/{Path(video_url).name}")
    
    print(f"✓ Cached {count} background videos")

# One-time setup
cache_background_videos(20)

# Then: use get_background_video() - no API calls needed
```

**Benefit:**
- Zero video generation API costs
- Instant background selection
- 20 unique backgrounds per day

---

### 5.3 Database Tracking

Store metrics for analysis:

```python
import sqlite3

def init_database():
    """Initialize tracking database"""
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY,
        video_id TEXT UNIQUE,
        question TEXT,
        category TEXT,
        created_at TIMESTAMP,
        uploaded_at TIMESTAMP,
        youtube_url TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def log_video(video_id, question, category):
    """Log uploaded video"""
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO videos 
    (video_id, question, category, created_at, uploaded_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (video_id, question, category, datetime.now(), datetime.now()))
    
    conn.commit()
    conn.close()

def get_stats():
    """Get performance stats"""
    conn = sqlite3.connect('analytics.db')
    c = conn.cursor()
    
    c.execute('SELECT COUNT(*) FROM videos')
    total_videos = c.fetchone()[0]
    
    c.execute('SELECT category, COUNT(*) FROM videos GROUP BY category')
    category_counts = c.fetchall()
    
    return {
        'total_videos': total_videos,
        'categories': dict(category_counts)
    }
```

---

## Part 6: Multi-Platform Posting

### 6.1 TikTok Upload

```python
def upload_to_tiktok(video_file, question):
    """Upload same video to TikTok"""
    
    import requests
    
    # Note: TikTok API requires business account & approval
    # This is simplified - actual implementation more complex
    
    with open(video_file, 'rb') as f:
        files = {'video': f}
        data = {
            'caption': question,
            'upload_type': 'video'
        }
        
        response = requests.post(
            'https://open.tiktok.com/v1/post/publish/video/',
            files=files,
            data=data,
            headers={'Authorization': f'Bearer {TIKTOK_ACCESS_TOKEN}'}
        )
    
    return response.json()
```

### 6.2 Instagram Reels

```python
def upload_to_instagram(video_file, question):
    """Upload to Instagram Reels"""
    
    # Similar to TikTok - requires Meta Business Account
    
    from facebook_business.api import FacebookAdsApi
    
    FacebookAdsApi.init(access_token=INSTAGRAM_ACCESS_TOKEN)
    
    # Upload video
    # Implementation depends on Meta Graph API
```

---

## Part 7: Monetization & Analytics

### 7.1 YouTube Partner Program

Requirements for monetization:
- ✅ 1,000 subscribers
- ✅ 4,000 watch hours in last 12 months
- ✅ Original content (not scraped)

**Timeline:** With 30 shorts/day × 100K views each = ~1M views/month
- Typical CPM: $1-5 per 1,000 views
- **Estimated earnings:** $1,000-5,000/month

### 7.2 Track Performance

```python
def get_video_stats(video_id):
    """Get real YouTube performance stats"""
    
    service = build('youtube', 'v3', credentials=credentials)
    
    response = service.videos().list(
        part='statistics',
        id=video_id
    ).execute()
    
    stats = response['items'][0]['statistics']
    
    return {
        'views': int(stats.get('viewCount', 0)),
        'likes': int(stats.get('likeCount', 0)),
        'comments': int(stats.get('commentCount', 0))
    }
```

---

## Part 8: Production Checklist

Before going live with 24/7 automation:

- [ ] All API keys added to config
- [ ] Local test run successful (python main.py --once)
- [ ] Videos uploading to YouTube correctly
- [ ] Captions synced properly with audio
- [ ] Audio quality acceptable
- [ ] Story quality meets expectations
- [ ] Database initialized and tracking working
- [ ] Error logging configured
- [ ] Backup plan if API limits exceeded
- [ ] Cloud deployment tested (if using)
- [ ] 24-hour test run completed without errors
- [ ] YouTube channel SEO optimized

---

## Part 9: Cost Projections

### Zero-Cost Scenario (Recommended)

| Component | Monthly Limit | Videos/Day | Cost |
|-----------|---------------|-----------|------|
| Cohere Stories | 1,000 | 33 | $0 |
| Google TTS | 1M chars | 100+ | $0 |
| YouTube API | Unlimited | Unlimited | $0 |
| Background Videos | Cached locally | Unlimited | $0 |
| Storage | Google free tier | 5GB | $0 |
| **TOTAL** | | **33 videos/day** | **$0/month** |

### Low-Cost Scenario (If maximizing)

| Service | Cost | Use |
|---------|------|-----|
| Claude API | $3 | 1,000 stories |
| Google TTS | $16 | 1M characters |
| Replicate | $20 | Video backgrounds |
| Cloud Run | $30 | 24/7 hosting |
| Storage | $5 | Video files |
| **TOTAL** | **$74/month** | **~500 videos/month** |

### Revenue at Scale (After monetization)

- 500 videos/month × 100K views = **50M views**
- CPM $2 average = **$100K/month** 💰

---

## Final Notes

1. **Start small:** Test with 3 uploads/day first
2. **Monitor quality:** Check YouTube videos weekly
3. **Iterate:** Update prompts based on engagement
4. **Automate completely:** Use Cloud Run for 24/7 operation
5. **Scale up:** Add more platforms (TikTok, Instagram)
6. **Monetize:** Apply for YouTube Partner Program at 1K subs

**Expected Timeline:**
- Week 1: Setup & first uploads
- Week 2-4: Growing audience (100-500 subs)
- Month 2-3: Viral videos, faster growth (500-5K subs)
- Month 4-6: Monetization milestone (1K subs)

Good luck! 🚀