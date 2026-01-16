# YouTube Shorts Automation: Code Templates & Implementation

## Quick-Start Template Files

### File 1: story_generator.py

```python
import cohere
import json
import random
from datetime import datetime

# Initialize Cohere client
co = cohere.ClientV2(api_key="YOUR_COHERE_API_KEY")

STORY_CATEGORIES = [
    "childhood_innocence",
    "accidental_good_deed",
    "second_chance",
    "small_moment_big_impact",
    "human_kindness",
    "regret_lesson"
]

STORY_PROMPT_TEMPLATE = """You are a viral storyteller specializing in emotional, relatable short-form content for YouTube Shorts.
Your task is to create a 1-minute story in the style of r/AskReddit posts.

CATEGORY: {category}

STRUCTURE:
1. An emotionally compelling QUESTION (hook) - this is what gets people clicking
2. A 1-minute STORY with buildup, conflict, and resolution
3. Emotional arc: Setup → Tension → Climax → Resolution

GUIDELINES:
- Genuine and relatable - people can see themselves in the story
- Specific details make stories more believable (names, dates, locations)
- Build mystery/suspense in the first 20 seconds
- Include dialogue when possible
- Resolution should hit hard emotionally (happy or bittersweet, never neutral)
- Aim for 400-500 words (natural speaking pace at 120 wpm = ~60 seconds)

EXAMPLES OF OPENING QUESTIONS:
- "What's a memory from your childhood you thought was innocent, but later came back to haunt you?"
- "Have you ever accidentally saved someone's life?"
- "What moment made you realize your best friend wasn't who you thought they were?"
- "What's the smallest decision that completely changed your life?"

NOW GENERATE A NEW STORY IN THIS JSON FORMAT:
{{
  "question": "[YOUR COMPELLING QUESTION]",
  "story": "[COMPLETE 1-MINUTE STORY - write it as spoken word, natural pauses]",
  "category": "{category}",
  "emotional_beats": [
    {{"time": 0, "emotion": "curiosity"}},
    {{"time": 15, "emotion": "[emotion]"}},
    {{"time": 30, "emotion": "[emotion]"}},
    {{"time": 45, "emotion": "[emotion]"}},
    {{"time": 60, "emotion": "resolution"}}
  ],
  "keywords": ["tag1", "tag2", "tag3"]
}}

Create ONE powerful, viral-worthy story now."""

def generate_story():
    """Generate an emotional story using Cohere API"""
    
    # Random category
    category = random.choice(STORY_CATEGORIES)
    prompt = STORY_PROMPT_TEMPLATE.format(category=category)
    
    try:
        response = co.generate(
            model="command-r-plus",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.85,
            p=0.8
        )
        
        story_text = response.generations[0].text
        
        # Try to parse as JSON
        try:
            story_data = json.loads(story_text)
        except:
            # Fallback: just use the text as story
            story_data = {
                "question": "What changed your life forever?",
                "story": story_text,
                "category": category,
                "keywords": ["story", "emotional", category]
            }
        
        return story_data
        
    except Exception as e:
        print(f"Error generating story: {str(e)}")
        return None

def generate_batch_stories(count=10):
    """Generate multiple stories at once (batch)"""
    stories = []
    for i in range(count):
        story = generate_story()
        if story:
            stories.append(story)
            print(f"✓ Story {i+1}/{count} generated")
    return stories

# ALTERNATIVE: Using Claude API
def generate_story_claude(api_key):
    """Alternative using Claude (free trial with $5 credit)"""
    
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    category = random.choice(STORY_CATEGORIES)
    prompt = STORY_PROMPT_TEMPLATE.format(category=category)
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    story_text = message.content[0].text
    
    try:
        return json.loads(story_text)
    except:
        return {"story": story_text, "question": "What's your story?", "category": category}

# ALTERNATIVE: Local Llama 2 via Replicate
def generate_story_llama_local(api_key):
    """Use local Llama 2 via Replicate (free tier has $25 credits)"""
    
    import replicate
    import os
    
    os.environ["REPLICATE_API_TOKEN"] = api_key
    
    category = random.choice(STORY_CATEGORIES)
    prompt = STORY_PROMPT_TEMPLATE.format(category=category)
    
    output = replicate.run(
        "meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29f8bd9b4f6c9f7c039e2000234cc27940684f",
        input={"prompt": prompt}
    )
    
    story_text = "".join(output)
    
    try:
        return json.loads(story_text)
    except:
        return {"story": story_text, "question": "What's your story?", "category": category}

if __name__ == "__main__":
    # Test story generation
    story = generate_story()
    print(json.dumps(story, indent=2))
```

---

### File 2: video_generator.py

```python
import os
import json
import random
import subprocess
from pathlib import Path
from moviepy.editor import (
    VideoFileClip, TextClip, CompositeVideoClip, 
    AudioFileClip, concatenate_videoclips
)
import librosa
import numpy as np

class VideoGenerator:
    """Complete video composition pipeline"""
    
    def __init__(self, output_dir="./videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def synthesize_speech_google(self, text):
        """Convert text to speech using Google Cloud TTS"""
        
        from google.cloud import texttospeech
        
        client = texttospeech.TextToSpeechClient()
        
        input_text = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-C",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,
            pitch=0.0
        )
        
        response = client.synthesize_speech(
            input=input_text,
            voice=voice,
            audio_config=audio_config
        )
        
        audio_file = self.output_dir / "narration.mp3"
        with open(audio_file, "wb") as out:
            out.write(response.audio_content)
        
        return str(audio_file)
    
    def extract_word_timings(self, audio_file, text):
        """Extract word-level timing from audio"""
        
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050)
        
        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_env=onset_env,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Split text into words
        words = text.split()
        
        # Map words to timing
        word_timings = []
        for i, word in enumerate(words):
            if i < len(onset_times):
                start = float(onset_times[i])
                end = float(onset_times[i+1]) if i+1 < len(onset_times) else start + 0.5
                
                word_timings.append({
                    "word": word,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start
                })
        
        return word_timings
    
    def get_background_video(self):
        """Get random background video from stock sources"""
        
        # Local cached videos (pre-downloaded)
        video_sources = [
            "videos/subway_surfers_01.mp4",
            "videos/minecraft_parkour_01.mp4",
            "videos/action_01.mp4",
        ]
        
        # Filter existing files
        available = [v for v in video_sources if Path(v).exists()]
        
        if available:
            return random.choice(available)
        else:
            # Return placeholder
            return None
    
    def create_caption_clips(self, word_timings):
        """Create synchronized caption clips"""
        
        caption_clips = []
        i = 0
        
        while i < len(word_timings):
            current = word_timings[i]
            
            # Show 1-2 words at a time (70% 1 word, 30% 2 words)
            if random.random() < 0.3 and i + 1 < len(word_timings):
                next_word = word_timings[i + 1]
                caption_text = f"{current['word']} {next_word['word']}"
                start_time = current["start_time"]
                duration = next_word["end_time"] - current["start_time"]
                i += 2
            else:
                caption_text = current["word"]
                start_time = current["start_time"]
                duration = current["duration"]
                i += 1
            
            # Create text clip
            txt_clip = TextClip(
                caption_text,
                fontsize=70,
                color='white',
                method='caption',
                size=(900, 120),
                font='Arial-Bold',
                stroke_color='black',
                stroke_width=3,
                align='center'
            ).set_position(('center', 'center')).set_duration(duration)
            
            txt_clip = txt_clip.set_start(start_time)
            caption_clips.append(txt_clip)
        
        return caption_clips
    
    def compose_video(self, background_video, audio_file, word_timings):
        """Composite all elements into final video"""
        
        # Load background video
        video = VideoFileClip(background_video)
        
        # Ensure 60 seconds
        if video.duration > 60:
            video = video.subclip(0, 60)
        elif video.duration < 60:
            # Loop to fill 60 seconds
            num_loops = int(60 / video.duration) + 1
            video = concatenate_videoclips([video] * num_loops).subclip(0, 60)
        
        # Resize to 1080x1920 (vertical shorts format)
        video = video.resize(height=1920, width=1080)
        
        # Create captions
        caption_clips = self.create_caption_clips(word_timings)
        
        # Load audio
        audio = AudioFileClip(audio_file)
        
        # Composite
        final = CompositeVideoClip([video] + caption_clips)
        final = final.set_audio(audio)
        
        # Export
        output_file = self.output_dir / "final_short.mp4"
        final.write_videofile(
            str(output_file),
            fps=30,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        return str(output_file)
    
    def generate_complete_video(self, story_data):
        """End-to-end video generation"""
        
        # Step 1: Synthesize speech
        print("[1/3] Synthesizing speech...")
        audio_file = self.synthesize_speech_google(story_data["story"])
        
        # Step 2: Extract timing
        print("[2/3] Extracting word timings...")
        word_timings = self.extract_word_timings(audio_file, story_data["story"])
        
        # Step 3: Get background video
        print("[3/3] Getting background video...")
        background = self.get_background_video()
        
        if not background:
            print("⚠ No background video found. Please add videos to /videos directory")
            return None
        
        # Step 4: Compose
        print("[4/3] Composing video...")
        video_file = self.compose_video(background, audio_file, word_timings)
        
        return {
            "video_file": video_file,
            "audio_file": audio_file,
            "word_timings": word_timings,
            "duration": 60,
            "format": "1080x1920"
        }

def download_sample_background_videos():
    """Download sample background videos to use"""
    
    videos = {
        "subway": "https://www.pexels.com/video/[subway-surfers-id]/",
        "minecraft": "https://www.pexels.com/video/[minecraft-parkour-id]/",
    }
    
    # Use Pixabay API to get free videos
    import requests
    
    print("Downloading background videos...")
    
    # Example using Pixabay API
    for name, video_url in videos.items():
        print(f"✓ {name} video ready")

if __name__ == "__main__":
    # Test video generation
    generator = VideoGenerator()
    
    # Sample story
    test_story = {
        "question": "What's your story?",
        "story": "When I was seven, I found my neighbor's dog in the street..."
    }
    
    # Generate video
    result = generator.generate_complete_video(test_story)
    
    if result:
        print(f"\n✓ Video created: {result['video_file']}")
```

---

### File 3: youtube_uploader.py

```python
import pickle
import os
from pathlib import Path
from datetime import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class YouTubeUploader:
    """Handle YouTube API uploads"""
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    
    def __init__(self, credentials_file='client_secret.json', token_file='youtube_token.pickle'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with YouTube API"""
        
        credentials = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            with open(self.token_file, 'rb') as token:
                credentials = pickle.load(token)
        
        # Refresh or create new credentials
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        elif not credentials or not credentials.valid:
            flow = InstalledAppFlow.from_client_secrets_file(
                self.credentials_file,
                self.SCOPES
            )
            credentials = flow.run_local_server(port=8080)
        
        # Save credentials
        with open(self.token_file, 'wb') as token:
            pickle.dump(credentials, token)
        
        self.service = build('youtube', 'v3', credentials=credentials)
    
    def upload_video(self, video_file, title, description, tags=None):
        """Upload video to YouTube"""
        
        if tags is None:
            tags = ['story', 'emotional', 'viral', 'shorts']
        
        # Video metadata
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags,
                'categoryId': '22',  # People & Blogs
                'defaultLanguage': 'en'
            },
            'status': {
                'privacyStatus': 'public',
                'madeForKids': False
            }
        }
        
        # Media upload
        media = MediaFileUpload(
            video_file,
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024
        )
        
        # Create upload request
        request = self.service.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )
        
        # Execute upload
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        
        return {
            'videoId': response['id'],
            'title': title,
            'uploadedAt': response['snippet']['publishedAt']
        }
    
    def schedule_video(self, video_file, title, description, publish_time):
        """Schedule video for future publish"""
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': ['story', 'emotional', 'viral'],
                'categoryId': '22'
            },
            'status': {
                'privacyStatus': 'private',
                'publishAt': publish_time.isoformat() + 'Z',
                'madeForKids': False
            }
        }
        
        media = MediaFileUpload(
            video_file,
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024
        )
        
        request = self.service.videos().insert(
            part='snippet,status',
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
        
        return response['id']

# Usage functions
def upload_to_youtube(video_file, question, channel_id):
    """Convenience function for uploading"""
    
    uploader = YouTubeUploader()
    
    result = uploader.upload_video(
        video_file=video_file,
        title=question,
        description=question,
        tags=['story', 'emotional', 'viral', 'shorts']
    )
    
    return result

if __name__ == "__main__":
    # Setup: Download client_secret.json from Google Cloud Console first!
    print("YouTube Uploader Ready")
    print("1. Download client_secret.json from Google Cloud Console")
    print("2. Place in current directory")
    print("3. Run: uploader = YouTubeUploader()")
```

---

### File 4: main.py (Complete Orchestration)

```python
#!/usr/bin/env python3
"""
Complete YouTube Shorts Automation
Run with: python main.py [--once | --schedule]
"""

import sys
import schedule
import time
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

# Import modules
from story_generator import generate_story
from video_generator import VideoGenerator
from youtube_uploader import upload_to_youtube

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class YouTubeShortsAutomation:
    
    def __init__(self):
        self.video_generator = VideoGenerator()
        Path('logs').mkdir(exist_ok=True)
        logger.info("✓ Automation initialized")
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        
        try:
            # Step 1: Generate story
            logger.info("[1/3] Generating story...")
            story = generate_story()
            
            if not story:
                logger.error("Failed to generate story")
                return
            
            logger.info(f"✓ Story: {story['question'][:50]}...")
            
            # Step 2: Create video
            logger.info("[2/3] Creating video...")
            video_data = self.video_generator.generate_complete_video(story)
            
            if not video_data:
                logger.error("Failed to generate video")
                return
            
            logger.info(f"✓ Video: {video_data['video_file']}")
            
            # Step 3: Upload to YouTube
            logger.info("[3/3] Uploading to YouTube...")
            result = upload_to_youtube(
                video_file=video_data['video_file'],
                question=story['question'],
                channel_id="YOUR_CHANNEL_ID"
            )
            
            logger.info(f"✓ Uploaded: {result['videoId']}")
            
            # Log success
            self.log_upload(story, result, video_data)
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
    
    def log_upload(self, story, upload_result, video_data):
        """Log upload to database"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'video_id': upload_result['videoId'],
            'question': story['question'],
            'category': story.get('category', 'unknown'),
            'video_file': video_data['video_file']
        }
        
        with open('logs/uploads.jsonl', 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def start_schedule(self):
        """Start daily schedule"""
        
        # Schedule times
        times = ['08:00', '14:00', '20:00']
        
        for time_str in times:
            schedule.every().day.at(time_str).do(self.run_pipeline)
            logger.info(f"✓ Scheduled at {time_str}")
        
        logger.info("\n🚀 Scheduler running. Press Ctrl+C to stop.\n")
        
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    
    automation = YouTubeShortsAutomation()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            logger.info("\n▶ Running single pipeline...\n")
            automation.run_pipeline()
        elif sys.argv[1] == "--schedule":
            logger.info("\n▶ Starting scheduler...\n")
            automation.start_schedule()
    else:
        # Default: run once
        automation.run_pipeline()
```

---

## Setup Instructions

### 1. Install Dependencies

```bash
pip install cohere anthropic google-cloud-texttospeech google-auth-oauthlib moviepy librosa requests schedule replicate
```

### 2. Get API Keys

1. **Cohere**: https://dashboard.cohere.io/ (free tier: 1,000 calls/month)
2. **Google Cloud**: https://cloud.google.com/ (free tier: $300 credit + always-free)
3. **YouTube**: https://console.developers.google.com/ (free)

### 3. Configure Files

Create `config.yaml`:
```yaml
APIs:
  cohere:
    api_key: "YOUR_API_KEY"
  google_cloud:
    credentials_file: "google-credentials.json"

schedule:
  times:
    - "08:00"
    - "14:00"
    - "20:00"
```

### 4. Run

```bash
# Test once
python main.py --once

# Run on schedule
python main.py --schedule

# Generate batch of stories
python -c "from story_generator import generate_batch_stories; generate_batch_stories(10)"
```

---

## Database Schema (SQLite)

```sql
CREATE TABLE videos (
  id INTEGER PRIMARY KEY,
  video_id TEXT UNIQUE,
  question TEXT,
  category TEXT,
  created_at TIMESTAMP,
  uploaded_at TIMESTAMP,
  youtube_url TEXT,
  views INTEGER DEFAULT 0,
  likes INTEGER DEFAULT 0,
  comments INTEGER DEFAULT 0
);

CREATE TABLE errors (
  id INTEGER PRIMARY KEY,
  error_message TEXT,
  error_type TEXT,
  created_at TIMESTAMP,
  resolved BOOLEAN DEFAULT FALSE
);
```

---

## Monitoring & Analytics

```python
import sqlite3

def get_channel_stats():
    """Get channel performance stats"""
    
    conn = sqlite3.connect('videos.db')
    c = conn.cursor()
    
    # Total videos
    c.execute('SELECT COUNT(*) FROM videos')
    total = c.fetchone()[0]
    
    # Total views
    c.execute('SELECT SUM(views) FROM videos')
    total_views = c.fetchone()[0] or 0
    
    # Avg engagement
    c.execute('SELECT AVG(likes + comments) FROM videos')
    avg_engagement = c.fetchone()[0] or 0
    
    return {
        'total_videos': total,
        'total_views': total_views,
        'avg_engagement': avg_engagement
    }
```

This provides everything needed to deploy a fully-automated YouTube Shorts system!