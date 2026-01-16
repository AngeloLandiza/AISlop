# 🎬 YouTube Shorts Automation

**Version:** 1.0.0

Fully automated system for generating and uploading viral YouTube Shorts using AI.

## ✨ Features

- **AI Story Generation** - Uses Cohere API to create emotional, viral stories
- **Text-to-Speech** - gTTS (default) or Piper for voice narration  
- **Auto Captions** - Word-by-word synchronized captions
- **YouTube Upload** - Automatic upload with metadata
- **Scheduling** - Fixed or random daily windows
- **Simple UI** - Streamlit dashboard for setup, config, and actions
- **Cost: $0/month** - All free API tiers

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows: Download from https://ffmpeg.org/download.html
```

### Step 2: Get API Keys & Credentials

You can do this in the UI (recommended) or directly in `config.yaml`.

#### 🔑 Cohere API (Required)
1. Go to [dashboard.cohere.ai/api-keys](https://dashboard.cohere.ai/api-keys)
2. Sign up for free (1,000 calls/month free tier)
3. Copy your API key to `config.yaml`:
```yaml
APIs:
  cohere:
    api_key: "your-cohere-api-key-here"
```

#### 🔑 YouTube API (Required)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create/select a project
3. Enable **YouTube Data API v3**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client IDs**
5. Choose **Desktop application**
6. Download the JSON file
7. **Drag & drop** into the UI (it saves to `credentials/client_secret.json`)
8. Run **Authenticate** in the UI (or `python main.py --auth`)

### Step 3: Add Background Videos

Add some MP4 videos to the `videos/` directory. These will be used as backgrounds:

```bash
# Create videos directory if it doesn't exist
mkdir -p videos

# Add any .mp4 files (Minecraft parkour, Subway Surfers, etc.)
# You can download free videos from:
# - Pexels.com
# - Pixabay.com
# - YouTube (with proper rights)
```

### Step 4: Run Tests

```bash
python main.py --test
```

All tests should pass ✓

### Step 5: Generate Your First Video

```bash
# Generate and upload one video
python main.py --once

# Or run on schedule (3x daily)
python main.py --schedule
```

---

## 📁 Project Structure

```
youtube-shorts-automation/
├── app.py               # ✅ UI Dashboard (Streamlit)
├── config.yaml          # 🔑 Settings & keys
├── credentials/         # 🔐 Drop secrets here
│   ├── client_secret.json
│   └── youtube_token*.pickle
├── main.py              # Main orchestrator
├── story_generator.py   # AI story generation
├── video_generator.py   # Video composition with TTS
├── youtube_uploader.py  # YouTube API integration
├── requirements.txt     # Python dependencies
├── videos/              # Put background videos here
├── output/              # Generated videos appear here
├── logs/                # Automation logs
└── WorkflowDocs/        # Documentation
```

---

## 🔧 Commands

```bash
# Run once
python main.py --once

# Run X times (back-to-back)
python main.py --run 3

# Run on schedule (3x daily at 8am, 2pm, 8pm)
python main.py --schedule

# Authenticate with YouTube
python main.py --auth

# Show current authenticated channel
python main.py --whoami

# Test all components
python main.py --test

# Show statistics
python main.py --stats

# Generate batch of stories for caching
python main.py --batch 20
```

---

## ⚙️ Configuration

Edit `config.yaml` or use the UI to customize:
## 🧭 UI Dashboard (Recommended)

```bash
streamlit run app.py
```

The UI lets you:
- Drag & drop `client_secret.json`
- Set API keys safely
- Edit settings (TTS, captions, schedule)
- Run auth / tests / once-run without the CLI


### API Keys (At the Top!)
```yaml
APIs:
  cohere:
    api_key: "your-key"    # Required
  claude:
    api_key: ""            # Optional fallback
```

### YouTube Settings
```yaml
youtube:
  channel_id: "UC..."      # Your channel ID (recommended)
  privacy_status: "public" # public, private, or unlisted
```

### Schedule
```yaml
schedule:
  times:
    - "08:00"
    - "14:00"
    - "20:00"
```

### TTS Options
```yaml
tts:
  provider: "gtts"         # "gtts" (easy) or "piper" (better quality)
```

---

## 🆘 Troubleshooting

### "FFmpeg not found"
Install FFmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from ffmpeg.org

### "Cohere API key not configured"
1. Get free key at [dashboard.cohere.ai](https://dashboard.cohere.ai/api-keys)
2. Add to `config.yaml` under `APIs.cohere.api_key`

### "client_secret.json not found"
Place it in `credentials/` or use the UI uploader.

### "No background videos"
Add .mp4 files to the `videos/` directory.

### Wrong YouTube channel
1. Delete `credentials/youtube_token*.pickle`
2. Re-run `python main.py --auth`
3. Select correct channel

### OAuth "Access blocked"
Your OAuth app is in **Testing** mode. Add your Google account to **Test Users**
in the OAuth Consent Screen, or publish the app.

---

## 💰 Cost Breakdown

| Component | Provider | Cost | Free Tier |
|-----------|----------|------|-----------|
| Stories | Cohere API | $0 | 1,000/month |
| TTS | gTTS | $0 | Unlimited |
| Video | FFmpeg | $0 | Local |
| Upload | YouTube API | $0 | Unlimited |
| **Total** | | **$0** | |

---

## 📈 Expected Results

- **Month 1**: 90 videos (3/day), building audience
- **Month 2-3**: Algorithm recognition, viral potential
- **Month 4+**: Monetization eligible (1K subs required)

---

## 📄 License

MIT License - Use freely for personal or commercial projects.
