#!/usr/bin/env python3
"""
YouTube Shorts Automation - Master Orchestrator
================================================

Complete automation system for generating and uploading viral YouTube Shorts.

Usage:
    python main.py --once       # Generate and upload one video
    python main.py --schedule   # Run on schedule (3x daily)
    python main.py --auth       # Authenticate with YouTube API
    python main.py --test       # Test all components
    python main.py --stats      # Show channel/upload statistics

Features:
- Story generation with Cohere AI
- Video composition with TTS (gTTS or Piper)
- Automated YouTube uploads
- SQLite database for tracking
- Comprehensive logging
- Configurable scheduling

Cost: $0/month with free API tiers
"""

import sys
import os
import argparse
import logging
import json
import sqlite3
import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
from random import uniform
import platform

# Ensure UTF-8 on Windows only (macOS/Linux default to UTF-8)
if platform.system() == "Windows":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from story_generator import generate_story, get_cached_story, generate_batch_stories, test_connection as test_cohere
from video_generator import VideoGenerator, check_ffmpeg, check_tts
from youtube_uploader import YouTubeUploader, upload_to_youtube, whoami

# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).parent
LOGS_DIR = PROJECT_DIR / "logs"
OUTPUT_DIR = PROJECT_DIR / "output"
VIDEOS_DIR = PROJECT_DIR / "videos"
DB_FILE = PROJECT_DIR / "analytics.db"
VERSION_FILE = PROJECT_DIR / "VERSION"

def get_version() -> str:
    try:
        return VERSION_FILE.read_text().strip()
    except Exception:
        return "0.0.0"

# Create directories
LOGS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging to file and console."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler (daily log file)
    log_file = LOGS_DIR / f"automation_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)


# =============================================================================
# Database Functions
# =============================================================================

def init_database():
    """Initialize SQLite database for tracking uploads."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Videos table
    c.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT UNIQUE,
        question TEXT,
        category TEXT,
        video_file TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        uploaded_at TIMESTAMP,
        youtube_url TEXT,
        views INTEGER DEFAULT 0,
        likes INTEGER DEFAULT 0,
        comments INTEGER DEFAULT 0
    )
    ''')
    
    # Errors table
    c.execute('''
    CREATE TABLE IF NOT EXISTS errors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        error_type TEXT,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        resolved BOOLEAN DEFAULT FALSE
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("✓ Database initialized")


def log_video_upload(
    video_id: str,
    question: str,
    category: str,
    video_file: str,
    youtube_url: str
):
    """Log successful video upload to database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO videos (video_id, question, category, video_file, uploaded_at, youtube_url)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (video_id, question, category, video_file, datetime.now(), youtube_url))
    
    conn.commit()
    conn.close()
    
    # Also log to JSON file for easy viewing
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'video_id': video_id,
        'question': question,
        'category': category,
        'youtube_url': youtube_url
    }
    
    uploads_log = LOGS_DIR / "uploads.jsonl"
    with open(uploads_log, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def log_error(error_type: str, error_message: str):
    """Log error to database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
    INSERT INTO errors (error_type, error_message)
    VALUES (?, ?)
    ''', (error_type, error_message))
    
    conn.commit()
    conn.close()


def get_stats() -> Dict[str, Any]:
    """Get upload statistics from database."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Total videos
    c.execute('SELECT COUNT(*) FROM videos')
    total_videos = c.fetchone()[0]
    
    # Videos today
    c.execute('''
        SELECT COUNT(*) FROM videos 
        WHERE DATE(uploaded_at) = DATE('now')
    ''')
    videos_today = c.fetchone()[0]
    
    # Videos this week
    c.execute('''
        SELECT COUNT(*) FROM videos 
        WHERE DATE(uploaded_at) >= DATE('now', '-7 days')
    ''')
    videos_this_week = c.fetchone()[0]
    
    # Errors today
    c.execute('''
        SELECT COUNT(*) FROM errors 
        WHERE DATE(created_at) = DATE('now')
    ''')
    errors_today = c.fetchone()[0]
    
    # Categories
    c.execute('''
        SELECT category, COUNT(*) FROM videos 
        GROUP BY category
    ''')
    categories = dict(c.fetchall())
    
    # Recent uploads
    c.execute('''
        SELECT video_id, question, youtube_url, uploaded_at 
        FROM videos 
        ORDER BY uploaded_at DESC 
        LIMIT 5
    ''')
    recent_uploads = c.fetchall()
    
    conn.close()
    
    return {
        'total_videos': total_videos,
        'videos_today': videos_today,
        'videos_this_week': videos_this_week,
        'errors_today': errors_today,
        'categories': categories,
        'recent_uploads': recent_uploads
    }


# =============================================================================
# Main Automation Class
# =============================================================================

class YouTubeShortsAutomation:
    """
    Complete automation pipeline for YouTube Shorts.
    
    Handles:
    - Story generation (Cohere AI)
    - Video creation (TTS + FFmpeg)
    - YouTube upload
    - Scheduling
    - Error handling
    """
    
    def __init__(self):
        """Initialize automation system."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        init_database()
        
        # Initialize video generator
        self.video_generator = VideoGenerator(
            output_dir=str(OUTPUT_DIR),
            videos_dir=str(VIDEOS_DIR)
        )
        
        # Load configuration
        self.config = self._load_config()
        
        self.logger.info("✓ YouTube Shorts Automation initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        import yaml
        
        config_path = PROJECT_DIR / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def review_content(self, story: Dict[str, Any], video_data: Dict[str, Any]) -> bool:
        """
        Show content for review and ask user if it should be posted.
        
        Args:
            story: Generated story data
            video_data: Generated video data
        
        Returns:
            True if user approves, False otherwise
        """
        print("\n" + "="*60)
        print("📋 CONTENT REVIEW")
        print("="*60)
        
        print("\n📝 STORY QUESTION (Title):")
        print("-" * 40)
        print(story.get('question', 'N/A'))
        
        print("\n📖 STORY CONTENT:")
        print("-" * 40)
        print(story.get('story', 'N/A'))
        
        print("\n🏷️  CATEGORY:", story.get('category', 'N/A'))
        print("🔑 KEYWORDS:", ', '.join(story.get('keywords', [])))
        
        print("\n🎬 VIDEO FILE:")
        print("-" * 40)
        print(f"   {video_data['video_file']}")
        print(f"   Format: {video_data.get('format', '1080x1920')}")
        
        print("\n" + "="*60)
        print("💡 TIP: You can open the video file to preview it")
        print("="*60)
        
        while True:
            response = input("\n🚀 Post this video? [y/n/open]: ").strip().lower()
            
            if response in ['y', 'yes']:
                print("✓ Approved for posting\n")
                return True
            elif response in ['n', 'no']:
                print("✗ Skipped - not posting\n")
                return False
            elif response == 'open':
                # Try to open the video file
                import subprocess
                import platform
                try:
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', video_data['video_file']])
                    elif platform.system() == 'Windows':
                        subprocess.run(['start', video_data['video_file']], shell=True)
                    else:  # Linux
                        subprocess.run(['xdg-open', video_data['video_file']])
                    print("   Opening video...")
                except Exception as e:
                    print(f"   Could not open video: {e}")
            else:
                print("   Please enter 'y' (yes), 'n' (no), or 'open' (preview video)")
    
    def run_pipeline(self, use_cache: bool = True, force_review: bool = None) -> Optional[Dict[str, Any]]:
        """
        Execute complete automation pipeline.
        
        Args:
            use_cache: Whether to use cached stories first
            force_review: Override config auto_post setting (None = use config)
        
        Returns:
            Upload result dictionary or None if pipeline fails
        """
        # Check if we should auto-post or review
        auto_post = self.config.get('automation', {}).get('auto_post', False)
        if force_review is not None:
            auto_post = not force_review  # force_review=True means auto_post=False
        
        self.logger.info("\n" + "="*50)
        self.logger.info("Starting automation pipeline")
        self.logger.info(f"Mode: {'Auto-post' if auto_post else 'Review before posting'}")
        self.logger.info("="*50 + "\n")
        
        try:
            # Step 1: Get story (from cache or generate new)
            self.logger.info("[1/3] Getting story...")
            
            story = None
            if use_cache:
                story = get_cached_story()
            
            if not story:
                story = generate_story()
            
            if not story:
                self.logger.error("✗ Failed to get story")
                log_error("story_generation", "Failed to generate or retrieve story")
                return None
            
            self.logger.info(f"✓ Story: {story.get('question', 'Unknown')[:50]}...")
            
            # Step 2: Create video
            self.logger.info("[2/3] Creating video...")
            video_data = self.video_generator.generate_complete_video(story)
            
            if not video_data:
                self.logger.error("✗ Failed to create video")
                log_error("video_generation", "Failed to generate video")
                return None
            
            self.logger.info(f"✓ Video: {video_data['video_file']}")
            
            # Step 2.5: Review content (if not auto-posting)
            if not auto_post:
                if not self.review_content(story, video_data):
                    self.logger.info("Content skipped by user review")
                    return None
            
            # Step 3: Upload to YouTube
            self.logger.info("[3/3] Uploading to YouTube...")
            
            result = upload_to_youtube(
                video_file=video_data['video_file'],
                question=story['question'],
                channel_id=self.config.get('youtube', {}).get('channel_id', '')
            )
            
            if not result:
                self.logger.error("✗ Failed to upload video")
                log_error("youtube_upload", "Failed to upload to YouTube")
                return None
            
            self.logger.info(f"✓ Uploaded: {result.get('url', result.get('videoId', 'Unknown'))}")
            
            # Log successful upload
            log_video_upload(
                video_id=result['videoId'],
                question=story['question'],
                category=story.get('category', 'unknown'),
                video_file=video_data['video_file'],
                youtube_url=result.get('url', f"https://youtube.com/shorts/{result['videoId']}")
            )
            
            self.logger.info("\n" + "="*50)
            self.logger.info("✓ Pipeline completed successfully!")
            self.logger.info("="*50 + "\n")
            
            return {
                'story': story,
                'video': video_data,
                'upload': result
            }
        
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            log_error("pipeline", str(e))
            return None
    
    def run_with_retry(self, max_retries: int = 3, force_review: bool = None) -> Optional[Dict[str, Any]]:
        """
        Run pipeline with exponential backoff retry.
        
        Args:
            max_retries: Maximum number of retry attempts
            force_review: Override config auto_post setting
        
        Returns:
            Pipeline result or None if all retries fail
        """
        for attempt in range(max_retries):
            result = self.run_pipeline(force_review=force_review)
            
            if result:
                return result
            
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + uniform(0, 1)
                self.logger.info(f"Retrying in {wait_time:.1f}s... (attempt {attempt + 2}/{max_retries})")
                time.sleep(wait_time)
        
        self.logger.error(f"All {max_retries} attempts failed")
        return None
    
    def start_schedule(self):
        """
        Start automated schedule.
        
        Default: 3x daily at 8 AM, 2 PM, 8 PM (configurable in config.yaml)
        """
        schedule_config = self.config.get('schedule', {})
        random_config = schedule_config.get('random', {})
        random_enabled = random_config.get('enabled', False)

        if random_enabled:
            self._schedule_random_times(random_config)
        else:
            self._schedule_fixed_times(schedule_config)
        
        self.logger.info("\n🚀 Scheduler running. Press Ctrl+C to stop.\n")
        
        # Show next run time
        next_run = schedule.next_run()
        if next_run:
            self.logger.info(f"Next run: {next_run}")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.logger.info("\n\nScheduler stopped by user")

    def _schedule_fixed_times(self, schedule_config: Dict[str, Any]):
        """Schedule fixed daily times from config."""
        schedule_times = schedule_config.get('times', ['08:00', '14:00', '20:00'])
        for time_str in schedule_times:
            schedule.every().day.at(time_str).do(self.run_with_retry)
            self.logger.info(f"✓ Scheduled at {time_str}")

    def _schedule_random_times(self, random_config: Dict[str, Any]):
        """Schedule random daily times within configured windows."""
        # Clear existing random jobs
        schedule.clear("random")
        schedule.clear("random_regen")

        windows = random_config.get('windows', [])
        per_window_count = int(random_config.get('per_window_count', 1))

        if not windows:
            self.logger.warning("Random scheduling enabled but no windows configured. Falling back to fixed times.")
            self._schedule_fixed_times(self.config.get('schedule', {}))
            return

        for window in windows:
            start = window.get('start')
            end = window.get('end')
            count = int(window.get('count', per_window_count))
            if not start or not end:
                self.logger.warning(f"Skipping window with missing start/end: {window}")
                continue
            times = self._pick_random_times_in_window(start, end, count)
            for time_str in times:
                schedule.every().day.at(time_str).do(self.run_with_retry).tag("random")
                self.logger.info(f"✓ Random scheduled at {time_str} (window {start}-{end})")

        # Regenerate random schedule daily
        regenerate_at = random_config.get('regenerate_at', '00:05')
        schedule.every().day.at(regenerate_at).do(self._regenerate_random_schedule).tag("random_regen")
        self.logger.info(f"✓ Random schedule will regenerate daily at {regenerate_at}")

    def _regenerate_random_schedule(self):
        """Rebuild random times for the next day."""
        random_config = self.config.get('schedule', {}).get('random', {})
        self._schedule_random_times(random_config)

    def _pick_random_times_in_window(self, start: str, end: str, count: int) -> List[str]:
        """Pick random HH:MM times between start and end."""
        try:
            start_h, start_m = map(int, start.split(":"))
            end_h, end_m = map(int, end.split(":"))
        except Exception:
            self.logger.warning(f"Invalid window time format: {start}-{end}. Expected HH:MM.")
            return []

        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m

        if end_minutes <= start_minutes:
            self.logger.warning(f"Invalid window range (end <= start): {start}-{end}")
            return []

        window_size = end_minutes - start_minutes
        max_unique = max(1, window_size)
        count = max(1, min(count, max_unique))

        chosen = set()
        while len(chosen) < count:
            minute = random.randint(start_minutes, end_minutes - 1)
            chosen.add(minute)

        times = []
        for minute in sorted(chosen):
            h = minute // 60
            m = minute % 60
            times.append(f"{h:02d}:{m:02d}")
        return times


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_once(args):
    """Run pipeline once."""
    automation = YouTubeShortsAutomation()
    
    # Determine review mode from args
    force_review = None
    if hasattr(args, 'review') and args.review:
        force_review = True
    elif hasattr(args, 'auto') and args.auto:
        force_review = False
    
    result = automation.run_with_retry(force_review=force_review)
    return 0 if result else 1


def cmd_run(args):
    """Run pipeline multiple times."""
    automation = YouTubeShortsAutomation()
    
    # Determine review mode from args
    force_review = None
    if hasattr(args, 'review') and args.review:
        force_review = True
    elif hasattr(args, 'auto') and args.auto:
        force_review = False
    
    # Determine run count (CLI overrides config)
    default_count = automation.config.get('automation', {}).get('run_times_default', 1)
    count = args.run if args.run is not None else default_count
    count = max(1, int(count))
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running pipeline {count} time(s)...")
    
    success_count = 0
    for i in range(count):
        logger.info(f"Run {i + 1}/{count}")
        result = automation.run_with_retry(force_review=force_review)
        if result:
            success_count += 1
    
    logger.info(f"Completed {success_count}/{count} successful runs")
    return 0 if success_count > 0 else 1


def cmd_schedule(args):
    """Start scheduled automation."""
    automation = YouTubeShortsAutomation()
    automation.start_schedule()
    return 0


def cmd_auth(args):
    """Authenticate with YouTube."""
    uploader = YouTubeUploader()
    success = uploader.authenticate()
    
    if success:
        print("\n✓ YouTube authentication successful!")
        channel = uploader.get_channel_info()
        if channel:
            print(f"\nChannel ID:   {channel['id']}")
            print(f"Channel Name: {channel['title']}")
            print(f"Subscribers:  {channel['subscriberCount']}")
            print(f"\nToken saved for this channel.")
        return 0
    else:
        print("\n✗ Authentication failed (possibly wrong channel selected)")
        return 1


def cmd_whoami(args):
    """Show currently authenticated channel."""
    channel = whoami()
    
    if channel:
        print("\n" + "="*50)
        print("Currently Authenticated Channel")
        print("="*50)
        print(f"\nChannel ID:   {channel['id']}")
        print(f"Channel Name: {channel['title']}")
        print(f"Subscribers:  {channel['subscriberCount']}")
        print(f"Videos:       {channel['videoCount']}")
        print(f"Total Views:  {channel['viewCount']}")
        print("\n" + "="*50 + "\n")
        return 0
    else:
        print("\n✗ Not authenticated")
        print("Run: python main.py --auth")
        return 1


def cmd_test(args):
    """Test all components."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*50)
    print("Component Tests")
    print("="*50 + "\n")
    
    all_passed = True
    
    # Test 1: FFmpeg
    print("1. Testing FFmpeg...")
    if check_ffmpeg():
        print("   ✓ FFmpeg installed\n")
    else:
        print("   ✗ FFmpeg not found\n")
        all_passed = False
    
    # Test 2: Cohere API
    print("2. Testing Cohere API...")
    if test_cohere():
        print("   ✓ Cohere API working\n")
    else:
        print("   ✗ Cohere API failed\n")
        all_passed = False
    
    # Test 3: TTS (gTTS or Piper)
    print("3. Testing TTS (gTTS/Piper)...")
    if check_tts():
        print("   ✓ TTS available\n")
    else:
        print("   ✗ TTS not available")
        print("   Install gTTS: pip install gtts\n")
        all_passed = False
    
    # Test 4: YouTube credentials
    print("4. Testing YouTube credentials...")
    uploader = YouTubeUploader()
    if uploader.is_authenticated():
        print("   ✓ YouTube authenticated\n")
        channel = uploader.get_channel_info()
        if channel:
            print(f"   Channel: {channel['title']}\n")
    else:
        client_secret = PROJECT_DIR / "client_secret.json"
        if client_secret.exists():
            print("   ⚠ client_secret.json found but not authenticated")
            print("   Run: python main.py --auth\n")
        else:
            print("   ✗ client_secret.json not found\n")
            all_passed = False
    
    # Test 5: Background videos
    print("5. Testing background videos...")
    video_generator = VideoGenerator()
    bg = video_generator.get_background_video()
    if bg:
        print(f"   ✓ Found background video: {Path(bg).name}\n")
    else:
        print("   ⚠ No background videos in videos/ directory\n")
    
    # Test 6: Database
    print("6. Testing database...")
    init_database()
    if DB_FILE.exists():
        print("   ✓ Database initialized\n")
    else:
        print("   ✗ Database initialization failed\n")
        all_passed = False
    
    print("="*50)
    if all_passed:
        print("All critical tests passed! ✓")
    else:
        print("Some tests failed. See above for details.")
    print("="*50 + "\n")
    
    return 0 if all_passed else 1


def cmd_stats(args):
    """Show statistics."""
    init_database()
    stats = get_stats()
    
    print("\n" + "="*50)
    print("📊 Upload Statistics")
    print("="*50)
    
    print(f"\nTotal Videos: {stats['total_videos']}")
    print(f"Videos Today: {stats['videos_today']}")
    print(f"Videos This Week: {stats['videos_this_week']}")
    print(f"Errors Today: {stats['errors_today']}")
    
    if stats['categories']:
        print("\nBy Category:")
        for cat, count in stats['categories'].items():
            print(f"  - {cat}: {count}")
    
    if stats['recent_uploads']:
        print("\nRecent Uploads:")
        for video_id, question, url, uploaded_at in stats['recent_uploads']:
            print(f"  - {question[:40]}...")
            print(f"    {url}")
    
    print("\n" + "="*50 + "\n")
    return 0


def cmd_batch(args):
    """Generate batch of stories for caching."""
    count = args.count or 10
    
    print(f"\nGenerating {count} stories...")
    stories = generate_batch_stories(count=count, save_to_file=True)
    
    print(f"\n✓ Generated {len(stories)} stories")
    print(f"Saved to: stories_queue.json\n")
    
    return 0


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f"YouTube Shorts Automation System (v{get_version()})",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --once       Generate and upload one video
  python main.py --schedule   Run on schedule (3x daily)
  python main.py --auth       Authenticate with YouTube
  python main.py --whoami     Show authenticated channel ID
  python main.py --test       Test all components
  python main.py --stats      Show statistics
  python main.py --batch 20   Generate 20 stories for cache
  python main.py --run 3      Run pipeline 3 times (back-to-back)

Multi-Channel: Set youtube.channel_id in config.yaml to lock to a specific channel.

Cost: $0/month with free API tiers (Cohere, Google Cloud, YouTube)
        """
    )
    
    parser.add_argument('--once', action='store_true', help='Run pipeline once')
    parser.add_argument('--schedule', action='store_true', help='Run on schedule')
    parser.add_argument('--auth', action='store_true', help='Authenticate with YouTube')
    parser.add_argument('--whoami', action='store_true', help='Show authenticated channel ID')
    parser.add_argument('--test', action='store_true', help='Test all components')
    parser.add_argument('--stats', action='store_true', help='Show statistics')
    parser.add_argument('--batch', type=int, metavar='N', help='Generate N stories for cache')
    parser.add_argument('--review', action='store_true', help='Force review mode (preview before posting)')
    parser.add_argument('--auto', action='store_true', help='Force auto-post mode (no review)')
    parser.add_argument('--run', type=int, metavar='N', help='Run pipeline N times back-to-back')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle commands
    if args.auth:
        return cmd_auth(args)
    elif args.whoami:
        return cmd_whoami(args)
    elif args.test:
        return cmd_test(args)
    elif args.stats:
        return cmd_stats(args)
    elif args.batch:
        args.count = args.batch
        return cmd_batch(args)
    elif args.schedule:
        return cmd_schedule(args)
    elif args.run is not None:
        return cmd_run(args)
    elif args.once:
        return cmd_once(args)
    else:
        # Default: run once
        logger.info("No command specified. Running once...")
        return cmd_once(args)


if __name__ == "__main__":
    sys.exit(main())
