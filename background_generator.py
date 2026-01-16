#!/usr/bin/env python3
"""
Background Video Generator for YouTube Shorts Automation
=========================================================

Generates fresh AI background videos for each short using:
- Pollinations.ai (FREE - no API key required)
- FFmpeg for image-to-video conversion with motion effects

Also supports downloading real gameplay videos from YouTube:
- Searches for similar no-copyright gameplay videos
- Downloads from multiple sources
- Tracks parsed videos to avoid duplicates

Supports styles:
- Minecraft parkour
- Subway Surfers
- Abstract/neon
- Nature/satisfying

Cost: $0 (completely free)
"""

import os
import random
import logging
import subprocess
import requests
import urllib.parse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Project directory
PROJECT_DIR = Path(__file__).parent

# File to track parsed/downloaded videos (to avoid duplicates)
PARSED_VIDEOS_FILE = PROJECT_DIR / "parsed_videos.json"

# =============================================================================
# BACKGROUND STYLE PROMPTS
# =============================================================================

BACKGROUND_STYLES = {
    "minecraft_parkour": [
        # First-person POV running on grass blocks, trees, blue sky - like actual Minecraft parkour gameplay
        "first person view running on minecraft grass block pathway, green grass blocks forming path, oak trees with orange autumn leaves on sides, bright blue sky with white clouds, blocky voxel 3D style, Minecraft game screenshot aesthetic, depth perspective, gaming POV",
        "minecraft parkour first person perspective, running on green grass block trail, colorful trees with fall leaves, water visible below, bright daylight, blocky pixel art 3D game style, motion blur effect",
        "POV minecraft gameplay running forward on grass blocks, oak and birch trees with autumn foliage, clear blue sky, classic Minecraft texture style, parkour map aesthetic, vibrant green and orange colors",
        "first person minecraft view, grass block pathway stretching ahead, trees with red and orange leaves, blue sky background, blocky voxel game graphics, parkour course perspective",
        "minecraft first person running on floating grass platforms, autumn colored trees, bright sunny sky, classic blocky game aesthetic, parkour speedrun view, green grass blocks",
    ],
    "subway_surfers": [
        # Third-person behind character, train tracks, blue barriers, coins, cartoon style - like actual Subway Surfers
        "third person view endless runner game, character running on train tracks, blue metal barriers on sides, floating gold coins, yellow train in background, bright cartoon 3D style, Subway Surfers aesthetic, green trees alongside tracks",
        "cartoon endless runner gameplay, running on railway tracks, blue guardrails, collectible coins floating, colorful urban background, mobile game 3D graphics style, vibrant saturated colors",
        "behind character running on subway tracks, blue barriers, gold coins and power-ups floating, yellow train approaching, cartoon city background with trees, bright colorful mobile game style",
        "endless runner third person view, train tracks perspective, blue safety barriers, floating golden coins, cartoon 3D graphics, urban environment with vegetation, Subway Surfers game look",
        "mobile game runner screenshot, character on train tracks from behind, blue railings, collectible items floating, yellow subway train, cartoon style bright colors, city park background",
    ],
    "satisfying_abstract": [
        "satisfying abstract fluid art, colorful liquid flowing, smooth gradients, mesmerizing patterns, ASMR visuals",
        "neon geometric shapes morphing, abstract digital art, glowing lines, dark background, hypnotic movement",
        "colorful paint mixing in slow motion, abstract satisfying visuals, vibrant swirls, calming aesthetic",
        "abstract holographic waves, iridescent colors, smooth flowing shapes, digital art, relaxing visuals",
        "kinetic sand satisfying texture, colorful patterns, abstract close-up, smooth aesthetic, calming",
    ],
    "nature_calming": [
        "aerial view of ocean waves on beach, turquoise water, white sand, calming nature, 4K cinematic",
        "forest stream with sunlight filtering through trees, peaceful nature scene, green aesthetic, serene",
        "clouds moving across sunset sky, orange and purple gradient, time-lapse style, peaceful atmosphere",
        "waterfall in tropical jungle, mist and rainbows, lush green plants, nature documentary style",
        "northern lights aurora borealis, night sky stars, snowy mountains, magical atmosphere, ethereal",
    ],
    "gaming_aesthetic": [
        "retro arcade game background, neon pixel art, glowing grid lines, synthwave aesthetic, 80s style",
        "futuristic gaming setup, RGB lights, dark room with neon glow, cyberpunk aesthetic, tech vibes",
        "video game loading screen aesthetic, abstract shapes, glowing particles, dark background, modern",
        "esports arena background, dramatic lighting, gaming tournament stage, professional competitive",
        "virtual reality cyberspace, digital grid, neon blue and purple, matrix style, sci-fi gaming",
    ],
}

# Default styles to use if none specified
DEFAULT_STYLES = ["minecraft_parkour", "subway_surfers", "satisfying_abstract"]


# =============================================================================
# SEARCH QUERIES FOR FINDING SIMILAR VIDEOS
# =============================================================================

VIDEO_SEARCH_QUERIES = {
    "minecraft_parkour": [
        "minecraft parkour gameplay no copyright",
        "minecraft parkour free to use",
        "minecraft parkour background video no copyright",
        "minecraft first person parkour royalty free",
        "minecraft parkour POV no copyright music",
        "minecraft parkour 4K free use",
        "minecraft speedrun parkour no copyright",
        "minecraft parkour gameplay footage free",
    ],
    "subway_surfers": [
        "subway surfers gameplay no copyright",
        "subway surfers free to use background",
        "subway surfers 4K gameplay royalty free",
        "subway surfers no copyright gameplay",
        "subway surfers background video free",
        "subway surfers endless run no copyright",
        "subway surfers gameplay footage free use",
    ],
    "satisfying": [
        "satisfying video no copyright",
        "satisfying compilation free to use",
        "ASMR satisfying background free",
        "satisfying slime no copyright",
        "oddly satisfying royalty free",
    ],
    "gaming_general": [
        "gaming background video no copyright",
        "mobile game gameplay free to use",
        "endless runner gameplay no copyright",
        "gaming footage royalty free",
    ],
}


# =============================================================================
# PARSED VIDEO TRACKING
# =============================================================================

def load_parsed_videos() -> Dict[str, List[str]]:
    """
    Load the list of already parsed/downloaded video IDs.
    
    Returns:
        Dictionary mapping style -> list of video IDs
    """
    if PARSED_VIDEOS_FILE.exists():
        try:
            with open(PARSED_VIDEOS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_parsed_videos(parsed: Dict[str, List[str]]):
    """
    Save the list of parsed video IDs to file.
    
    Args:
        parsed: Dictionary mapping style -> list of video IDs
    """
    with open(PARSED_VIDEOS_FILE, 'w') as f:
        json.dump(parsed, f, indent=2)


def add_parsed_video(style: str, video_id: str):
    """
    Add a video ID to the parsed list.
    
    Args:
        style: The video style category
        video_id: YouTube video ID
    """
    parsed = load_parsed_videos()
    if style not in parsed:
        parsed[style] = []
    if video_id not in parsed[style]:
        parsed[style].append(video_id)
    save_parsed_videos(parsed)


def is_video_parsed(video_id: str) -> bool:
    """
    Check if a video has already been parsed.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        True if already parsed, False otherwise
    """
    parsed = load_parsed_videos()
    for style_videos in parsed.values():
        if video_id in style_videos:
            return True
    return False


def get_parsed_count(style: str = None) -> int:
    """
    Get count of parsed videos.
    
    Args:
        style: Specific style or None for total
    
    Returns:
        Number of parsed videos
    """
    parsed = load_parsed_videos()
    if style:
        return len(parsed.get(style, []))
    return sum(len(v) for v in parsed.values())


def clear_parsed_videos(style: str = None):
    """
    Clear parsed videos tracking.
    
    Args:
        style: Specific style to clear, or None for all
    """
    if style:
        parsed = load_parsed_videos()
        if style in parsed:
            del parsed[style]
        save_parsed_videos(parsed)
    else:
        save_parsed_videos({})
    print(f"✓ Cleared parsed videos tracking" + (f" for {style}" if style else ""))


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    import yaml
    
    config_path = PROJECT_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_random_prompt(style: str = None) -> tuple[str, str]:
    """
    Get a random background prompt.
    
    Args:
        style: Specific style to use, or None for random from defaults
    
    Returns:
        Tuple of (style_name, prompt)
    """
    config = load_config()
    enabled_styles = config.get('background', {}).get('styles', DEFAULT_STYLES)
    
    if style and style in BACKGROUND_STYLES:
        selected_style = style
    else:
        # Filter to only enabled styles that exist
        valid_styles = [s for s in enabled_styles if s in BACKGROUND_STYLES]
        if not valid_styles:
            valid_styles = DEFAULT_STYLES
        selected_style = random.choice(valid_styles)
    
    prompt = random.choice(BACKGROUND_STYLES[selected_style])
    return selected_style, prompt


def generate_image_pollinations(
    prompt: str,
    output_path: str,
    width: int = 1080,
    height: int = 1920
) -> Optional[str]:
    """
    Generate an image using Pollinations.ai (FREE, no API key).
    
    Args:
        prompt: Text prompt for image generation
        output_path: Where to save the image
        width: Image width
        height: Image height
    
    Returns:
        Path to saved image, or None if failed
    """
    try:
        # Pollinations.ai free API - just encode prompt in URL
        encoded_prompt = urllib.parse.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
        
        logger.info(f"Generating background image...")
        logger.debug(f"Prompt: {prompt[:50]}...")
        
        # Download the generated image
        response = requests.get(url, timeout=120)  # May take a while
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✓ Image saved: {output_path}")
            return output_path
        else:
            logger.error(f"Pollinations API error: {response.status_code}")
            return None
    
    except requests.Timeout:
        logger.error("Image generation timed out (>120s)")
        return None
    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        return None


def image_to_video_kenburns(
    image_path: str,
    output_path: str,
    duration: int = 65,
    effect: str = "zoom_in"
) -> Optional[str]:
    """
    Convert a static image to video with Ken Burns effect (pan/zoom).
    
    Args:
        image_path: Path to source image
        output_path: Where to save the video
        duration: Video duration in seconds
        effect: Effect type - "zoom_in", "zoom_out", "pan_left", "pan_right", "random"
    
    Returns:
        Path to video, or None if failed
    """
    try:
        # Check FFmpeg
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found!")
        return None
    
    # Choose random effect if specified
    if effect == "random":
        effect = random.choice(["zoom_in", "zoom_out", "pan_up", "pan_down"])
    
    # Define zoom/pan filter based on effect
    # These create smooth motion over the image
    effects = {
        "zoom_in": f"scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*30}:s=1080x1920:fps=30",
        "zoom_out": f"scale=8000:-1,zoompan=z='if(lte(zoom,1.0),1.5,max(1.001,zoom-0.0015))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={duration*30}:s=1080x1920:fps=30",
        "pan_up": f"scale=1080:3840,zoompan=z='1':x='0':y='ih-oh-((ih-oh)/{duration*30})*on':d={duration*30}:s=1080x1920:fps=30",
        "pan_down": f"scale=1080:3840,zoompan=z='1':x='0':y='((ih-oh)/{duration*30})*on':d={duration*30}:s=1080x1920:fps=30",
    }
    
    filter_str = effects.get(effect, effects["zoom_in"])
    
    logger.info(f"Converting image to video with {effect} effect...")
    
    cmd = [
        'ffmpeg', '-y',
        '-loop', '1',
        '-i', image_path,
        '-vf', filter_str,
        '-c:v', 'libx264',
        '-t', str(duration),
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0 and Path(output_path).exists():
        logger.info(f"✓ Video created: {output_path}")
        return output_path
    else:
        logger.error(f"FFmpeg error: {result.stderr}")
        return None


def generate_background_video(
    style: str = None,
    output_dir: str = None,
    duration: int = 65
) -> Optional[Dict[str, Any]]:
    """
    Generate a complete background video.
    
    Args:
        style: Background style (minecraft_parkour, subway_surfers, etc.)
        output_dir: Where to save files
        duration: Video duration in seconds
    
    Returns:
        Dictionary with video info, or None if failed
    """
    if output_dir is None:
        output_dir = PROJECT_DIR / "videos" / "generated"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get prompt
    selected_style, prompt = get_random_prompt(style)
    
    logger.info(f"Generating {selected_style} background...")
    
    # Step 1: Generate image
    image_path = str(output_dir / f"bg_image_{timestamp}.png")
    image_result = generate_image_pollinations(
        prompt=prompt,
        output_path=image_path,
        width=1080,
        height=1920
    )
    
    if not image_result:
        logger.error("Failed to generate image")
        return None
    
    # Step 2: Convert to video with motion
    video_path = str(output_dir / f"bg_video_{timestamp}.mp4")
    effect = random.choice(["zoom_in", "zoom_out", "pan_up", "pan_down"])
    
    video_result = image_to_video_kenburns(
        image_path=image_path,
        output_path=video_path,
        duration=duration,
        effect=effect
    )
    
    if not video_result:
        logger.error("Failed to convert to video")
        return None
    
    # Optionally delete the image to save space
    # Path(image_path).unlink()
    
    return {
        "video_path": video_path,
        "image_path": image_path,
        "style": selected_style,
        "prompt": prompt,
        "effect": effect,
        "duration": duration,
        "created_at": datetime.now().isoformat()
    }


def get_or_generate_background(
    style: str = None,
    force_new: bool = False
) -> Optional[str]:
    """
    Get a background video - generate new or use existing.
    
    This is the main function called by the video pipeline.
    
    Args:
        style: Preferred style (or None for random)
        force_new: Always generate new (True) or allow reuse (False)
    
    Returns:
        Path to background video file
    """
    config = load_config()
    bg_config = config.get('background', {})
    
    # Check if background generation is enabled
    if not bg_config.get('auto_generate', True):
        # Use static backgrounds from videos/ folder
        logger.info("Auto-generate disabled, using static backgrounds")
        return None
    
    # Check for existing generated backgrounds
    generated_dir = PROJECT_DIR / "videos" / "generated"
    if not force_new and generated_dir.exists():
        existing = list(generated_dir.glob("bg_video_*.mp4"))
        
        # Reuse chance (to save API calls)
        reuse_chance = bg_config.get('reuse_chance', 0.3)
        if existing and random.random() < reuse_chance:
            selected = random.choice(existing)
            logger.info(f"Reusing existing background: {selected.name}")
            return str(selected)
    
    # Generate new background
    result = generate_background_video(style=style)
    
    if result:
        return result["video_path"]
    
    return None


def list_available_styles() -> List[str]:
    """List all available background styles."""
    return list(BACKGROUND_STYLES.keys())


def preview_prompts(style: str = None):
    """Print sample prompts for a style."""
    if style:
        styles = [style] if style in BACKGROUND_STYLES else []
    else:
        styles = BACKGROUND_STYLES.keys()
    
    for s in styles:
        print(f"\n🎨 {s.upper().replace('_', ' ')}")
        print("-" * 40)
        for i, prompt in enumerate(BACKGROUND_STYLES[s][:3], 1):
            print(f"  {i}. {prompt[:80]}...")


# =============================================================================
# RECOMMENDED: Download Real Gameplay Videos (Better Quality)
# =============================================================================

RECOMMENDED_GAMEPLAY_SOURCES = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  📥 RECOMMENDED: Download Real No-Copyright Gameplay Videos                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  AI-generated backgrounds work, but REAL gameplay footage looks better!      ║
║  These YouTube videos are FREE TO USE (No Copyright):                        ║
║                                                                              ║
║  🎮 MINECRAFT PARKOUR:                                                       ║
║     https://www.youtube.com/watch?v=_-2ZUciZgls                              ║
║     (20+ minutes of first-person parkour gameplay)                           ║
║                                                                              ║
║  🏃 SUBWAY SURFERS:                                                          ║
║     https://www.youtube.com/watch?v=i0M4ARe9v0Y                              ║
║     (Official channel, no copyright, 4K quality)                             ║
║                                                                              ║
║  HOW TO DOWNLOAD & USE:                                                      ║
║  1. Use yt-dlp to download: yt-dlp -f best <URL>                            ║
║  2. Cut into 60-90 second clips using FFmpeg:                               ║
║     ffmpeg -i video.mp4 -ss 00:00:00 -t 00:01:30 -c copy clip1.mp4         ║
║  3. Place clips in the videos/ folder                                        ║
║  4. Set background.auto_generate: false in config.yaml                       ║
║                                                                              ║
║  The system will randomly pick from your clips for each video!               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def search_similar_videos(
    style: str = "minecraft_parkour",
    num_results: int = 10,
    exclude_parsed: bool = True
) -> List[Dict[str, Any]]:
    """
    Search YouTube for similar no-copyright gameplay videos.
    
    Args:
        style: Video style to search for
        num_results: Number of results to return
        exclude_parsed: Whether to exclude already-parsed videos
    
    Returns:
        List of video info dictionaries
    """
    # Get search queries for this style
    queries = VIDEO_SEARCH_QUERIES.get(style, VIDEO_SEARCH_QUERIES.get("gaming_general", []))
    if not queries:
        print(f"❌ Unknown style: {style}")
        return []
    
    # Pick a random query to add variety
    query = random.choice(queries)
    
    print(f"🔍 Searching for: {query}")
    
    # Use yt-dlp to search YouTube
    cmd = [
        'yt-dlp',
        '--flat-playlist',
        '--print', '%(id)s|%(title)s|%(duration)s|%(channel)s',
        f'ytsearch{num_results * 3}:{query}'  # Search more to account for filtering
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"❌ Search failed: {result.stderr[:100]}")
            return []
    except subprocess.TimeoutExpired:
        print("❌ Search timed out")
        return []
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if not line or '|' not in line:
            continue
        
        parts = line.split('|')
        if len(parts) < 4:
            continue
        
        video_id, title, duration_str, channel = parts[0], parts[1], parts[2], parts[3]
        
        # Skip already-parsed videos
        if exclude_parsed and is_video_parsed(video_id):
            continue
        
        # Parse duration (may be in seconds or "NA")
        try:
            duration = int(duration_str) if duration_str and duration_str != 'NA' else 0
        except ValueError:
            duration = 0
        
        # Filter for appropriate videos
        # - Prefer longer videos (more clips possible)
        # - Skip very short videos (< 60 seconds)
        if duration > 0 and duration < 60:
            continue
        
        videos.append({
            'id': video_id,
            'url': f'https://www.youtube.com/watch?v={video_id}',
            'title': title,
            'duration': duration,
            'channel': channel,
            'style': style
        })
        
        if len(videos) >= num_results:
            break
    
    return videos


def download_gameplay_videos(
    output_dir: str = None,
    style: str = "minecraft_parkour",
    num_videos: int = 3,
    clips_per_video: int = 5,
    clip_duration: int = 90,
    search_new: bool = True
):
    """
    Download no-copyright gameplay videos using yt-dlp.
    
    Searches for similar videos and downloads from multiple sources,
    then automatically splits them into clips.
    
    Args:
        output_dir: Where to save videos (default: videos/)
        style: Video style (minecraft_parkour, subway_surfers, etc.)
        num_videos: Number of source videos to download
        clips_per_video: Number of clips to create from each video
        clip_duration: Duration of each clip in seconds
        search_new: Search for new videos (True) or use recommended URLs (False)
    
    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = PROJECT_DIR / "videos"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n❌ yt-dlp not installed!")
        print("   Install with: pip install yt-dlp")
        print("   Or: brew install yt-dlp")
        return False
    
    total_clips_target = num_videos * clips_per_video
    
    print("\n" + "="*60)
    print(f"📥 Background Video Downloader")
    print("="*60)
    print(f"   Style:           {style}")
    print(f"   Source videos:   {num_videos}")
    print(f"   Clips per video: {clips_per_video}")
    print(f"   Clip duration:   {clip_duration}s")
    print(f"   Total clips:     ~{total_clips_target}")
    print(f"   Already parsed:  {get_parsed_count(style)} videos")
    print("="*60)
    
    if search_new:
        # Search for similar videos
        videos = search_similar_videos(
            style=style,
            num_results=num_videos,
            exclude_parsed=True
        )
        
        if not videos:
            print("\n⚠️  No new videos found! Trying fallback URLs...")
            search_new = False
    
    if not search_new:
        # Use fallback recommended videos
        fallback_videos = {
            "minecraft_parkour": [
                {"id": "_-2ZUciZgls", "url": "https://www.youtube.com/watch?v=_-2ZUciZgls", 
                 "title": "Minecraft Parkour Gameplay NO COPYRIGHT", "style": "minecraft_parkour"},
            ],
            "subway_surfers": [
                {"id": "i0M4ARe9v0Y", "url": "https://www.youtube.com/watch?v=i0M4ARe9v0Y",
                 "title": "Subway Surfers (2026) - Gameplay [4K] No Copyright", "style": "subway_surfers"},
            ]
        }
        videos = fallback_videos.get(style, fallback_videos.get("minecraft_parkour", []))
        # Filter out already parsed
        videos = [v for v in videos if not is_video_parsed(v['id'])]
        
        if not videos:
            print("\n⚠️  All known videos have been parsed!")
            print("   Try clearing parsed history: --clear-parsed")
            print("   Or search for new videos will find fresh content next time.")
            return False
    
    print(f"\n📋 Found {len(videos)} videos to download:\n")
    for i, v in enumerate(videos, 1):
        print(f"   {i}. {v['title'][:50]}...")
    
    clips_created = 0
    videos_processed = 0
    
    for video in videos:
        if videos_processed >= num_videos:
            break
        
        video_id = video['id']
        video_url = video['url']
        video_title = video.get('title', f'video_{video_id}')
        
        # Create a safe filename from title
        safe_title = "".join(c if c.isalnum() or c in ' -_' else '' for c in video_title)[:30]
        safe_title = safe_title.strip().replace(' ', '_') or video_id
        
        output_file = output_dir / f"{style}_{safe_title}_full.mp4"
        
        print(f"\n{'='*50}")
        print(f"⏳ [{videos_processed + 1}/{num_videos}] Downloading: {video_title[:40]}...")
        print(f"   URL: {video_url}")
        
        # Download the video
        cmd = [
            'yt-dlp',
            '-f', 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
            video_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0 or not output_file.exists():
            print(f"   ❌ Download failed, skipping...")
            continue
        
        print(f"   ✓ Downloaded: {output_file.name}")
        
        # Mark as parsed
        add_parsed_video(style, video_id)
        videos_processed += 1
        
        # Split into clips
        print(f"   ✂️  Splitting into {clips_per_video} clips...")
        split_video_into_clips(
            input_video=str(output_file),
            output_dir=str(output_dir),
            clip_duration=clip_duration,
            num_clips=clips_per_video
        )
        clips_created += clips_per_video
        
        # Optionally delete the full video to save space
        # output_file.unlink()
    
    print("\n" + "="*60)
    print(f"✅ DOWNLOAD COMPLETE")
    print("="*60)
    print(f"   Videos processed: {videos_processed}")
    print(f"   Clips created:    ~{clips_created}")
    print(f"   Total parsed:     {get_parsed_count(style)} videos")
    print(f"   Output directory: {output_dir}")
    print("="*60 + "\n")
    
    return True


def download_multiple_styles(
    styles: List[str] = None,
    num_videos_per_style: int = 2,
    clips_per_video: int = 5,
    clip_duration: int = 90
):
    """
    Download videos for multiple styles.
    
    Args:
        styles: List of styles to download (default: minecraft_parkour, subway_surfers)
        num_videos_per_style: Videos to download per style
        clips_per_video: Clips to create per video
        clip_duration: Duration of each clip
    """
    if styles is None:
        styles = ["minecraft_parkour", "subway_surfers"]
    
    print("\n" + "="*60)
    print("📥 BATCH DOWNLOAD - Multiple Styles")
    print("="*60)
    print(f"   Styles: {', '.join(styles)}")
    print(f"   Videos per style: {num_videos_per_style}")
    print(f"   Total target clips: ~{len(styles) * num_videos_per_style * clips_per_video}")
    print("="*60)
    
    for style in styles:
        print(f"\n🎮 Downloading {style} videos...")
        download_gameplay_videos(
            style=style,
            num_videos=num_videos_per_style,
            clips_per_video=clips_per_video,
            clip_duration=clip_duration
        )


def split_video_into_clips(
    input_video: str,
    output_dir: str = None,
    clip_duration: int = 90,
    num_clips: int = 5
):
    """
    Split a long video into multiple short clips.
    
    Args:
        input_video: Path to input video
        output_dir: Where to save clips
        clip_duration: Duration of each clip in seconds
        num_clips: Number of clips to create
    """
    input_path = Path(input_video)
    if not input_path.exists():
        print(f"❌ Video not found: {input_video}")
        return
    
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
    
    # Get video duration using ffprobe
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(input_path)
        ], capture_output=True, text=True)
        total_duration = float(result.stdout.strip())
    except:
        total_duration = 1200  # Default 20 minutes if can't detect
    
    base_name = input_path.stem.replace('_full', '')
    
    print(f"\n✂️  Splitting {input_path.name} into {num_clips} clips...")
    
    # Calculate start times spread across the video
    usable_duration = total_duration - clip_duration
    interval = usable_duration / num_clips if num_clips > 1 else 0
    
    for i in range(num_clips):
        start_time = int(i * interval)
        hours = start_time // 3600
        minutes = (start_time % 3600) // 60
        seconds = start_time % 60
        start_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        output_file = output_dir / f"{base_name}_clip{i+1}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', start_str,
            '-i', str(input_path),
            '-t', str(clip_duration),
            '-c', 'copy',
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✓ Created: {output_file.name} (starts at {start_str})")
        else:
            print(f"   ❌ Failed to create clip {i+1}")
    
    print(f"\n✓ Done! Clips saved to {output_dir}")


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Background Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AI Background Generation
  python background_generator.py --list              List AI generation styles
  python background_generator.py --generate          Generate AI background
  python background_generator.py --preview           Preview AI prompts

  # Download Real Gameplay Videos (RECOMMENDED)
  python background_generator.py --download          Download with default settings
  python background_generator.py --download --style minecraft_parkour --num-videos 5 --clips-per-video 10
  python background_generator.py --download --style subway_surfers -n 3 -c 8
  
  # Batch download multiple styles at once
  python background_generator.py --download-all      Download minecraft + subway surfers
  python background_generator.py --download-all --num-videos 3 --clips-per-video 10

  # Video Management
  python background_generator.py --split video.mp4  Split video into clips
  python background_generator.py --show-parsed      Show which videos have been downloaded
  python background_generator.py --clear-parsed     Clear download history (allows re-downloading)
  
  # Search for new videos without downloading
  python background_generator.py --search minecraft_parkour

Available styles for --style:
  - minecraft_parkour  (Minecraft parkour POV gameplay)
  - subway_surfers     (Subway Surfers endless runner)
  - satisfying         (Satisfying/ASMR videos)
  - gaming_general     (General gaming backgrounds)
        """
    )
    
    # AI Generation
    parser.add_argument('--generate', action='store_true', help='Generate an AI background video')
    parser.add_argument('--list', action='store_true', help='List available AI styles')
    parser.add_argument('--preview', action='store_true', help='Preview AI prompts')
    parser.add_argument('--recommend', action='store_true', help='Show recommended gameplay sources')
    
    # Download options
    parser.add_argument('--download', action='store_true', 
                        help='Download real no-copyright gameplay videos')
    parser.add_argument('--download-all', action='store_true',
                        help='Download videos for all common styles (minecraft + subway surfers)')
    parser.add_argument('--style', type=str, default='minecraft_parkour',
                        help='Style: minecraft_parkour, subway_surfers, satisfying, gaming_general')
    parser.add_argument('--num-videos', '-n', type=int, default=3,
                        help='Number of source videos to download (default: 3)')
    parser.add_argument('--clips-per-video', '-c', type=int, default=5,
                        help='Number of clips to extract from each video (default: 5)')
    parser.add_argument('--clip-duration', '-d', type=int, default=90,
                        help='Duration of each clip in seconds (default: 90)')
    
    # Manual splitting
    parser.add_argument('--split', type=str, metavar='VIDEO', help='Split a video into clips')
    parser.add_argument('--clips', type=int, default=5, help='Number of clips for --split (default: 5)')
    
    # Parsed video management
    parser.add_argument('--show-parsed', action='store_true',
                        help='Show which videos have already been downloaded')
    parser.add_argument('--clear-parsed', action='store_true',
                        help='Clear the parsed videos history (allows re-downloading)')
    parser.add_argument('--clear-style', type=str, metavar='STYLE',
                        help='Clear parsed history for a specific style only')
    
    # Search only (no download)
    parser.add_argument('--search', type=str, metavar='STYLE',
                        help='Search for similar videos without downloading')
    
    args = parser.parse_args()
    
    if args.list:
        print("\n📋 Available AI Background Styles:")
        for style in list_available_styles():
            print(f"  • {style}")
        print("\n📋 Available Download Styles:")
        for style in VIDEO_SEARCH_QUERIES.keys():
            print(f"  • {style}")
        print("\n💡 For better quality, use --download to get real gameplay videos!")
        print()
    
    elif args.preview:
        preview_prompts(args.style)
    
    elif args.recommend:
        print(RECOMMENDED_GAMEPLAY_SOURCES)
    
    elif args.show_parsed:
        parsed = load_parsed_videos()
        print("\n📋 Parsed Videos History")
        print("="*50)
        if not parsed:
            print("No videos have been downloaded yet.")
        else:
            total = 0
            for style, video_ids in parsed.items():
                print(f"\n🎮 {style}: {len(video_ids)} videos")
                for vid in video_ids[:5]:
                    print(f"   • {vid}")
                if len(video_ids) > 5:
                    print(f"   ... and {len(video_ids) - 5} more")
                total += len(video_ids)
            print(f"\n📊 Total: {total} videos parsed")
        print("="*50 + "\n")
    
    elif args.clear_parsed:
        clear_parsed_videos()
    
    elif args.clear_style:
        clear_parsed_videos(args.clear_style)
    
    elif args.search:
        print(f"\n🔍 Searching for {args.search} videos...")
        videos = search_similar_videos(
            style=args.search,
            num_results=10,
            exclude_parsed=True
        )
        print(f"\n📋 Found {len(videos)} new videos:\n")
        for i, v in enumerate(videos, 1):
            duration_str = f"{v.get('duration', 0)//60}m" if v.get('duration') else "?"
            print(f"  {i:2}. [{duration_str:>4}] {v['title'][:50]}...")
            print(f"      {v['url']}")
        print()
    
    elif args.download_all:
        download_multiple_styles(
            styles=["minecraft_parkour", "subway_surfers"],
            num_videos_per_style=args.num_videos,
            clips_per_video=args.clips_per_video,
            clip_duration=args.clip_duration
        )
    
    elif args.download:
        download_gameplay_videos(
            style=args.style,
            num_videos=args.num_videos,
            clips_per_video=args.clips_per_video,
            clip_duration=args.clip_duration
        )
    
    elif args.split:
        split_video_into_clips(args.split, num_clips=args.clips, clip_duration=args.clip_duration)
    
    elif args.generate:
        print("\n🎬 Generating AI Background Video...")
        result = generate_background_video(style=args.style)
        
        if result:
            print(f"\n✓ Generated successfully!")
            print(f"  Style:  {result['style']}")
            print(f"  Effect: {result['effect']}")
            print(f"  Video:  {result['video_path']}")
            print(f"  Image:  {result['image_path']}")
        else:
            print("\n✗ Generation failed")
    
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("💡 RECOMMENDED: Use real gameplay videos for best quality!")
        print("="*60)
        print("\nQuick start:")
        print("  python background_generator.py --download --num-videos 5 --clips-per-video 10")
        print("  (Downloads 5 videos, creates ~50 clips)")
        print("="*60)