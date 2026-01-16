#!/usr/bin/env python3
"""
Story Generator Module for YouTube Shorts Automation
=====================================================

Generates emotional, viral-worthy stories using AI (Cohere API free tier).
Features:
- Question-based storytelling (r/AskReddit style)
- Multiple story categories for variety
- Batch generation support
- Fallback to Claude API if needed

Cost: $0/month with Cohere free tier (1,000 calls/month)
"""

import json
import random
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Setup logging
logger = logging.getLogger(__name__)

# Story categories for variety
STORY_CATEGORIES = [
    "childhood_innocence",
    "accidental_good_deed",
    "second_chance",
    "small_moment_big_impact",
    "human_kindness",
    "regret_lesson",
    "emotional_trauma",
    "mental_health",
    "very_sad_story",
]

# Comprehensive story generation prompt
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
- IMPORTANT: Do NOT start the story with "It was..." or any generic time/place opener
- Vary openings: start with dialogue, a shocking moment, a decision, a confession, or an action
- Keep the first sentence punchy (6–12 words) and hooky

EXAMPLES OF OPENING QUESTIONS:
- "What's a memory from your childhood you thought was innocent, but later came back to haunt you?"
- "Have you ever accidentally saved someone's life?"
- "What moment made you realize your best friend wasn't who you thought they were?"
- "What's the smallest decision that completely changed your life?"

EXAMPLES OF STORY OPENINGS (varied):
- "The voicemail ended, and my hands wouldn’t stop shaking."
- "I shouldn’t have opened that text in the grocery aisle."
- "“Don’t tell anyone,” she whispered, handing me the envelope."
- "I hit 'send' and instantly regretted it."
- "The stranger smiled like they already knew me."

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


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml or environment variables."""
    import yaml
    
    config = {}
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    
    # Override with environment variables
    if os.getenv('COHERE_API_KEY'):
        if 'APIs' not in config:
            config['APIs'] = {}
        if 'cohere' not in config['APIs']:
            config['APIs']['cohere'] = {}
        config['APIs']['cohere']['api_key'] = os.getenv('COHERE_API_KEY')
    
    return config


def get_cohere_client():
    """Initialize and return Cohere client."""
    try:
        import cohere
    except ImportError:
        raise ImportError("Please install cohere: pip install cohere")
    
    config = load_config()
    api_key = config.get('APIs', {}).get('cohere', {}).get('api_key', '')
    
    if not api_key or api_key == 'YOUR_COHERE_API_KEY_HERE':
        raise ValueError(
            "Cohere API key not configured!\n"
            "1. Get free key at: https://dashboard.cohere.ai/api-keys\n"
            "2. Add to config.yaml or set COHERE_API_KEY environment variable"
        )
    
    return cohere.ClientV2(api_key=api_key)


def generate_story(category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generate an emotional story using Cohere API.
    
    Args:
        category: Optional story category. If None, randomly selected.
    
    Returns:
        Dictionary containing story data, or None if generation fails.
    """
    if category is None:
        category = random.choice(STORY_CATEGORIES)
    
    prompt = STORY_PROMPT_TEMPLATE.format(category=category)
    
    try:
        co = get_cohere_client()
        
        logger.info(f"Generating story in category: {category}")
        
        # Use command-a-03-2025 (current model as of 2026)
        # See: https://docs.cohere.com/docs/models
        response = co.chat(
            model="command-a-03-2025",  # Updated to current model
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.85,
            p=0.8
        )
        
        story_text = response.message.content[0].text
        
        # Try to parse as JSON
        try:
            # Find JSON in response (handle markdown code blocks)
            if "```json" in story_text:
                json_start = story_text.find("```json") + 7
                json_end = story_text.find("```", json_start)
                story_text = story_text[json_start:json_end].strip()
            elif "```" in story_text:
                json_start = story_text.find("```") + 3
                json_end = story_text.find("```", json_start)
                story_text = story_text[json_start:json_end].strip()
            
            story_data = json.loads(story_text)
            story_data['generated_at'] = datetime.now().isoformat()
            logger.info(f"✓ Story generated: {story_data.get('question', '')[:50]}...")
            return story_data
            
        except json.JSONDecodeError:
            # Fallback: use text as story
            logger.warning("Could not parse JSON, using fallback format")
            return {
                "question": "What changed your life forever?",
                "story": story_text,
                "category": category,
                "keywords": ["story", "emotional", category],
                "generated_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Error generating story: {str(e)}")
        return None


def generate_batch_stories(count: int = 10, save_to_file: bool = True) -> List[Dict[str, Any]]:
    """
    Generate multiple stories at once (batch processing).
    
    Args:
        count: Number of stories to generate
        save_to_file: Whether to save stories to stories_queue.json
    
    Returns:
        List of generated story dictionaries
    """
    stories = []
    
    for i in range(count):
        story = generate_story()
        if story:
            stories.append(story)
            logger.info(f"✓ Story {i+1}/{count} generated")
        else:
            logger.warning(f"✗ Story {i+1}/{count} failed")
    
    if save_to_file and stories:
        queue_path = Path(__file__).parent / "stories_queue.json"
        
        # Load existing queue
        existing = []
        if queue_path.exists():
            with open(queue_path, 'r') as f:
                existing = json.load(f)
        
        # Append new stories
        existing.extend(stories)
        
        with open(queue_path, 'w') as f:
            json.dump(existing, f, indent=2)
        
        logger.info(f"✓ Saved {len(stories)} stories to {queue_path}")
    
    return stories


def get_cached_story() -> Optional[Dict[str, Any]]:
    """
    Get a story from the cached queue (FIFO).
    
    Returns:
        Story dictionary or None if queue is empty
    """
    queue_path = Path(__file__).parent / "stories_queue.json"
    
    if not queue_path.exists():
        return None
    
    with open(queue_path, 'r') as f:
        stories = json.load(f)
    
    if not stories:
        return None
    
    # Pop first story (FIFO)
    story = stories.pop(0)
    
    # Save updated queue
    with open(queue_path, 'w') as f:
        json.dump(stories, f, indent=2)
    
    logger.info(f"✓ Retrieved cached story. {len(stories)} remaining in queue.")
    return story


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for text-to-speech processing.
    Removes problematic characters and limits length.
    
    Args:
        text: Raw story text
    
    Returns:
        Cleaned text suitable for TTS
    """
    # Replace smart quotes with regular quotes
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '—': '-',
        '…': '...',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Limit to 5000 characters (Google TTS API limit per call)
    if len(text) > 5000:
        text = text[:5000]
        logger.warning("Story text truncated to 5000 characters for TTS")
    
    return text


# Alternative: Claude API (if Cohere depleted)
def generate_story_claude(api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Generate story using Claude API (free trial with $5 credit).
    Use this as fallback when Cohere free tier is exhausted.
    
    Args:
        api_key: Claude API key. If None, loads from config/environment.
    
    Returns:
        Story dictionary or None if generation fails
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("Please install anthropic: pip install anthropic")
    
    if api_key is None:
        config = load_config()
        api_key = config.get('APIs', {}).get('claude', {}).get('api_key', '')
        if not api_key:
            api_key = os.getenv('CLAUDE_API_KEY', '')
    
    if not api_key:
        raise ValueError("Claude API key not configured")
    
    client = anthropic.Anthropic(api_key=api_key)
    
    category = random.choice(STORY_CATEGORIES)
    prompt = STORY_PROMPT_TEMPLATE.format(category=category)
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        story_text = message.content[0].text
        
        try:
            return json.loads(story_text)
        except json.JSONDecodeError:
            return {
                "story": story_text,
                "question": "What's your story?",
                "category": category,
                "generated_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Claude API error: {str(e)}")
        return None


def test_connection() -> bool:
    """
    Test Cohere API connection.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        co = get_cohere_client()
        response = co.chat(
            model="command-a-03-2025",  # Updated to current model
            messages=[{"role": "user", "content": "Say 'API working' in 3 words or less."}],
            max_tokens=20
        )
        logger.info("✓ Cohere API connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Cohere API connection failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Test story generation
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*50)
    print("Story Generator Test")
    print("="*50 + "\n")
    
    # Test connection first
    if test_connection():
        # Generate a test story
        story = generate_story()
        if story:
            print("\n✓ Generated Story:")
            print("-" * 40)
            print(f"Question: {story.get('question', 'N/A')}")
            print(f"Category: {story.get('category', 'N/A')}")
            print(f"Keywords: {story.get('keywords', [])}")
            print("-" * 40)
            print(f"Story preview: {story.get('story', '')[:200]}...")
        else:
            print("\n✗ Failed to generate story")
    else:
        print("\n✗ API connection failed. Please check your API key.")
