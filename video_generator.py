#!/usr/bin/env python3
"""
Video Generator Module for YouTube Shorts Automation
=====================================================

Creates complete YouTube Shorts videos with:
- AI voice narration (gTTS or Piper TTS)
- Word-level synchronized captions
- Background video composition
- Vertical format output (1080x1920)

TTS Options:
- gTTS: Google Translate TTS - free, online, easy setup (DEFAULT)
- Piper: Local/offline, better quality, requires additional setup

Cost: $0 (both gTTS and Piper are free)
"""

import os
import json
import random
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Project directory
PROJECT_DIR = Path(__file__).parent


def check_piper_available() -> bool:
    """
    Check if Piper TTS is installed and available.
    
    Returns:
        True if Piper is available, False otherwise
    """
    try:
        import piper
        return True
    except ImportError:
        pass
    
    # Try CLI as fallback
    try:
        result = subprocess.run(
            ['piper', '--help'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        pass
    
    return False


def check_gtts_available() -> bool:
    """
    Check if gTTS is installed and available.
    
    Returns:
        True if gTTS is available, False otherwise
    """
    try:
        from gtts import gTTS
        return True
    except ImportError:
        return False


def get_piper_install_instructions() -> str:
    """Get Piper installation instructions."""
    return """
Piper TTS is not installed!

Install Piper:
  pip install piper-tts

Or for specific platforms:
  # macOS/Linux
  pip install piper-tts
  
  # With specific voice models
  pip install piper-tts[voices]

After installing, voice models will be auto-downloaded on first use.
Available voices: https://github.com/rhasspy/piper#voices

Default voice: en_US-lessac-medium (clear American English)
"""


class VideoGenerator:
    """
    Complete video composition pipeline for YouTube Shorts.
    
    Features:
    - Multiple TTS options (gTTS default, Piper for better quality)
    - Word-level timing extraction
    - Caption generation (1-2 words at a time)
    - Background video handling with caching
    - FFmpeg-based video composition
    """
    
    def __init__(self, output_dir: str = "./output", videos_dir: str = "./videos"):
        """
        Initialize the video generator.
        
        Args:
            output_dir: Directory for generated videos
            videos_dir: Directory containing background videos
        """
        self.output_dir = Path(output_dir)
        self.videos_dir = Path(videos_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config()
        
        # TTS configuration
        self.tts_config = self.config.get('tts', {})
        self.tts_provider = self.tts_config.get('provider', 'gtts')
        self.tts_speed = self.tts_config.get('speed', 1.0)  # Global speed multiplier
        
        # ElevenLabs settings (best quality)
        elevenlabs_config = self.tts_config.get('elevenlabs', {})
        self.elevenlabs_api_key = elevenlabs_config.get('api_key', '')
        self.elevenlabs_voice = elevenlabs_config.get('voice', 'Rachel')
        self.elevenlabs_model = elevenlabs_config.get('model', 'eleven_multilingual_v2')
        self.elevenlabs_stability = elevenlabs_config.get('stability', 0.5)
        self.elevenlabs_similarity = elevenlabs_config.get('similarity_boost', 0.75)
        
        # gTTS settings
        gtts_config = self.tts_config.get('gtts', {})
        self.gtts_language = gtts_config.get('language', 'en')
        self.gtts_slow = gtts_config.get('slow', False)
        
        # Piper settings
        piper_config = self.tts_config.get('piper', {})
        self.piper_voice = piper_config.get('voice', 'en_US-lessac-medium')
        self.piper_speaker_id = piper_config.get('speaker_id', 0)
        self.piper_speed = piper_config.get('speed', 1.0)
        self.piper_data_dir = Path(piper_config.get('data_dir', './.piper'))
        
        # Video settings
        video_config = self.config.get('video', {})
        self.max_duration = video_config.get('max_duration', 120)
        self.video_speed = video_config.get('speed', 1.0)
        
        # Caption settings (viral style)
        captions_config = video_config.get('captions', {})
        self.viral_style = captions_config.get('viral_style', True)
        
        # Word grouping settings (1-3 words at a time)
        self.caption_min_words = captions_config.get('min_words', 1)
        self.caption_max_words = captions_config.get('max_words', 3)
        
        # Animation settings (bounce effect)
        anim_config = captions_config.get('animation', {})
        self.anim_enabled = anim_config.get('enabled', True)
        self.anim_scale_start = anim_config.get('scale_start', 130)
        self.anim_scale_end = anim_config.get('scale_end', 100)
        self.anim_duration_ms = anim_config.get('duration_ms', 80)
        
        # Center word settings
        center_config = captions_config.get('center', {})
        self.center_fontsize = center_config.get('fontsize', 100)
        self.center_color = center_config.get('color', 'white')
        self.center_outline_color = center_config.get('outline_color', 'black')
        self.center_outline_width = center_config.get('outline_width', 5)
        self.center_bold = center_config.get('bold', True)
        self.center_italic = center_config.get('italic', True)
        self.center_position_y = center_config.get('position_y', 0.48)
        
        # Top context settings
        top_config = captions_config.get('top_context', {})
        self.top_context_enabled = top_config.get('enabled', False)
        self.top_fontsize = top_config.get('fontsize', 36)
        self.top_color = top_config.get('color', 'white')
        self.top_bg_color = top_config.get('background_color', 'black')
        self.top_bg_opacity = top_config.get('background_opacity', 0.7)
        self.top_position_y = top_config.get('position_y', 0.08)
        self.top_max_words = top_config.get('max_words', 12)
        
        # Create directories
        self.piper_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"VideoGenerator initialized. Output: {self.output_dir}")
        logger.info(f"TTS Provider: {self.tts_provider}")
        if self.tts_speed != 1.0:
            logger.info(f"TTS Speed: {self.tts_speed}x")
        if self.video_speed != 1.0:
            logger.info(f"Final Video Speed: {self.video_speed}x")

    def _build_atempo_filter(self, speed: float) -> str:
        """
        Build an FFmpeg atempo filter chain for the given speed.

        FFmpeg atempo only supports 0.5x - 2.0x per filter, so we chain if needed.
        """
        atempo_filters = []
        remaining_speed = speed

        while remaining_speed > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_speed /= 2.0
        while remaining_speed < 0.5:
            atempo_filters.append("atempo=0.5")
            remaining_speed /= 0.5

        atempo_filters.append(f"atempo={remaining_speed:.4f}")
        return ",".join(atempo_filters)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        import yaml
        
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def synthesize_speech_elevenlabs(self, text: str, output_filename: str = "narration.mp3") -> Optional[str]:
        """
        Convert text to speech using ElevenLabs API.
        
        BEST QUALITY option - realistic AI voices.
        FREE tier: 10,000 characters per month
        
        Args:
            text: Story text to convert
            output_filename: Name of output audio file
        
        Returns:
            Path to audio file, or None if synthesis fails
        """
        if not self.elevenlabs_api_key:
            logger.warning("ElevenLabs API key not set! Add it to config.yaml")
            return None
        
        try:
            from elevenlabs import ElevenLabs
        except ImportError:
            logger.error("ElevenLabs not installed! Run: pip install elevenlabs")
            return None
        
        # Ensure output filename ends with .mp3
        if not output_filename.endswith('.mp3'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp3'
        
        audio_path = self.output_dir / output_filename
        
        try:
            logger.info(f"Synthesizing speech with ElevenLabs (voice: {self.elevenlabs_voice})...")
            
            # Initialize client
            client = ElevenLabs(api_key=self.elevenlabs_api_key)
            
            # Get voice ID from voice name
            voice_id = self._get_elevenlabs_voice_id(client, self.elevenlabs_voice)
            if not voice_id:
                logger.warning(f"Voice '{self.elevenlabs_voice}' not found, using default")
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel default
            
            # Generate audio
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=self.elevenlabs_model,
                voice_settings={
                    "stability": self.elevenlabs_stability,
                    "similarity_boost": self.elevenlabs_similarity
                }
            )
            
            # Save to file (audio is a generator)
            with open(audio_path, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            
            logger.info(f"✓ Audio saved: {audio_path}")
            return str(audio_path)
        
        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {str(e)}")
            return None
    
    def _get_elevenlabs_voice_id(self, client, voice_name: str) -> Optional[str]:
        """Get voice ID from voice name."""
        # Common voice name to ID mappings
        voice_map = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "drew": "29vD33N1CtxCmqQRPOHJ",
            "clyde": "2EiwWnXFnvU5JabPnv8n",
            "paul": "5Q0t7uMcjvnagumLfvZi",
            "domi": "AZnzlk1XvdvUeBnXmlld",
            "dave": "CYw3kZ02Hs0563khs1Fj",
            "fin": "D38z5RcWu1voky8WS1ja",
            "sarah": "EXAVITQu4vr4xnSDxMaL",
            "antoni": "ErXwobaYiN019PkySvjV",
            "thomas": "GBv7mTt0atIp3Br8iCZE",
            "charlie": "IKne3meq5aSn9XLyUdCD",
            "george": "JBFqnCBsd6RMkjVDRZzb",
            "emily": "LcfcDJNUP1GQjkzn1xUU",
            "elli": "MF3mGyEYCl7XYWbV9V6O",
            "callum": "N2lVS1w4EtoT3dr4eOWO",
            "patrick": "ODq5zmih8GrVes37Dizd",
            "harry": "SOYHLrjzK2X1ezoPC6cr",
            "liam": "TX3LPaxmHKxFdv7VOQHJ",
            "dorothy": "ThT5KcBeYPX3keUQqHPh",
            "josh": "TxGEqnHWrfWFTfGW9XjX",
            "arnold": "VR6AewLTigWG4xSOukaG",
            "charlotte": "XB0fDUnXU5powFXDhCwa",
            "matilda": "XrExE9yKIg1WjnnlVkGX",
            "matthew": "Yko7PKHZNXotIFUBG7I9",
            "james": "ZQe5CZNOzWyzPSCn5a3c",
            "joseph": "Zlb1dXrM653N07WRdFW3",
            "jeremy": "bVMeCyTHy58xNoL34h3p",
            "michael": "flq6f7yk4E4fJM5XTYuZ",
            "ethan": "g5CIjZEefAph4nQFvHAz",
            "gigi": "jBpfuIE2acCO8z3wKNLl",
            "freya": "jsCqWAovK2LkecY7zXl4",
            "grace": "oWAxZDx7w5VEj9dCyTzz",
            "daniel": "onwK4e9ZLuTAKqWW03F9",
            "lily": "pFZP5JQG7iQjIQuC4Bku",
            "serena": "pMsXgVXv3BLzUgSXRplE",
            "adam": "pNInz6obpgDQGcFmaJgB",
            "nicole": "piTKgcLEGmPE4e6mEKli",
            "bill": "pqHfZKP75CvOlQylNhV4",
            "jessie": "t0jbNlBVZ17f02VDIeMI",
            "sam": "yoZ06aMxZJJ28mfd3POQ",
            "glinda": "z9fAnlkpzviPz146aGWa",
            "mimi": "zrHiDhphv9ZnVXBqCLjz",
        }
        
        # Try direct lookup
        voice_lower = voice_name.lower()
        if voice_lower in voice_map:
            return voice_map[voice_lower]
        
        # Try to fetch from API if not in map
        try:
            voices = client.voices.get_all()
            for voice in voices.voices:
                if voice.name.lower() == voice_lower:
                    return voice.voice_id
        except Exception as e:
            logger.warning(f"Could not fetch voices: {e}")
        
        return None
    
    def synthesize_speech_gtts(self, text: str, output_filename: str = "narration.mp3") -> Optional[str]:
        """
        Convert text to speech using gTTS (Google Translate TTS).
        
        This is the DEFAULT and EASIEST option - requires internet but no API keys.
        
        Args:
            text: Story text to convert
            output_filename: Name of output audio file
        
        Returns:
            Path to audio file, or None if synthesis fails
        """
        try:
            from gtts import gTTS
        except ImportError:
            logger.error(
                "gTTS not installed!\n"
                "Install with: pip install gtts"
            )
            return None
        
        # Ensure output filename ends with .mp3
        if not output_filename.endswith('.mp3'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.mp3'
        
        audio_path = self.output_dir / output_filename
        
        try:
            logger.info(f"Synthesizing speech with gTTS (language: {self.gtts_language})...")
            
            # Create TTS object
            tts = gTTS(text=text, lang=self.gtts_language, slow=self.gtts_slow)
            
            # Save to file
            tts.save(str(audio_path))
            
            logger.info(f"✓ Audio saved: {audio_path}")
            return str(audio_path)
        
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {str(e)}")
            return None
    
    def synthesize_speech_piper(self, text: str, output_filename: str = "narration.wav") -> Optional[str]:
        """
        Convert text to speech using Piper TTS (local/offline).
        
        Better quality than gTTS but requires additional setup.
        
        Args:
            text: Story text to convert
            output_filename: Name of output audio file
        
        Returns:
            Path to audio file, or None if synthesis fails
        """
        # Ensure output filename ends with .wav
        if not output_filename.endswith('.wav'):
            output_filename = output_filename.rsplit('.', 1)[0] + '.wav'
        
        audio_path = self.output_dir / output_filename
        
        try:
            # Try Python API first
            try:
                from piper import PiperVoice
                from piper.download import ensure_voice_exists, find_voice, get_voices
                import wave
                
                logger.info(f"Synthesizing speech with Piper TTS (voice: {self.piper_voice})...")
                
                # Get voice model path
                model_path = None
                config_path = None
                
                # Check if voice exists, download if needed
                try:
                    voices = get_voices(str(self.piper_data_dir), update_voices=True)
                    if self.piper_voice in voices:
                        voice_info = voices[self.piper_voice]
                        # Download voice if needed
                        ensure_voice_exists(
                            self.piper_voice,
                            data_dirs=[str(self.piper_data_dir)],
                            download_dir=str(self.piper_data_dir),
                            update_voices=False
                        )
                        model_path, config_path = find_voice(
                            self.piper_voice,
                            data_dirs=[str(self.piper_data_dir)]
                        )
                except Exception as e:
                    logger.warning(f"Could not get voice list: {e}")
                    # Try direct path
                    model_path = self.piper_data_dir / f"{self.piper_voice}.onnx"
                    config_path = self.piper_data_dir / f"{self.piper_voice}.onnx.json"
                
                if model_path and Path(model_path).exists():
                    # Load voice
                    voice = PiperVoice.load(str(model_path), config_path=str(config_path) if config_path else None)
                    
                    # Synthesize to WAV
                    with wave.open(str(audio_path), 'wb') as wav_file:
                        voice.synthesize(text, wav_file, speaker_id=self.piper_speaker_id)
                    
                    logger.info(f"✓ Audio saved: {audio_path}")
                    return str(audio_path)
                else:
                    logger.info("Voice model not found locally, trying CLI download...")
                    raise FileNotFoundError("Voice model not found")
                    
            except ImportError:
                logger.info("Piper Python API not available, trying CLI...")
                raise ImportError("Use CLI instead")
            
        except (ImportError, FileNotFoundError) as e:
            # Fallback to CLI
            logger.info("Using Piper CLI...")
            return self._synthesize_piper_cli(text, audio_path)
        
        except Exception as e:
            logger.error(f"Piper TTS synthesis failed: {str(e)}")
            return None
    
    def _synthesize_piper_cli(self, text: str, audio_path: Path) -> Optional[str]:
        """
        Synthesize speech using Piper CLI.
        
        Args:
            text: Text to synthesize
            audio_path: Output audio file path
        
        Returns:
            Path to audio file, or None if failed
        """
        try:
            # Build Piper CLI command
            cmd = [
                'piper',
                '--model', self.piper_voice,
                '--data-dir', str(self.piper_data_dir),
                '--output_file', str(audio_path),
                '--download-dir', str(self.piper_data_dir)
            ]
            
            # Add speaker ID if specified
            if self.piper_speaker_id:
                cmd.extend(['--speaker', str(self.piper_speaker_id)])
            
            logger.info(f"Running Piper CLI with voice: {self.piper_voice}")
            
            # Run Piper with text as stdin
            result = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and audio_path.exists():
                logger.info(f"✓ Audio saved: {audio_path}")
                return str(audio_path)
            else:
                logger.error(f"Piper CLI error: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.error(get_piper_install_instructions())
            return None
        except Exception as e:
            logger.error(f"Piper CLI failed: {str(e)}")
            return None
    
    def adjust_audio_speed(self, audio_file: str, speed: float) -> Optional[str]:
        """
        Adjust audio playback speed using FFmpeg.
        
        Args:
            audio_file: Path to input audio file
            speed: Speed multiplier (1.0 = normal, 1.5 = 50% faster, 0.8 = 20% slower)
        
        Returns:
            Path to speed-adjusted audio file, or None if failed
        """
        if speed == 1.0:
            return audio_file
        
        # Create output path
        audio_path = Path(audio_file)
        output_path = audio_path.parent / f"speed_{audio_path.name}"
        
        try:
            # FFmpeg atempo filter only accepts values between 0.5 and 2.0
            # For larger changes, we chain multiple atempo filters
            atempo_filters = []
            remaining_speed = speed
            
            while remaining_speed > 2.0:
                atempo_filters.append("atempo=2.0")
                remaining_speed /= 2.0
            while remaining_speed < 0.5:
                atempo_filters.append("atempo=0.5")
                remaining_speed /= 0.5
            
            atempo_filters.append(f"atempo={remaining_speed:.4f}")
            filter_str = ",".join(atempo_filters)
            
            cmd = [
                'ffmpeg', '-y',
                '-i', audio_file,
                '-filter:a', filter_str,
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"✓ Audio speed adjusted to {speed}x")
                return str(output_path)
            else:
                logger.warning(f"Speed adjustment failed: {result.stderr}")
                return audio_file
                
        except Exception as e:
            logger.warning(f"Could not adjust speed: {e}")
            return audio_file
    
    def synthesize_speech(self, text: str, output_filename: str = "narration.wav") -> Optional[str]:
        """
        Convert text to speech using configured TTS provider.
        
        Priority order: ElevenLabs → gTTS → Piper
        Applies speed adjustment if configured.
        
        Args:
            text: Story text to convert
            output_filename: Name of output audio file
        
        Returns:
            Path to audio file, or None if synthesis fails
        """
        result = None
        
        if self.tts_provider == 'elevenlabs':
            result = self.synthesize_speech_elevenlabs(text, output_filename)
            if not result:
                # Fall back to gTTS if ElevenLabs fails
                logger.info("ElevenLabs failed, trying gTTS as fallback...")
                result = self.synthesize_speech_gtts(text, output_filename)
        
        elif self.tts_provider == 'gtts':
            result = self.synthesize_speech_gtts(text, output_filename)
            if not result:
                # Fall back to Piper if gTTS fails
                logger.info("gTTS failed, trying Piper as fallback...")
                result = self.synthesize_speech_piper(text, output_filename)
        
        elif self.tts_provider == 'piper':
            result = self.synthesize_speech_piper(text, output_filename)
            if not result:
                # Fall back to gTTS if Piper fails
                logger.info("Piper failed, trying gTTS as fallback...")
                result = self.synthesize_speech_gtts(text, output_filename)
        
        else:
            logger.error(f"Unknown TTS provider: {self.tts_provider}")
            # Try gTTS as default
            result = self.synthesize_speech_gtts(text, output_filename)
        
        # Apply speed adjustment if configured
        if result and self.tts_speed != 1.0:
            result = self.adjust_audio_speed(result, self.tts_speed)
        
        return result
    
    def extract_word_timings_forcealign(self, audio_file: str, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract ACCURATE word-level timing using Forced Alignment.
        
        This uses the ForceAlign library (Wav2Vec2-based) to get precise
        timestamps for each word in the audio. This is the most accurate method
        because it analyzes the actual audio waveform.
        
        Args:
            audio_file: Path to audio file
            text: Original transcript text
        
        Returns:
            List of word timing dictionaries, or None if alignment fails
        """
        try:
            from forcealign import ForceAlign
        except ImportError:
            logger.warning("ForceAlign not available. Install with: pip install forcealign")
            return None
        
        logger.info("Extracting word timings with Forced Alignment (most accurate)...")
        
        try:
            import re
            import tempfile
            from difflib import SequenceMatcher

            def normalize_word(word: str) -> str:
                # Keep letters, numbers, and apostrophes; strip everything else
                return re.sub(r"[^a-z0-9']+", "", word.lower())

            # Convert audio to WAV (16kHz mono) for more stable alignment
            alignment_audio = audio_file
            temp_wav = None
            try:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_wav = temp_file.name
                temp_file.close()
                convert_cmd = [
                    'ffmpeg', '-y', '-i', audio_file,
                    '-ar', '16000', '-ac', '1', '-f', 'wav', temp_wav
                ]
                result = subprocess.run(convert_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    alignment_audio = temp_wav
                else:
                    temp_wav = None
            except Exception:
                temp_wav = None

            # ForceAlign requires the transcript as a string
            fa = ForceAlign(audio_file=alignment_audio, transcript=text)
            
            # Run alignment - this returns word objects with time_start and time_end
            aligned_words = fa.inference()
            
            if not aligned_words:
                logger.warning("ForceAlign returned no words")
                return None
            
            # Get original words to preserve case and align by normalized tokens
            original_words = text.split()
            original_norm = [normalize_word(w) for w in original_words]
            aligned_list = [w.word for w in aligned_words]
            aligned_norm = [normalize_word(w) for w in aligned_list]

            # Build mapping between aligned indices and original indices
            aligned_to_original = {}
            original_to_aligned = {}
            matcher = SequenceMatcher(None, aligned_norm, original_norm)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for offset in range(i2 - i1):
                        aligned_idx = i1 + offset
                        original_idx = j1 + offset
                        aligned_to_original[aligned_idx] = original_idx
                        original_to_aligned[original_idx] = aligned_idx

            # Build timings for ALL original words (prevents numbers from disappearing)
            aligned_times = [(w.time_start, w.time_end) for w in aligned_words]
            last_aligned_end = aligned_times[-1][1] if aligned_times else 0.0

            word_timings = []
            idx = 0
            while idx < len(original_words):
                if idx in original_to_aligned:
                    aligned_idx = original_to_aligned[idx]
                    start_time, end_time = aligned_times[aligned_idx]
                    if end_time < start_time:
                        end_time = start_time + 0.05
                    word_timings.append({
                        "word": original_words[idx],
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time
                    })
                    idx += 1
                    continue

                # Handle a block of unmatched original words by interpolating timing
                block_start = idx
                while idx < len(original_words) and idx not in original_to_aligned:
                    idx += 1
                block_end = idx - 1
                block_len = block_end - block_start + 1

                # Find surrounding matched timings
                prev_idx = block_start - 1
                while prev_idx >= 0 and prev_idx not in original_to_aligned:
                    prev_idx -= 1
                next_idx = idx
                while next_idx < len(original_words) and next_idx not in original_to_aligned:
                    next_idx += 1

                if prev_idx >= 0:
                    prev_aligned = original_to_aligned[prev_idx]
                    prev_time = aligned_times[prev_aligned][1]
                else:
                    prev_time = 0.0

                if next_idx < len(original_words):
                    next_aligned = original_to_aligned[next_idx]
                    next_time = aligned_times[next_aligned][0]
                else:
                    next_time = last_aligned_end

                gap = max(0.0, next_time - prev_time)
                if gap <= 0.0:
                    per = 0.12
                else:
                    per = gap / (block_len + 1)

                for offset in range(block_len):
                    start_time = prev_time + per * (offset + 1)
                    end_time = start_time + max(0.08, per * 0.9)
                    word_timings.append({
                        "word": original_words[block_start + offset],
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time
                    })
            
            logger.info(f"✓ Forced alignment complete: {len(word_timings)} words with accurate timing")
            return word_timings
            
        except Exception as e:
            logger.warning(f"ForceAlign failed: {e}")
            return None
        finally:
            if 'temp_wav' in locals() and temp_wav:
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass
    
    def extract_word_timings_fallback(self, audio_file: str, text: str) -> List[Dict[str, Any]]:
        """
        Fallback word timing extraction using weighted estimation.
        
        Used when ForceAlign is not available or fails.
        
        Args:
            audio_file: Path to audio file
            text: Original text (for word extraction)
        
        Returns:
            List of word timing dictionaries
        """
        # Get audio duration using FFprobe
        duration = self.get_audio_duration(audio_file)
        logger.info(f"Using fallback timing estimation (audio duration: {duration:.2f}s)")
        
        # Split text into words
        words = text.split()
        
        if not words:
            return []
        
        # Calculate weights based on word characteristics
        word_weights = []
        for word in words:
            char_weight = len(word)
            vowels = sum(1 for c in word.lower() if c in 'aeiou')
            syllable_weight = max(1, vowels)
            weight = max(char_weight * 0.7 + syllable_weight * 0.3, 1.0)
            word_weights.append(weight)
        
        total_weight = sum(word_weights)
        
        # Small buffer at start and end
        start_buffer = 0.05
        end_buffer = 0.1
        usable_duration = duration - start_buffer - end_buffer
        
        # Calculate timings based on weights
        word_timings = []
        current_time = start_buffer
        
        for i, word in enumerate(words):
            word_duration = max((word_weights[i] / total_weight) * usable_duration, 0.08)
            
            word_timings.append({
                "word": word,
                "start_time": current_time,
                "end_time": current_time + word_duration,
                "duration": word_duration
            })
            
            current_time += word_duration
        
        # Normalize to fit actual usable duration
        actual_total = sum(wt["duration"] for wt in word_timings)
        if actual_total > 0:
            scale = usable_duration / actual_total
            current_time = start_buffer
            for wt in word_timings:
                wt["duration"] *= scale
                wt["start_time"] = current_time
                wt["end_time"] = current_time + wt["duration"]
                current_time += wt["duration"]
        
        logger.info(f"✓ Fallback: {len(word_timings)} word timings (weighted estimation)")
        return word_timings
    
    def extract_word_timings(self, audio_file: str, text: str) -> List[Dict[str, Any]]:
        """
        Extract word-level timing from audio.
        
        Priority:
        1. ForceAlign (Wav2Vec2-based forced alignment) - MOST ACCURATE
        2. Fallback to weighted estimation if ForceAlign fails
        
        Args:
            audio_file: Path to audio file
            text: Original text (for word extraction)
        
        Returns:
            List of word timing dictionaries
        """
        # Try ForceAlign first (most accurate)
        word_timings = self.extract_word_timings_forcealign(audio_file, text)
        
        if word_timings:
            return word_timings
        
        # Fallback to estimation
        logger.info("Falling back to weighted timing estimation...")
        return self.extract_word_timings_fallback(audio_file, text)
    
    def get_background_video(self, force_generate: bool = False) -> Optional[str]:
        """
        Get a background video - either AI-generated or from static files.
        
        Priority:
        1. If auto_generate enabled: Generate fresh AI background
        2. Fall back to static videos in videos/ directory
        
        Args:
            force_generate: Force new AI generation even if static videos exist
        
        Returns:
            Path to video file, or None if no videos available
        """
        # Check if AI generation is enabled
        bg_config = self.config.get('background', {})
        auto_generate = bg_config.get('auto_generate', False)
        
        if auto_generate or force_generate:
            try:
                from background_generator import get_or_generate_background
                
                logger.info("Attempting AI background generation...")
                generated = get_or_generate_background(force_new=force_generate)
                
                if generated:
                    logger.info(f"✓ Using AI-generated background: {Path(generated).name}")
                    return generated
                else:
                    logger.warning("AI generation failed, falling back to static videos")
            except ImportError:
                logger.warning("background_generator not available, using static videos")
            except Exception as e:
                logger.warning(f"AI generation error: {e}, using static videos")
        
        # Fall back to static videos
        if not self.videos_dir.exists():
            self.videos_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported video formats
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        # Find all videos (including generated subfolder)
        videos = []
        for ext in video_extensions:
            videos.extend(self.videos_dir.glob(f"*{ext}"))
            videos.extend(self.videos_dir.glob(f"generated/*{ext}"))
        
        if not videos:
            logger.warning(
                f"No background videos found in {self.videos_dir}\n"
                "Either enable background.auto_generate in config.yaml\n"
                "Or add .mp4 videos to the videos/ directory"
            )
            return None
        
        # Select random video
        selected = random.choice(videos)
        logger.info(f"Selected background: {selected.name}")
        return str(selected)
    
    def create_caption_clips(self, word_timings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create caption timing for video overlay - VIRAL STYLE.
        
        Groups 1-3 words randomly for variety, creating the fast,
        engaging TikTok-style word pop-up effect.
        
        Args:
            word_timings: List of word timing dictionaries
        
        Returns:
            List of caption clip data (1-3 words per clip)
        """
        if not word_timings:
            return []
        
        caption_clips = []
        i = 0
        
        while i < len(word_timings):
            # Randomly choose how many words for this caption (1-3)
            num_words = random.randint(self.caption_min_words, self.caption_max_words)
            
            # Don't exceed available words
            num_words = min(num_words, len(word_timings) - i)
            
            # Get the words for this caption
            group = word_timings[i:i + num_words]
            
            # Combine words into caption text
            caption_text = " ".join(t['word'].strip() for t in group if t['word'].strip())
            
            if caption_text:
                caption_clips.append({
                    "text": caption_text,
                    "start": group[0]["start_time"],
                    "end": group[-1]["end_time"],
                    "duration": group[-1]["end_time"] - group[0]["start_time"]
                })
            
            i += num_words
        
        return caption_clips
    
    def create_context_captions(self, word_timings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create running context captions for the top of the screen.
        
        Shows a sliding window of recent words to give context,
        like the smaller text at the top in viral TikTok videos.
        
        Args:
            word_timings: List of word timing dictionaries
        
        Returns:
            List of context caption data
        """
        if not word_timings or not self.top_context_enabled:
            return []
        
        context_clips = []
        words = [t['word'] for t in word_timings]
        
        # Create context windows
        window_size = self.top_max_words
        
        for i, timing in enumerate(word_timings):
            # Get surrounding context (words before and current)
            start_idx = max(0, i - window_size + 1)
            context_words = words[start_idx:i + 1]
            context_text = " ".join(context_words)
            
            # Trim if too long
            if len(context_text) > 60:
                # Keep the most recent words
                while len(context_text) > 60 and len(context_words) > 3:
                    context_words = context_words[1:]
                    context_text = " ".join(context_words)
            
            context_clips.append({
                "text": context_text,
                "start": timing["start_time"],
                "end": timing["end_time"],
                "highlight_word": timing['word']  # The current word to potentially highlight
            })
        
        return context_clips
    
    def _group_captions(self, captions: List[Dict[str, Any]], max_groups: int = 50) -> List[Dict[str, Any]]:
        """
        Group captions together to reduce FFmpeg filter complexity.
        
        Args:
            captions: List of caption timing data
            max_groups: Maximum number of caption groups to create
        
        Returns:
            List of grouped caption data
        """
        if not captions:
            return []
        
        if len(captions) <= max_groups:
            return captions
        
        # Calculate how many captions to combine per group
        group_size = max(1, len(captions) // max_groups)
        
        grouped = []
        i = 0
        while i < len(captions):
            # Get a group of captions
            group = captions[i:i + group_size]
            if group:
                # Combine text and timing
                combined_text = " ".join(c['text'] for c in group)
                # Limit text length for readability
                if len(combined_text) > 40:
                    words = combined_text.split()
                    combined_text = " ".join(words[:6])  # Max 6 words
                
                grouped.append({
                    'text': combined_text,
                    'start': group[0]['start'],
                    'end': group[-1]['end'],
                    'duration': group[-1]['end'] - group[0]['start']
                })
            i += group_size
        
        return grouped
    
    def create_viral_ass_subtitles(
        self, 
        center_captions: List[Dict[str, Any]], 
        context_captions: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Create an ASS subtitle file with VIRAL TikTok-style captions.
        
        Features:
        - CenterWord: Large, bold, italic word in the center that pops up
        - TopContext: Running sentence at the top with semi-transparent background
        
        Args:
            center_captions: List of single-word caption timing data (center pop-up)
            context_captions: List of context sentence data (top running text)
            output_path: Where to save the .ass file
        
        Returns:
            Path to the ASS file
        """
        # Calculate vertical positions in ASS format (margin from bottom)
        # ASS uses MarginV from bottom for Alignment 5 (center)
        center_margin_v = int(1920 * (1 - self.center_position_y) - 50)
        top_margin_v = int(1920 * (1 - self.top_position_y) - 20)
        
        # Bold and italic flags for ASS (0=off, 1=on, -1=use style default)
        center_bold = -1 if self.center_bold else 0
        center_italic = -1 if self.center_italic else 0
        
        # Convert hex opacity to ASS alpha (00=opaque, FF=transparent)
        bg_alpha = format(int((1 - self.top_bg_opacity) * 255), '02X')
        
        ass_header = f"""[Script Info]
Title: Viral Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CenterWord,Arial Black,{self.center_fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,{center_bold},{center_italic},0,0,100,100,0,0,1,{self.center_outline_width},2,5,10,10,{center_margin_v},1
Style: TopContext,Arial,{self.top_fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H{bg_alpha}000000,0,0,0,0,100,100,0,0,3,0,0,8,20,20,{top_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        def format_time(seconds: float) -> str:
            """Convert seconds to ASS time format (H:MM:SS.cc)"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            centisecs = int((seconds % 1) * 100)
            return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
        
        def escape_ass_text(text: str) -> str:
            """Escape special characters for ASS format."""
            return text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}').replace('\n', '\\N')
        
        lines = [ass_header]
        
        # Add center word captions (Layer 1 - on top) with bounce animation
        for cap in center_captions:
            start = format_time(cap['start'])
            end = format_time(cap['end'])
            text = escape_ass_text(cap['text'])
            
            # Add bounce animation if enabled
            # ASS animation: \fscx = scale X, \fscy = scale Y, \t = transition
            if self.anim_enabled:
                # Start big, shrink to normal = bounce/pop effect
                anim_prefix = (
                    f"{{\\fscx{self.anim_scale_start}\\fscy{self.anim_scale_start}"
                    f"\\t(0,{self.anim_duration_ms},\\fscx{self.anim_scale_end}\\fscy{self.anim_scale_end})}}"
                )
                text = anim_prefix + text
            
            lines.append(f"Dialogue: 1,{start},{end},CenterWord,,0,0,0,,{text}")
        
        # Add top context captions (Layer 0 - behind)
        if self.top_context_enabled and context_captions:
            for cap in context_captions:
                start = format_time(cap['start'])
                end = format_time(cap['end'])
                text = escape_ass_text(cap['text'])
                lines.append(f"Dialogue: 0,{start},{end},TopContext,,0,0,0,,{text}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"✓ Created viral ASS subtitle file with bounce animation: {output_path}")
        return output_path
    
    def get_audio_duration(self, audio_file: str) -> float:
        """Get the duration of an audio file using FFprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_file
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
        return 60.0  # Default fallback
    
    def compose_video_ffmpeg(
        self,
        background_video: str,
        audio_file: str,
        captions: List[Dict[str, Any]],
        context_captions: List[Dict[str, Any]] = None,
        output_filename: str = "final_short.mp4"
    ) -> Optional[str]:
        """
        Compose final video using FFmpeg with VIRAL-STYLE ASS subtitles.
        
        Features:
        - Big bold word pop-ups in center (one word at a time)
        - Running context sentence at top
        - Proper audio duration matching
        - Background video loops to fill entire audio
        
        Args:
            background_video: Path to background video
            audio_file: Path to audio narration
            captions: List of single-word caption timing data
            context_captions: List of context sentence data (optional)
            output_filename: Name of output file
        
        Returns:
            Path to output video, or None if composition fails
        """
        output_path = self.output_dir / output_filename
        
        # Check FFmpeg availability
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "FFmpeg not found! Please install FFmpeg:\n"
                "  macOS: brew install homebrew-ffmpeg/ffmpeg/ffmpeg\n"
                "  Ubuntu: sudo apt-get install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html"
            )
            return None
        
        # Get audio duration to ensure video is long enough
        audio_duration = self.get_audio_duration(audio_file)
        logger.info(f"Audio duration: {audio_duration:.1f}s")

        # Apply final video speed (affects both audio + video)
        speed = self.video_speed if self.video_speed else 1.0
        if speed != 1.0:
            effective_audio_duration = audio_duration / speed
            logger.info(f"Applying final video speed: {speed}x (effective duration {effective_audio_duration:.1f}s)")
        else:
            effective_audio_duration = audio_duration
        
        # Apply max duration limit if configured
        if self.max_duration > 0:
            video_duration = min(effective_audio_duration + 0.5, self.max_duration)  # Add small buffer
        else:
            video_duration = effective_audio_duration + 0.5
        
        logger.info("Composing video with FFmpeg (viral caption style)...")
        
        try:
            # Create ASS subtitle file with viral style
            import tempfile
            ass_file = os.path.join(tempfile.gettempdir(), "viral_captions.ass")
            
            video_filters = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
            if captions:
                self.create_viral_ass_subtitles(
                    center_captions=captions,
                    context_captions=context_captions or [],
                    output_path=ass_file
                )
                
                # Build filter with ASS subtitles
                # Escape path for FFmpeg filter
                escaped_ass = ass_file.replace("\\", "/").replace(":", "\\:")
                video_filters = f"{video_filters},ass={escaped_ass}"

            if speed != 1.0:
                video_filters = f"{video_filters},setpts=PTS/{speed}"

            if speed != 1.0:
                audio_filters = self._build_atempo_filter(speed)
                filter_complex = f"[0:v]{video_filters}[v];[1:a]{audio_filters}[a]"
                audio_map = "[a]"
            else:
                filter_complex = f"[0:v]{video_filters}[v]"
                audio_map = "1:a"
            
            cmd = [
                'ffmpeg', '-y',
                '-stream_loop', '-1',  # Loop background video infinitely
                '-i', background_video,
                '-i', audio_file,
                '-filter_complex', filter_complex,
                '-map', '[v]',
                '-map', audio_map,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-t', f'{video_duration:.2f}',  # Use audio duration (not -shortest)
                str(output_path)
            ]
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                # Try simpler composition without captions
                return self._compose_simple(background_video, audio_file, output_path, video_duration)
            
            logger.info(f"✓ Video composed with viral captions: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Video composition failed: {str(e)}")
            return None
    
    def _compose_simple(self, background: str, audio: str, output: Path, duration: float = None) -> Optional[str]:
        """Fallback simple composition without captions."""
        logger.info("Attempting simple composition without captions...")
        
        # Get duration if not provided
        if duration is None:
            duration = self.get_audio_duration(audio) + 0.5

        # Apply final video speed (affects both audio + video)
        speed = self.video_speed if self.video_speed else 1.0
        if speed != 1.0:
            duration = (duration / speed)
        
        video_filters = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        if speed != 1.0:
            video_filters = f"{video_filters},setpts=PTS/{speed}"

        if speed != 1.0:
            audio_filters = self._build_atempo_filter(speed)
            filter_complex = f"[0:v]{video_filters}[v];[1:a]{audio_filters}[a]"
            audio_map = "[a]"
        else:
            filter_complex = f"[0:v]{video_filters}[v]"
            audio_map = "1:a"

        cmd = [
            'ffmpeg', '-y',
            '-stream_loop', '-1',
            '-i', background,
            '-i', audio,
            '-filter_complex', filter_complex,
            '-map', '[v]',
            '-map', audio_map,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-c:a', 'aac',
            '-t', f'{duration:.2f}',
            str(output)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✓ Simple video composed: {output}")
            return str(output)
        
        logger.error(f"Simple composition also failed: {result.stderr}")
        return None
    
    def generate_complete_video(self, story_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        End-to-end video generation from story data with VIRAL CAPTION STYLE.
        
        Features:
        - Big bold word pop-ups in center
        - Running context sentence at top
        - Configurable TTS speed
        - Video duration matches audio
        
        Args:
            story_data: Dictionary containing 'story' and 'question' keys
        
        Returns:
            Dictionary with video information, or None if generation fails
        """
        story_text = story_data.get('story', '')
        question = story_data.get('question', 'Unknown')
        
        if not story_text:
            logger.error("No story text provided")
            return None
        
        # Create unique filename based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Synthesize speech (with speed adjustment if configured)
        logger.info("[1/4] Synthesizing speech...")
        audio_file = self.synthesize_speech(
            story_text,
            f"narration_{timestamp}.mp3"
        )
        
        if not audio_file:
            logger.error("Speech synthesis failed")
            return None
        
        # Step 2: Extract word timings
        logger.info("[2/4] Extracting word timings...")
        word_timings = self.extract_word_timings(audio_file, story_text)
        
        # Step 3: Get background video
        logger.info("[3/4] Getting background video...")
        background = self.get_background_video()
        
        if not background:
            logger.error("No background video available")
            return None
        
        # Step 4: Create viral captions and compose video
        logger.info("[4/4] Composing video with viral captions...")
        
        # Create single-word center captions (fast word-by-word pop-up)
        center_captions = self.create_caption_clips(word_timings)
        
        # Create context captions (running sentence at top)
        context_captions = self.create_context_captions(word_timings) if self.top_context_enabled else []
        
        logger.info(f"Created {len(center_captions)} word captions, {len(context_captions)} context captions")
        
        # Get audio duration for accurate video length
        audio_duration = self.get_audio_duration(audio_file)
        
        video_file = self.compose_video_ffmpeg(
            background,
            audio_file,
            center_captions,
            context_captions,
            f"short_{timestamp}.mp4"
        )
        
        if not video_file:
            logger.error("Video composition failed")
            return None
        
        return {
            "video_file": video_file,
            "audio_file": audio_file,
            "word_timings": word_timings,
            "duration": audio_duration,
            "format": "1080x1920",
            "question": question,
            "created_at": datetime.now().isoformat()
        }


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            logger.info(f"✓ FFmpeg found: {version_line}")
            return True
    except FileNotFoundError:
        pass
    
    logger.error(
        "✗ FFmpeg not found!\n"
        "Install FFmpeg:\n"
        "  macOS: brew install ffmpeg\n"
        "  Ubuntu: sudo apt-get install ffmpeg\n"
        "  Windows: Download from https://ffmpeg.org/download.html"
    )
    return False


def check_tts() -> bool:
    """Check if any TTS provider is available."""
    if check_gtts_available():
        logger.info("✓ gTTS available")
        return True
    
    if check_piper_available():
        logger.info("✓ Piper TTS available")
        return True
    
    logger.error(
        "✗ No TTS provider available!\n"
        "Install gTTS (easiest): pip install gtts\n"
        "Or install Piper: pip install piper-tts"
    )
    return False


def check_piper() -> bool:
    """Check if Piper TTS is available."""
    if check_piper_available():
        logger.info("✓ Piper TTS available")
        return True
    
    logger.error(get_piper_install_instructions())
    return False


def test_video_generator():
    """Test video generator with sample content."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*50)
    print("Video Generator Test")
    print("="*50 + "\n")
    
    # Check FFmpeg
    print("1. Checking FFmpeg...")
    if not check_ffmpeg():
        return
    print()
    
    # Check TTS
    print("2. Checking TTS...")
    if not check_tts():
        return
    print()
    
    generator = VideoGenerator()
    
    # Check for background videos
    print("3. Checking background videos...")
    background = generator.get_background_video()
    if not background:
        print("\n⚠️  Add background videos to the videos/ directory")
        print("    Supported formats: .mp4, .mov, .avi, .mkv, .webm")
        return
    print()
    
    # Sample story for testing
    test_story = {
        "question": "What's a moment that changed your perspective on life?",
        "story": "When I was twelve, I found a wallet on the street with three hundred dollars inside. My first instinct was to keep it. But something made me look at the ID. It belonged to an elderly woman named Margaret. I walked two miles to return it. She cried when she opened the door. She told me that money was for her husband's medication. She said I restored her faith in humanity. That day I learned that doing the right thing isn't always easy, but it's always worth it."
    }
    
    print("4. Testing with sample story...")
    result = generator.generate_complete_video(test_story)
    
    if result:
        print(f"\n✓ Video created successfully!")
        print(f"  Video: {result['video_file']}")
        print(f"  Audio: {result['audio_file']}")
        print(f"  Duration: {result['duration']}s")
    else:
        print("\n✗ Video generation failed")


if __name__ == "__main__":
    test_video_generator()
