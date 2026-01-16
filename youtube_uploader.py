#!/usr/bin/env python3
"""
YouTube Uploader Module for YouTube Shorts Automation
======================================================

Handles YouTube API integration for automated video uploads:
- OAuth 2.0 authentication with token persistence
- Resumable video uploads
- Scheduled publishing support
- Metadata configuration (title, description, tags)
- Multi-channel support (enforces specific channel)

Cost: $0 (YouTube API is free, unlimited uploads)
"""

import os
import sys
import pickle
import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

# Setup logging
logger = logging.getLogger(__name__)

# Project directory
PROJECT_DIR = Path(__file__).parent
CREDENTIALS_DIR = PROJECT_DIR / "credentials"
CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = PROJECT_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    return {}


def get_token_filename(channel_id: str = None) -> str:
    """
    Get token filename, optionally per-channel.
    
    Args:
        channel_id: Optional channel ID to create channel-specific token
    
    Returns:
        Token filename (e.g., youtube_token_UC1234.pickle)
    """
    if channel_id:
        # Sanitize channel_id for filename
        safe_id = channel_id.replace("/", "_").replace("\\", "_")
        return f"youtube_token_{safe_id}.pickle"
    return "youtube_token.pickle"


class YouTubeUploader:
    """
    YouTube API client for automated video uploads.
    
    Features:
    - OAuth 2.0 authentication
    - Token persistence (auto-refresh)
    - Resumable uploads (handles large files)
    - Scheduled publishing
    - Metadata management
    - Multi-channel enforcement
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube.readonly'
    ]
    
    def __init__(
        self,
        credentials_file: str = None,
        token_file: str = None,
        target_channel_id: str = None
    ):
        """
        Initialize YouTube uploader.
        
        Args:
            credentials_file: Path to OAuth 2.0 client secrets JSON
            token_file: Path to store/load authentication token (auto-generated if None)
            target_channel_id: Required channel ID to enforce (from config if None)
        """
        if credentials_file:
            self.credentials_file = Path(credentials_file)
        else:
            self.credentials_file = CREDENTIALS_DIR / "client_secret.json"
        
        # Load config to get target channel
        self.config = load_config()
        self.target_channel_id = target_channel_id or self.config.get('youtube', {}).get('channel_id', '')
        
        # Use channel-specific token file
        if token_file:
            self.token_file = Path(token_file)
        elif self.target_channel_id:
            self.token_file = CREDENTIALS_DIR / get_token_filename(self.target_channel_id)
        else:
            self.token_file = CREDENTIALS_DIR / "youtube_token.pickle"

        # Legacy token fallback (root directory)
        if self.target_channel_id:
            self.legacy_token_file = PROJECT_DIR / get_token_filename(self.target_channel_id)
        else:
            self.legacy_token_file = PROJECT_DIR / "youtube_token.pickle"
        
        self.service = None
        self.authenticated_channel_id = None
        
        # Try to load existing credentials
        self._load_or_create_credentials()
    
    def _load_or_create_credentials(self):
        """Load existing credentials or prompt for new authentication."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
        except ImportError:
            raise ImportError(
                "Please install google-auth:\n"
                "pip install google-auth google-auth-oauthlib"
            )
        
        credentials = None
        
        # Load existing token if available (new location first, fallback to legacy)
        if self.token_file.exists():
            try:
                with open(self.token_file, 'rb') as token:
                    credentials = pickle.load(token)
                logger.info(f"✓ Loaded credentials from {self.token_file.name}")
            except Exception as e:
                logger.warning(f"Could not load token: {e}")
        elif self.legacy_token_file.exists():
            try:
                with open(self.legacy_token_file, 'rb') as token:
                    credentials = pickle.load(token)
                logger.info(f"✓ Loaded credentials from {self.legacy_token_file.name} (legacy)")
                # Save into new credentials folder for future runs
                self._save_credentials(credentials)
            except Exception as e:
                logger.warning(f"Could not load legacy token: {e}")
        
        # Refresh expired credentials
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                logger.info("✓ Refreshed YouTube credentials")
                self._save_credentials(credentials)
            except Exception as e:
                logger.warning(f"Could not refresh token: {e}")
                credentials = None
        
        # Create new credentials if needed
        if not credentials or not credentials.valid:
            if not self.credentials_file.exists():
                logger.warning(
                    f"OAuth credentials file not found: {self.credentials_file}\n"
                    "Run 'python main.py --auth' to authenticate"
                )
                return
            
            logger.info("Authentication required. Run 'python main.py --auth'")
            return
        
        # Build YouTube service
        self._build_service(credentials)
    
    def authenticate(self) -> bool:
        """
        Perform OAuth 2.0 authentication flow.
        Opens browser for user to authorize access.
        
        Returns:
            True if authentication successful and channel matches, False otherwise
        """
        # If running outside the project's venv, re-run auth using venv Python
        if os.environ.get("YOUTUBE_AUTH_CHILD") != "1":
            venv_python = PROJECT_DIR / "venv" / "bin" / "python"
            if venv_python.exists() and sys.prefix == sys.base_prefix:
                logger.warning("Not running inside venv. Re-running auth with venv Python...")
                try:
                    import subprocess
                    env = os.environ.copy()
                    env["YOUTUBE_AUTH_CHILD"] = "1"
                    result = subprocess.run(
                        [str(venv_python), str(PROJECT_DIR / "main.py"), "--auth"],
                        env=env
                    )
                    if result.returncode == 0:
                        # Reload credentials that were saved by the child process
                        self._load_or_create_credentials()
                        return self.is_authenticated()
                    return False
                except Exception as e:
                    logger.error(f"Failed to re-run auth with venv Python: {e}")
                    return False

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            # Attempt to install into the current interpreter, then retry import
            try:
                import subprocess
                logger.info("google-auth-oauthlib missing. Installing into current Python...")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "google-auth-oauthlib"],
                    check=True
                )
                from google_auth_oauthlib.flow import InstalledAppFlow
            except Exception:
                raise ImportError(
                    "Please install google-auth-oauthlib in the active Python env:\n"
                    "  source venv/bin/activate\n"
                    "  python -m pip install google-auth-oauthlib"
                )
        
        if not self.credentials_file.exists():
            logger.error(
                f"OAuth credentials file not found: {self.credentials_file}\n\n"
                "To get client_secret.json:\n"
                "1. Go to https://console.cloud.google.com/\n"
                "2. Create a project (or select existing)\n"
                "3. Enable 'YouTube Data API v3'\n"
                "4. Go to 'Credentials' → 'Create Credentials' → 'OAuth 2.0 Client IDs'\n"
                "5. Choose 'Desktop application'\n"
                "6. Download JSON and rename to 'client_secret.json'\n"
                "7. Place in project directory"
            )
            return False
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_file),
                self.SCOPES
            )
            
            logger.info("Opening browser for YouTube authorization...")
            if self.target_channel_id:
                logger.info(f"⚠️  Select channel: {self.target_channel_id}")
            
            credentials = flow.run_local_server(
                port=8080,
                prompt='consent',
                success_message="Authentication successful! You can close this window."
            )
            
            # Build service temporarily to check channel
            self._build_service(credentials)
            
            # Verify channel matches configuration
            if not self._verify_channel():
                return False
            
            # Save credentials with channel-specific filename
            self._save_credentials(credentials)
            
            logger.info("✓ YouTube authentication successful!")
            return True
        
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def _verify_channel(self) -> bool:
        """
        Verify authenticated channel matches configured target.
        
        Returns:
            True if channel matches or no target configured, False otherwise
        """
        if not self.service:
            return False
        
        # Get authenticated channel ID
        channel_info = self.get_channel_info()
        if not channel_info:
            logger.error("Could not retrieve channel info after authentication")
            return False
        
        self.authenticated_channel_id = channel_info['id']
        
        # If no target configured, accept any channel
        if not self.target_channel_id:
            logger.info(f"✓ Authenticated as channel: {self.authenticated_channel_id}")
            logger.info(f"  Channel name: {channel_info['title']}")
            return True
        
        # Check if channels match
        if self.authenticated_channel_id != self.target_channel_id:
            logger.error(
                f"\n{'='*60}\n"
                f"CHANNEL MISMATCH!\n"
                f"{'='*60}\n"
                f"Authenticated as: {self.authenticated_channel_id}\n"
                f"Config expects:   {self.target_channel_id}\n"
                f"\n"
                f"You authenticated with the wrong channel.\n"
                f"Please re-run 'python main.py --auth' and select the correct channel.\n"
                f"{'='*60}"
            )
            
            # Delete the token file to force re-auth
            if self.token_file.exists():
                self.token_file.unlink()
                logger.info(f"Deleted token file: {self.token_file.name}")
            
            self.service = None
            return False
        
        logger.info(f"✓ Channel verified: {channel_info['title']} ({self.authenticated_channel_id})")
        return True
    
    def _save_credentials(self, credentials):
        """Save credentials to pickle file."""
        with open(self.token_file, 'wb') as token:
            pickle.dump(credentials, token)
        logger.info(f"✓ Credentials saved to {self.token_file.name}")
    
    def _build_service(self, credentials):
        """Build YouTube API service."""
        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Please install google-api-python-client:\n"
                "pip install google-api-python-client"
            )
        
        self.service = build('youtube', 'v3', credentials=credentials)
        logger.debug("YouTube API service initialized")
    
    def is_authenticated(self) -> bool:
        """Check if uploader is authenticated."""
        return self.service is not None
    
    def verify_before_upload(self) -> bool:
        """
        Verify channel matches before upload.
        Call this before any upload operation.
        
        Returns:
            True if ready to upload, False if channel mismatch
        """
        if not self.is_authenticated():
            logger.error("Not authenticated! Run 'python main.py --auth' first")
            return False
        
        # If we already verified, skip
        if self.authenticated_channel_id:
            if self.target_channel_id and self.authenticated_channel_id != self.target_channel_id:
                logger.error(f"Channel mismatch! Run 'python main.py --auth' to re-authenticate")
                return False
            return True
        
        # Verify now
        return self._verify_channel()
    
    def upload_video(
        self,
        video_file: str,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "22",  # People & Blogs
        privacy_status: str = "public"
    ) -> Optional[Dict[str, Any]]:
        """
        Upload video to YouTube.
        
        Args:
            video_file: Path to video file
            title: Video title
            description: Video description
            tags: List of tags
            category_id: YouTube category ID (22 = People & Blogs)
            privacy_status: 'public', 'private', or 'unlisted'
        
        Returns:
            Dictionary with upload result, or None if upload fails
        """
        # Verify channel before upload
        if not self.verify_before_upload():
            return None
        
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            raise ImportError(
                "Please install google-api-python-client:\n"
                "pip install google-api-python-client"
            )
        
        video_path = Path(video_file)
        if not video_path.exists():
            logger.error(f"Video file not found: {video_file}")
            return None
        
        if tags is None:
            tags = ['story', 'emotional', 'viral', 'shorts']
        
        # If description is empty, use the title
        if not description:
            description = title
        
        # Video metadata
        body = {
            'snippet': {
                'title': title[:100],  # YouTube title limit
                'description': description[:5000],  # YouTube description limit
                'tags': tags[:500],  # Tag limit
                'categoryId': category_id,
                'defaultLanguage': 'en'
            },
            'status': {
                'privacyStatus': privacy_status,
                'madeForKids': False,
                'selfDeclaredMadeForKids': False
            }
        }
        
        # Media upload with resumable support
        media = MediaFileUpload(
            str(video_path),
            mimetype='video/mp4',
            resumable=True,
            chunksize=10 * 1024 * 1024  # 10MB chunks
        )
        
        try:
            logger.info(f"Uploading: {video_path.name}")
            
            # Create upload request
            request = self.service.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )
            
            # Execute upload with progress tracking
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")
            
            video_id = response['id']
            video_url = f"https://youtube.com/shorts/{video_id}"
            
            result = {
                'videoId': video_id,
                'title': title,
                'url': video_url,
                'uploadedAt': response['snippet'].get('publishedAt', datetime.now().isoformat()),
                'status': response['status']['privacyStatus'],
                'channelId': self.authenticated_channel_id
            }
            
            logger.info(f"✓ Upload complete: {video_url}")
            return result
        
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return None
    
    def schedule_video(
        self,
        video_file: str,
        title: str,
        description: str,
        publish_time: datetime,
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Upload video and schedule for future publishing.
        
        Args:
            video_file: Path to video file
            title: Video title
            description: Video description
            publish_time: When to publish (must be in future)
            tags: List of tags
        
        Returns:
            Video ID if successful, None otherwise
        """
        # Verify channel before upload
        if not self.verify_before_upload():
            return None
        
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError:
            raise ImportError(
                "Please install google-api-python-client:\n"
                "pip install google-api-python-client"
            )
        
        if tags is None:
            tags = ['story', 'emotional', 'viral']
        
        # Video metadata with scheduled publish
        body = {
            'snippet': {
                'title': title[:100],
                'description': description[:5000],
                'tags': tags,
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
        
        try:
            request = self.service.videos().insert(
                part='snippet,status',
                body=body,
                media_body=media
            )
            
            response = None
            while response is None:
                status, response = request.next_chunk()
            
            video_id = response['id']
            logger.info(f"✓ Scheduled for {publish_time}: {video_id}")
            return video_id
        
        except Exception as e:
            logger.error(f"Scheduled upload failed: {str(e)}")
            return None
    
    def get_channel_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the authenticated channel.
        
        Returns:
            Dictionary with channel info, or None if request fails
        """
        if not self.service:
            return None
        
        try:
            response = self.service.channels().list(
                part='snippet,statistics',
                mine=True
            ).execute()
            
            if response.get('items'):
                channel = response['items'][0]
                return {
                    'id': channel['id'],
                    'title': channel['snippet']['title'],
                    'subscriberCount': channel['statistics'].get('subscriberCount', 0),
                    'videoCount': channel['statistics'].get('videoCount', 0),
                    'viewCount': channel['statistics'].get('viewCount', 0)
                }
        except Exception as e:
            logger.error(f"Could not get channel info: {e}")
        
        return None
    
    def get_authenticated_channel_id(self) -> Optional[str]:
        """
        Get the channel ID from the current authentication.
        
        Returns:
            Channel ID string or None
        """
        if self.authenticated_channel_id:
            return self.authenticated_channel_id
        
        channel_info = self.get_channel_info()
        if channel_info:
            self.authenticated_channel_id = channel_info['id']
            return self.authenticated_channel_id
        
        return None
    
    def get_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a specific video.
        
        Args:
            video_id: YouTube video ID
        
        Returns:
            Dictionary with video stats, or None if request fails
        """
        if not self.is_authenticated():
            return None
        
        try:
            response = self.service.videos().list(
                part='statistics,snippet',
                id=video_id
            ).execute()
            
            if response.get('items'):
                video = response['items'][0]
                stats = video['statistics']
                return {
                    'videoId': video_id,
                    'title': video['snippet']['title'],
                    'views': int(stats.get('viewCount', 0)),
                    'likes': int(stats.get('likeCount', 0)),
                    'comments': int(stats.get('commentCount', 0))
                }
        except Exception as e:
            logger.error(f"Could not get video stats: {e}")
        
        return None


# Convenience function for quick uploads
def upload_to_youtube(
    video_file: str,
    question: str,
    channel_id: str = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to upload a video to YouTube.
    
    Args:
        video_file: Path to the video file
        question: Story question (used as title and description)
        channel_id: Target channel ID (uses config if not specified)
    
    Returns:
        Upload result dictionary or None if failed
    """
    uploader = YouTubeUploader(target_channel_id=channel_id)
    config = load_config()
    youtube_config = config.get('youtube', {})
    default_tags = youtube_config.get('default_tags', ['story', 'emotional', 'viral', 'shorts'])
    shorts_mode = youtube_config.get('shorts_mode', True)
    shorts_hashtags = youtube_config.get('shorts_hashtags', ['#shorts'])
    
    title = question
    description = question
    tags = list(default_tags)

    # Enforce Shorts metadata
    if shorts_mode:
        # Add hashtags to title/description if missing
        for tag in shorts_hashtags:
            if tag.lower() not in title.lower():
                title = f"{title} {tag}"
            if tag.lower() not in description.lower():
                description = f"{description} {tag}"
        # Ensure "shorts" tag exists
        if 'shorts' not in [t.lower() for t in tags]:
            tags.append('shorts')
    
    if not uploader.is_authenticated():
        logger.info("YouTube not authenticated. Attempting interactive auth...")
        if not uploader.authenticate():
            logger.error(
                "YouTube not authenticated!\n"
                "Run: python main.py --auth"
            )
            return None
    
    return uploader.upload_video(
        video_file=video_file,
        title=title,
        description=description,
        tags=tags
    )


def whoami() -> Optional[Dict[str, Any]]:
    """
    Get the currently authenticated channel info.
    
    Returns:
        Channel info dictionary or None
    """
    uploader = YouTubeUploader()
    
    if not uploader.is_authenticated():
        return None
    
    return uploader.get_channel_info()


def test_youtube_connection():
    """Test YouTube API connection."""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*50)
    print("YouTube Uploader Test")
    print("="*50 + "\n")
    
    config = load_config()
    target_channel = config.get('youtube', {}).get('channel_id', '')
    
    if target_channel:
        print(f"Target channel (from config): {target_channel}\n")
    
    uploader = YouTubeUploader()
    
    if uploader.is_authenticated():
        print("✓ YouTube API connected!")
        
        # Get channel info
        channel = uploader.get_channel_info()
        if channel:
            print(f"\nAuthenticated Channel:")
            print(f"  ID: {channel['id']}")
            print(f"  Name: {channel['title']}")
            print(f"  Subscribers: {channel['subscriberCount']}")
            print(f"  Videos: {channel['videoCount']}")
            print(f"  Total Views: {channel['viewCount']}")
            
            if target_channel and channel['id'] != target_channel:
                print(f"\n⚠️  WARNING: Authenticated channel does not match config!")
                print(f"   Run 'python main.py --auth' to re-authenticate")
    else:
        print("✗ Not authenticated")
        print("\nTo authenticate:")
        print("1. Ensure client_secret.json is in the project directory")
        print("2. Run: python main.py --auth")


if __name__ == "__main__":
    test_youtube_connection()
