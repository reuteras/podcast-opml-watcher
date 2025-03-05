#!/usr/bin/env python3
"""
podcast_opml_watcher.py

A companion tool for podcast2md that:
1. Reads podcast feeds from an OPML file
2. Watches for new episodes
3. Automatically transcribes them using podcast2md
4. Organizes transcripts in a structured folder hierarchy
"""

import os
import sys
import json
import time
import argparse
import subprocess
import logging
import xml.etree.ElementTree as ET
import feedparser
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
import re
import shutil
import tempfile
import urllib.request
import requests
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('podcast_watcher.log')
    ]
)
logger = logging.getLogger('podcast_watcher')

# Default configuration
DEFAULT_CONFIG = {
    "check_interval": 3600,  # Check feeds every hour by default
    "output_root": "Podcasts",
    "state_file": "podcast_watcher_state.json",
    "max_concurrent_downloads": 2,
    "podcast2md_path": "podcast2md",
    "podcast2md_model": "base",
    "apply_formatting": True,
    "create_links": True,
    "create_refs": True,
    "max_retry_attempts": 3,
    "retry_delay": 300,  # 5 minutes
}

def parse_opml(opml_file):
    """Parse OPML file and extract podcast feeds."""
    tree = ET.parse(opml_file)
    root = tree.getroot()

    feeds = []
    # Look for podcast feeds in OPML format
    for outline in root.findall(".//outline"):
        # Check if it's a podcast feed (has xmlUrl attribute)
        xml_url = outline.get('xmlUrl')
        if xml_url:
            feed_info = {
                'xml_url': xml_url,
                'title': outline.get('title') or outline.get('text', 'Unknown Podcast'),
                'html_url': outline.get('htmlUrl', ''),
                'description': outline.get('description', '')
            }
            feeds.append(feed_info)

    logger.info(f"Found {len(feeds)} podcast feeds in OPML file")
    return feeds

def load_state(state_file):
    """Load the state of previously transcribed episodes."""
    if not os.path.exists(state_file):
        return {"feeds": {}, "last_check": None}

    try:
        with open(state_file, 'r') as f:
            state = json.load(f)
            logger.info(f"Loaded state with {len(state.get('feeds', {}))} feeds")
            return state
    except json.JSONDecodeError:
        logger.warning(f"Could not parse state file {state_file}, starting with empty state")
        return {"feeds": {}, "last_check": None}

def save_state(state, state_file):
    """Save the current state to the state file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(state_file)), exist_ok=True)

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved state to {state_file}")

def get_episode_identifier(episode):
    """Generate a unique identifier for an episode based on its content."""
    # Try to use the episode GUID if available
    if hasattr(episode, 'id'):
        return episode.id

    # Fall back to title + publication date
    identifier = f"{episode.title}_{getattr(episode, 'published', '')}"
    return hashlib.md5(identifier.encode()).hexdigest()

def sanitize_filename(name):
    """Sanitize the given name for use as a filename."""
    # Replace any character that isn't alphanumeric, dash, underscore, or dot with an underscore
    sanitized = re.sub(r'[^\w\-\.]', '_', name)
    # Limit length to avoid path too long errors
    return sanitized[:100]

def get_feed_episodes(feed_url):
    """Get all episodes from a feed URL."""
    try:
        parsed_feed = feedparser.parse(feed_url)

        if parsed_feed.bozo and hasattr(parsed_feed, 'bozo_exception'):
            logger.warning(f"Feed parsing error for {feed_url}: {parsed_feed.bozo_exception}")

        if not hasattr(parsed_feed, 'entries') or not parsed_feed.entries:
            logger.warning(f"No entries found in feed: {feed_url}")
            return [], parsed_feed.feed

        logger.info(f"Found {len(parsed_feed.entries)} episodes in feed: {feed_url}")
        return parsed_feed.entries, parsed_feed.feed

    except Exception as e:
        logger.error(f"Error fetching feed {feed_url}: {str(e)}")
        return [], {}

def get_audio_url(episode):
    """Extract the audio URL from a podcast episode."""
    # Look for enclosures (standard podcast format)
    if hasattr(episode, 'enclosures') and episode.enclosures:
        for enclosure in episode.enclosures:
            if 'url' in enclosure and 'type' in enclosure and enclosure['type'].startswith('audio/'):
                return enclosure['url']

        # If no audio enclosure was found, take the first URL as fallback
        if 'url' in episode.enclosures[0]:
            return episode.enclosures[0]['url']

    # Look for media content (alternative format)
    if hasattr(episode, 'media_content') and episode.media_content:
        for media in episode.media_content:
            if 'url' in media and 'type' in media and media['type'].startswith('audio/'):
                return media['url']

        # Fallback to first media URL
        if 'url' in episode.media_content[0]:
            return media.media_content[0]['url']

    # Last resort: look for links
    if hasattr(episode, 'links'):
        for link in episode.links:
            if 'type' in link and link['type'].startswith('audio/') and 'href' in link:
                return link['href']

    # No audio URL found
    return None

def get_episode_url(episode):
    """Extract the episode webpage URL."""
    # Try to get the link to the episode page
    if hasattr(episode, 'link'):
        return episode.link

    # Look through links for alternatives
    if hasattr(episode, 'links'):
        for link in episode.links:
            if 'rel' in link and link['rel'] == 'alternate' and 'href' in link:
                return link['href']
            # Look for HTML links
            if 'type' in link and link['type'] == 'text/html' and 'href' in link:
                return link['href']

    # No episode URL found
    return None

def get_episode_date(episode):
    """Get the publication date of an episode."""
    for date_field in ['published_parsed', 'updated_parsed', 'created_parsed']:
        if hasattr(episode, date_field) and getattr(episode, date_field):
            time_struct = getattr(episode, date_field)
            return datetime(*time_struct[:6])

    # If no date is found, use current time
    return datetime.now()

def download_episode(episode, temp_dir):
    """Download an episode to a temporary location."""
    audio_url = get_audio_url(episode)
    if not audio_url:
        logger.error(f"No audio URL found for episode: {episode.title}")
        return None

    try:
        # Parse URL to get the filename
        parsed_url = urlparse(audio_url)
        file_name = os.path.basename(parsed_url.path)

        # If no filename in URL, create one based on episode title
        if not file_name or '.' not in file_name:
            ext = '.mp3'  # Default extension

            # Try to determine extension from content type
            if hasattr(episode, 'enclosures') and episode.enclosures:
                content_type = episode.enclosures[0].get('type', '')
                if content_type == 'audio/mpeg':
                    ext = '.mp3'
                elif content_type == 'audio/x-m4a' or content_type == 'audio/mp4':
                    ext = '.m4a'
                elif content_type == 'audio/ogg':
                    ext = '.ogg'

            file_name = f"{sanitize_filename(episode.title)}{ext}"

        # Full path to temporary file
        temp_file_path = os.path.join(temp_dir, file_name)

        logger.info(f"Downloading {audio_url} to {temp_file_path}")

        # Use requests for potentially better error handling
        with requests.get(audio_url, stream=True) as response:
            response.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info(f"Successfully downloaded episode to {temp_file_path}")
        return temp_file_path

    except Exception as e:
        logger.error(f"Error downloading episode {episode.title}: {str(e)}")
        return None

def transcribe_episode(audio_file, output_dir, podcast_title, episode_title, episode_date, config, episode_url=None):
    """Transcribe an episode using podcast2md."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create images directory if it doesn't exist
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Prepare filename for the markdown file
        md_filename = sanitize_filename(episode_title)
        output_file = os.path.join(output_dir, f"{md_filename}.md")

        # Prepare podcast2md command
        cmd = [
            config["podcast2md_path"],
            audio_file,
            "--model", config["podcast2md_model"],
            "--output", output_file
        ]

        # Add optional formatting flags
        if config.get("apply_formatting", True):
            cmd.append("--format")
        if config.get("create_links", True):
            cmd.append("--links")
        if config.get("create_refs", True):
            cmd.append("--refs")
        if config.get("vault_path"):
            cmd.extend(["--vault", config["vault_path"]])

        # Add episode URL if available
        if episode_url:
            cmd.extend(["--episode-url", episode_url])

        logger.info(f"Transcribing episode: {episode_title}")
        logger.debug(f"Running command: {' '.join(cmd)}")

        # Run podcast2md
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Error transcribing episode: {result.stderr}")
            return None

        logger.info(f"Successfully transcribed episode to {output_file}")

        # Move any image files to the images directory
        md_dir = os.path.dirname(output_file)
        for file in os.listdir(md_dir):
            if file.endswith(('.jpg', '.jpeg', '.png', '.gif')) and os.path.isfile(os.path.join(md_dir, file)):
                src_path = os.path.join(md_dir, file)
                dst_path = os.path.join(images_dir, file)
                shutil.move(src_path, dst_path)

                # Update image references in the markdown file
                with open(output_file, 'r') as f:
                    content = f.read()

                # Replace image references
                updated_content = re.sub(
                    r'!\[([^\]]*)\]\(([^)]+)\)',
                    lambda m: f'![{m.group(1)}](images/{os.path.basename(m.group(2))})',
                    content
                )

                with open(output_file, 'w') as f:
                    f.write(updated_content)

                logger.info(f"Moved image {file} to images directory and updated references")

        return output_file

    except Exception as e:
        logger.error(f"Error during transcription process: {str(e)}")
        return None

def process_episode(feed_info, episode, config, temp_dir, state):
    """Process a single episode: download and transcribe."""
    episode_id = get_episode_identifier(episode)

    # Check if this episode has already been processed
    feed_url = feed_info['xml_url']
    if feed_url in state['feeds'] and episode_id in state['feeds'][feed_url]['processed_episodes']:
        logger.info(f"Episode already processed: {episode.title}")
        return False

    # Get episode title and date
    episode_title = episode.title if hasattr(episode, 'title') else "Unknown Episode"
    episode_date = get_episode_date(episode)

    # Get episode URL if available
    episode_url = get_episode_url(episode)

    # Create output directory structure
    podcast_dir_name = sanitize_filename(feed_info['title'])
    podcast_dir = os.path.join(config['output_root'], podcast_dir_name)

    # Download the episode
    for attempt in range(config['max_retry_attempts']):
        try:
            audio_file = download_episode(episode, temp_dir)
            if not audio_file:
                logger.error(f"Failed to download episode: {episode_title}")
                return False

            # Transcribe the episode
            md_file = transcribe_episode(
                audio_file,
                podcast_dir,
                feed_info['title'],
                episode_title,
                episode_date,
                config,
                episode_url
            )

            if not md_file:
                logger.error(f"Failed to transcribe episode: {episode_title}")
                return False

            # Update state to mark this episode as processed
            if feed_url not in state['feeds']:
                state['feeds'][feed_url] = {
                    'title': feed_info['title'],
                    'processed_episodes': {}
                }

            state['feeds'][feed_url]['processed_episodes'][episode_id] = {
                'title': episode_title,
                'date': episode_date.isoformat(),
                'output_file': md_file,
                'processed_at': datetime.now().isoformat(),
                'episode_url': episode_url
            }

            # Success, no need to retry
            logger.info(f"Successfully processed episode: {episode_title}")
            return True

        except Exception as e:
            logger.error(f"Error processing episode (attempt {attempt+1}/{config['max_retry_attempts']}): {str(e)}")

            if attempt < config['max_retry_attempts'] - 1:
                logger.info(f"Retrying in {config['retry_delay']} seconds...")
                time.sleep(config['retry_delay'])

    return False

def process_feed(feed_info, state, config, initial_limit=None):
    """Process all new episodes from a feed."""
    feed_url = feed_info['xml_url']
    logger.info(f"Processing feed: {feed_info['title']} ({feed_url})")

    # Get all episodes from the feed
    episodes, feed_data = get_feed_episodes(feed_url)
    if not episodes:
        logger.warning(f"No episodes found for feed: {feed_url}")
        return 0

    # Update feed info with additional data from the parsed feed
    if hasattr(feed_data, 'title'):
        feed_info['title'] = feed_data.title
    if hasattr(feed_data, 'link'):
        feed_info['html_url'] = feed_data.link
    if hasattr(feed_data, 'description'):
        feed_info['description'] = feed_data.description

    # Sort episodes by date (newest first)
    episodes.sort(key=lambda ep: get_episode_date(ep), reverse=True)

    # Limit to specified number of episodes for initial run
    if initial_limit and initial_limit.lower() != 'all':
        try:
            limit = int(initial_limit)
            episodes = episodes[:limit]
            logger.info(f"Limited to {limit} most recent episodes for initial run")
        except ValueError:
            logger.warning(f"Invalid initial limit '{initial_limit}', processing all episodes")

    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_count = 0

        # Create a thread pool for parallel downloads and transcription
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['max_concurrent_downloads']) as executor:
            futures = []

            for episode in episodes:
                future = executor.submit(
                    process_episode,
                    feed_info,
                    episode,
                    config,
                    temp_dir,
                    state
                )
                futures.append(future)

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    if future.result():
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing episode: {str(e)}")

    logger.info(f"Processed {processed_count} new episodes from feed: {feed_info['title']}")
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Watch OPML file for new podcast episodes and transcribe them")
    parser.add_argument("opml_file", help="Path to the OPML file containing podcast feeds")
    parser.add_argument("--config", default="podcast_watcher_config.json", help="Path to configuration file")
    parser.add_argument("--initial", default=None, help="Number of existing episodes to transcribe on first run (number or 'all')")
    parser.add_argument("--state-file", default=None, help="Path to state file (overrides config file)")
    parser.add_argument("--output-dir", default=None, help="Root directory for output (overrides config file)")
    parser.add_argument("--once", action="store_true", help="Run once and exit (don't continue watching feeds)")
    parser.add_argument("--interval", type=int, default=None, help="Check interval in seconds (overrides config file)")
    parser.add_argument("--podcast2md", default=None, help="Path to podcast2md executable")
    parser.add_argument("--model", default=None, choices=["tiny", "base", "small", "medium", "large"], help="Whisper model to use")
    parser.add_argument("--skip-formatting", action="store_true", help="Skip applying text formatting")
    parser.add_argument("--skip-links", action="store_true", help="Skip creating wiki links")
    parser.add_argument("--skip-refs", action="store_true", help="Skip adding references")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # Load configuration
    config = DEFAULT_CONFIG.copy()

    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
                logger.info(f"Loaded configuration from {args.config}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse config file {args.config}, using defaults")

    # Override config with command-line arguments
    if args.state_file:
        config['state_file'] = args.state_file
    if args.output_dir:
        config['output_root'] = args.output_dir
    if args.interval:
        config['check_interval'] = args.interval
    if args.podcast2md:
        config['podcast2md_path'] = args.podcast2md
    if args.model:
        config['podcast2md_model'] = args.model

    # Update formatting flags
    if args.skip_formatting:
        config['apply_formatting'] = False
    if args.skip_links:
        config['create_links'] = False
    if args.skip_refs:
        config['create_refs'] = False

    # Ensure output directory exists
    os.makedirs(config['output_root'], exist_ok=True)

    # Load state
    state = load_state(config['state_file'])

    # Parse OPML file
    try:
        feeds = parse_opml(args.opml_file)
        if not feeds:
            logger.error("No podcast feeds found in the OPML file")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error parsing OPML file: {str(e)}")
        sys.exit(1)

    # Process feeds and update state
    total_processed = 0

    try:
        for feed_info in feeds:
            processed = process_feed(feed_info, state, config, args.initial)
            total_processed += processed

            # Save state after each feed to avoid losing progress
            state['last_check'] = datetime.now().isoformat()
            save_state(state, config['state_file'])

        logger.info(f"Completed processing {len(feeds)} feeds, {total_processed} new episodes transcribed")

        # If --once flag is set, exit after first run
        if args.once:
            logger.info("Exiting after one-time run (--once flag set)")
            return

        # Continue watching feeds for new episodes
        logger.info(f"Continuing to watch feeds, checking every {config['check_interval']} seconds")

        while True:
            time.sleep(config['check_interval'])

            logger.info("Checking feeds for new episodes...")
            new_total = 0

            for feed_info in feeds:
                processed = process_feed(feed_info, state, config)
                new_total += processed

                # Save state after each feed
                state['last_check'] = datetime.now().isoformat()
                save_state(state, config['state_file'])

            logger.info(f"Completed check, {new_total} new episodes transcribed")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Final state save on error
        state['last_check'] = datetime.now().isoformat()
        save_state(state, config['state_file'])
        sys.exit(1)

    # Final state save on clean exit
    state['last_check'] = datetime.now().isoformat()
    save_state(state, config['state_file'])

if __name__ == "__main__":
    main()
