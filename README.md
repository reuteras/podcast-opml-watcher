# Podcast OPML Watcher

A companion tool for podcast2md that automatically watches podcast feeds from an OPML file and transcribes new episodes.

## Overview

Podcast OPML Watcher is designed to work alongside [podcast2md](https://github.com/reuteras/podcast2md) to automate the process of transcribing podcast episodes. It reads an OPML file containing podcast feed URLs, monitors these feeds for new episodes, and automatically transcribes them using podcast2md.

## Features

- **OPML Import**: Process standard OPML subscription files exported from podcast apps
- **Feed Monitoring**: Continuously watch feeds for new episodes
- **Automated Transcription**: Leverage podcast2md for high-quality transcriptions
- **Structured Organization**:
  - Create a structured folder hierarchy for your podcast transcripts
  - Root "Podcasts" folder with subfolders for each podcast
  - Store images in dedicated "images" subfolders
- **State Management**: Keep track of transcribed episodes to avoid duplicates
- **Parallel Processing**: Download and transcribe multiple episodes concurrently
- **Configurable**: Extensive configuration options via command line or config file
- **Initial Backfill**: Option to transcribe a specific number of existing episodes on first run

## Prerequisites

- Python 3.12+
- [podcast2md](https://github.com/reuteras/podcast2md) installed and configured
- FFmpeg (required for audio processing)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

No installation required! Simply use `uvx` to run the tool directly:

```bash
alias podcast-opml-watcher="uvx --with git+https://github.com/reuteras/podcast-opml-watcher podcast-opml-watcher"
```

Or install the package using uv:

```bash
uv pip install git+https://github.com/reuteras/podcast-opml-watcher
```

## Usage

Watch an OPML file for new podcast episodes:

```bash
podcast-opml-watcher podcasts.opml
```

All other behavior is controlled by flags (each overrides the config file for that run):

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `podcast_watcher_config.json` | Path to configuration file |
| `--initial N \| all` | (none) | Transcribe N most recent episodes per feed, or all, on first run |
| `--once` | off | Process feeds once and exit, instead of watching continuously |
| `--output-dir DIR` | from config | Root directory for output |
| `--interval SECONDS` | from config | How often to check feeds |
| `--model NAME` | from config | Whisper model: `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large` |
| `--skip-formatting` | off | Disable text formatting |
| `--skip-links` | off | Disable wiki links |
| `--skip-refs` | off | Disable references |
| `--state-file PATH` | from config | Path to state file |
| `--podcast2md PATH` | from config | Path to the podcast2md executable |
| `--debug` | off | Enable debug logging |

Example: `podcast-opml-watcher podcasts.opml --once --model medium --interval 7200`

## Configuration File

Create a `podcast_watcher_config.json` file to store your preferences:

```json
{
  "check_interval": 3600,
  "output_root": "Podcasts",
  "state_file": "podcast_watcher_state.json",
  "max_concurrent_downloads": 2,
  "podcast2md_path": "podcast2md",
  "podcast2md_model": "base",
  "apply_formatting": true,
  "create_links": true,
  "create_refs": true,
  "max_retry_attempts": 3,
  "retry_delay": 300,
  "vault_path": null
}
```

Specify a custom config file:

```bash
podcast-opml-watcher podcasts.opml --config my_config.json
```

## Output Structure

The tool creates the following directory structure:

```bash
Podcasts/
├── Podcast_Name_1/
│   ├── images/
│   │   └── episode_cover.jpg
│   ├── Episode_Title_1.md
│   └── Episode_Title_2.md
├── Podcast_Name_2/
│   ├── images/
│   │   └── episode_cover.jpg
│   ├── Episode_Title_1.md
│   └── Episode_Title_2.md
└── ...
```

## State Management

The tool maintains a state file (`podcast_watcher_state.json` by default) to track which episodes have been processed. This prevents duplicate downloads and transcriptions.

## Integration with Obsidian

For Obsidian users, set `vault_path` in the config file to your vault's path. The watcher passes it through to podcast2md as `--vault`, so transcripts are written directly into your vault using podcast2md's Obsidian integration (wiki links, references, etc.):

```json
{
  "vault_path": "/path/to/your/Obsidian/Vault"
}
```
