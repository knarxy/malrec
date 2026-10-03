"""Personalised anime recommendations built from a MyAnimeList profile."""
import logging

__version__ = "0.1.0"

# httpx logs every request at INFO, which drowns out ingest progress.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
