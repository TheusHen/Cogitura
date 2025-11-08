"""Dictionary aggregation module.

Provides multiple open dictionary sources with lightweight scraping / API access
and a unified interface. Designed to be polite (custom User-Agent) and simple enough
for unit testing with mocks.

Sources implemented:
- Wiktionary (HTML scrape minimal definition list)
- Datamuse API (https://api.datamuse.com/words)
- Free Dictionary API (https://api.dictionaryapi.dev/)
- Wordnik API (if WORDNIK_API_KEY env var present) (https://developer.wordnik.com/)
- WordNet (NLTK) if available

Functions return lists of definition strings (may be empty). Errors are swallowed
and logged; returning [] keeps downstream robust.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import re
import time
import json
import requests
from html import unescape
from pathlib import Path
from cogitura.logger import log

USER_AGENT = "CogituraBot/1.0 (+https://github.com/TheusHen/Cogitura)"
REQUEST_TIMEOUT = 8
BACKOFF_SLEEP = 0.5

class DictionarySourceError(Exception):
    pass

# --- Helpers -----------------------------------------------------------------

def _safe_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Optional[requests.Response]:
    if headers is None:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/html"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 429:  # rate limit
            time.sleep(BACKOFF_SLEEP)
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception as e:
        log.warning(f"Dictionary request failed for {url}: {e}")
        return None

# --- Wiktionary --------------------------------------------------------------

def fetch_wiktionary(word: str, language: str = "English") -> List[str]:
    """Fetch definitions from Wiktionary via simple HTML scraping.
    We intentionally keep parsing minimal to reduce fragility.
    """
    url = f"https://en.wiktionary.org/wiki/{word}"  # Assumes English page
    resp = _safe_get(url)
    if not resp:
        return []
    html = resp.text
    # Capture list item lines inside definition sections
    pattern = re.compile(r"<li>(.*?)</li>", re.IGNORECASE | re.DOTALL)
    raw_items = pattern.findall(html)
    definitions: List[str] = []
    for item in raw_items:
        # Remove tags
        clean = re.sub(r"<.*?>", "", item)
        clean = unescape(clean).strip()
        # Filter out navigation / example sentences heuristically
        if len(clean.split()) < 2:
            continue
        if 'Wiktionary' in clean or 'HTML' in clean:
            continue
        definitions.append(clean)
        if len(definitions) >= 5:  # limit noise
            break
    return definitions

# --- Datamuse ----------------------------------------------------------------

def fetch_datamuse(word: str) -> List[str]:
    url = "https://api.datamuse.com/words"
    resp = _safe_get(url, params={"sp": word, "md": "d"})
    if not resp:
        return []
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return []
    defs: List[str] = []
    for entry in data:
        for d in entry.get("defs", []) or []:
            # Format is "pos\tdefinition"
            parts = d.split('\t', 1)
            if len(parts) == 2:
                defs.append(parts[1].strip())
        if len(defs) >= 5:
            break
    return defs

# --- Free Dictionary API -----------------------------------------------------

def fetch_free_dictionary(word: str) -> List[str]:
    url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    resp = _safe_get(url)
    if not resp:
        return []
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return []
    definitions: List[str] = []
    if isinstance(data, list):
        for entry in data:
            for meaning in entry.get("meanings", []):
                for d in meaning.get("definitions", []):
                    definition = d.get("definition")
                    if definition:
                        definitions.append(definition.strip())
                    if len(definitions) >= 5:
                        break
                if len(definitions) >= 5:
                    break
            if len(definitions) >= 5:
                break
    return definitions

# --- Wordnik -----------------------------------------------------------------

def fetch_wordnik(word: str) -> List[str]:
    api_key = os.getenv("WORDNIK_API_KEY")
    if not api_key:
        return []
    url = f"https://api.wordnik.com/v4/word.json/{word}/definitions"
    resp = _safe_get(url, params={"limit": 5, "api_key": api_key})
    if not resp:
        return []
    try:
        data = resp.json()
    except json.JSONDecodeError:
        return []
    defs: List[str] = []
    for entry in data:
        text = entry.get("text")
        if text:
            defs.append(text.strip())
    return defs

# --- WordNet (NLTK) ----------------------------------------------------------

def fetch_wordnet(word: str) -> List[str]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
    except Exception:
        return []
    definitions: List[str] = []
    try:
        for syn in wn.synsets(word):
            definition = syn.definition()
            if definition:
                definitions.append(definition)
            if len(definitions) >= 5:
                break
    except Exception:
        return []
    return definitions

# --- Unified -----------------------------------------------------------------

SOURCE_FUNCS = {
    "wiktionary": fetch_wiktionary,
    "datamuse": fetch_datamuse,
    "free_dictionary": fetch_free_dictionary,
    "wordnik": fetch_wordnik,
    "wordnet": fetch_wordnet,
}

def fetch_definitions(word: str, sources: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Fetch definitions from multiple sources.

    Returns dict mapping source name -> list of definitions.
    """
    if sources is None:
        sources = list(SOURCE_FUNCS.keys())
    results: Dict[str, List[str]] = {}
    for name in sources:
        func = SOURCE_FUNCS.get(name)
        if not func:
            continue
        try:
            defs = func(word)
            results[name] = defs
        except Exception as e:
            log.warning(f"Source {name} failed: {e}")
            results[name] = []
    return results

__all__ = [
    "fetch_wiktionary",
    "fetch_datamuse",
    "fetch_free_dictionary",
    "fetch_wordnik",
    "fetch_wordnet",
    "fetch_definitions",
]
