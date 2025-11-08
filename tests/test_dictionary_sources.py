"""Tests for dictionary_sources module.
All external HTTP calls are mocked to avoid network usage.
"""
from unittest.mock import patch, Mock
from src.cogitura.core.dictionary_sources import (
    fetch_wiktionary,
    fetch_datamuse,
    fetch_free_dictionary,
    fetch_wordnik,
    fetch_wordnet,
    fetch_definitions
)

HTML_SAMPLE = """
<html><body><ol><li>First definition of test.</li><li>Second definition with <b>markup</b>.</li></ol></body></html>
"""

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_wiktionary(mock_get):
    mock_resp = Mock(status_code=200, text=HTML_SAMPLE)
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    defs = fetch_wiktionary('test')
    assert len(defs) >= 1
    assert any('definition' in d.lower() for d in defs)

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_datamuse(mock_get):
    mock_resp = Mock(status_code=200)
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {"word": "test", "defs": ["n\tA procedure." ]}
    ]
    mock_get.return_value = mock_resp
    defs = fetch_datamuse('test')
    assert defs == ['A procedure.']

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_free_dictionary(mock_get):
    mock_resp = Mock(status_code=200)
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [
        {"word": "test", "meanings": [
            {"partOfSpeech": "noun", "definitions": [{"definition": "An examination."}]}
        ]}
    ]
    mock_get.return_value = mock_resp
    defs = fetch_free_dictionary('test')
    assert defs == ['An examination.']

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_wordnik_no_key(mock_get, monkeypatch):
    monkeypatch.delenv('WORDNIK_API_KEY', raising=False)
    defs = fetch_wordnik('test')
    assert defs == []  # no key returns empty list

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_wordnik_with_key(mock_get, monkeypatch):
    monkeypatch.setenv('WORDNIK_API_KEY', 'dummy')
    mock_resp = Mock(status_code=200)
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = [{"text": "A trial."}]
    mock_get.return_value = mock_resp
    defs = fetch_wordnik('test')
    assert defs == ['A trial.']

def test_fetch_wordnet_optional():
    # WordNet may not be available; function should not raise.
    defs = fetch_wordnet('test')
    assert isinstance(defs, list)

@patch('src.cogitura.core.dictionary_sources.requests.get')
def test_fetch_definitions(mock_get):
    mock_resp = Mock(status_code=200)
    mock_resp.raise_for_status.return_value = None
    mock_resp.text = HTML_SAMPLE
    mock_resp.json.return_value = []
    mock_get.return_value = mock_resp
    agg = fetch_definitions('test', sources=['wiktionary','datamuse'])
    assert 'wiktionary' in agg and 'datamuse' in agg
    assert isinstance(agg['wiktionary'], list)
