"""
Testes para processador TTS
"""

import pytest

from cogitura.config import Config
from cogitura.core.tts_processor import TTSProcessor


@pytest.fixture
def tts_processor():
    """Fixture para criar processador TTS"""
    return TTSProcessor()


def test_tts_initialization(tts_processor):
    """Testa inicialização do processador TTS"""
    assert tts_processor.language == Config.TTS_LANGUAGE
    assert tts_processor.output_dir.exists()


def test_text_to_speech(tts_processor, tmp_path):
    """Testa conversão de texto para fala"""
    # Usa diretório temporário
    tts_processor.output_dir = tmp_path

    text = "hello"
    audio_path = tts_processor.text_to_speech(text)

    assert audio_path.exists()
    assert audio_path.suffix == ".mp3"
    assert audio_path.stat().st_size > 0


def test_word_to_speech(tts_processor, tmp_path):
    """Testa conversão de palavra para fala"""
    tts_processor.output_dir = tmp_path

    word = "test"
    audio_path = tts_processor.word_to_speech(word)

    assert audio_path.exists()
    assert "word_test" in audio_path.name


def test_batch_process_words(tts_processor, tmp_path):
    """Testa processamento em lote de palavras"""
    tts_processor.output_dir = tmp_path

    words = ["hello", "world", "test"]
    results = tts_processor.batch_process_words(words, show_progress=False)

    assert len(results) == len(words)
    for word, audio_path in results.items():
        assert audio_path.exists()


def test_get_statistics(tts_processor):
    """Testa obtenção de estatísticas"""
    stats = tts_processor.get_statistics()

    assert "total_files" in stats
    assert "total_size_bytes" in stats
    assert "output_directory" in stats
