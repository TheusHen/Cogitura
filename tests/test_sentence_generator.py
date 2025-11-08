"""
Testes para módulo de geração de sentenças
"""
from unittest.mock import Mock, patch

import pytest

from cogitura.core.sentence_generator import SentenceGenerator


class TestSentenceGenerator:
    """Testes para classe SentenceGenerator"""

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_initialization(self, mock_get_provider):
        """Testa inicialização do gerador de sentenças"""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        config = {"ai_provider": "openai", "provider_config": {"api_key": "test_key"}}

        generator = SentenceGenerator(config)

        assert generator.provider == mock_provider
        mock_get_provider.assert_called_once_with("openai", {"api_key": "test_key"})

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_generate_single_sentence(self, mock_get_provider):
        """Testa geração de uma única sentença"""
        mock_provider = Mock()
        mock_provider.generate_sentence.return_value = "This is a test sentence."
        mock_get_provider.return_value = mock_provider

        config = {"ai_provider": "openai", "provider_config": {"api_key": "test_key"}}

        generator = SentenceGenerator(config)
        sentence = generator.generate()

        assert sentence == "This is a test sentence."
        mock_provider.generate_sentence.assert_called_once()

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_generate_multiple_sentences(self, mock_get_provider):
        """Testa geração de múltiplas sentenças"""
        mock_provider = Mock()
        mock_provider.generate_sentence.side_effect = [
            "Sentence one.",
            "Sentence two.",
            "Sentence three.",
        ]
        mock_get_provider.return_value = mock_provider

        config = {"ai_provider": "openai", "provider_config": {"api_key": "test_key"}}

        generator = SentenceGenerator(config)
        sentences = generator.generate_batch(count=3)

        assert len(sentences) == 3
        assert sentences == ["Sentence one.", "Sentence two.", "Sentence three."]
        assert mock_provider.generate_sentence.call_count == 3

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_extract_words_from_sentence(self, mock_get_provider):
        """Testa extração de palavras de uma sentença"""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        config = {"ai_provider": "openai", "provider_config": {"api_key": "test_key"}}

        generator = SentenceGenerator(config)
        words = generator.extract_words("Hello, world! This is a test.")

        # Palavras devem ser lowercase e sem pontuação
        expected = ["hello", "world", "this", "is", "a", "test"]
        assert words == expected

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_extract_words_handles_special_chars(self, mock_get_provider):
        """Testa extração de palavras com caracteres especiais"""
        mock_provider = Mock()
        mock_get_provider.return_value = mock_provider

        config = {"ai_provider": "openai", "provider_config": {"api_key": "test_key"}}

        generator = SentenceGenerator(config)
        words = generator.extract_words("It's a beautiful day, isn't it?")

        # Deve lidar corretamente com contrações e pontuação
        assert "it" in words or "its" in words
        assert "beautiful" in words
        assert "day" in words

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_generate_with_retry_on_failure(self, mock_get_provider):
        """Testa retry quando geração falha"""
        mock_provider = Mock()
        mock_provider.generate_sentence.side_effect = [Exception("API Error"), "Success sentence."]
        mock_get_provider.return_value = mock_provider

        config = {
            "ai_provider": "openai",
            "provider_config": {"api_key": "test_key"},
            "max_retries": 3,
        }

        generator = SentenceGenerator(config)
        sentence = generator.generate_with_retry()

        assert sentence == "Success sentence."
        assert mock_provider.generate_sentence.call_count == 2

    @patch("cogitura.core.sentence_generator.get_provider")
    def test_generate_fails_after_max_retries(self, mock_get_provider):
        """Testa falha após máximo de tentativas"""
        mock_provider = Mock()
        mock_provider.generate_sentence.side_effect = Exception("Persistent Error")
        mock_get_provider.return_value = mock_provider

        config = {
            "ai_provider": "openai",
            "provider_config": {"api_key": "test_key"},
            "max_retries": 2,
        }

        generator = SentenceGenerator(config)

        with pytest.raises(Exception):
            generator.generate_with_retry()

        assert mock_provider.generate_sentence.call_count == 2
