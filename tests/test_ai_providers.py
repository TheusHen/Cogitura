"""
Testes para módulo de providers de IA
"""
from unittest.mock import Mock, patch

import pytest

from cogitura.providers.ai_providers import (
    AIProvider,
    AnthropicProvider,
    GeminiProvider,
    LocalModelProvider,
    OpenAIProvider,
    get_provider,
)


class TestAIProvider:
    """Testes para classe base AIProvider"""

    def test_ai_provider_is_abstract(self):
        """Testa que AIProvider não pode ser instanciada diretamente"""
        with pytest.raises(TypeError):
            AIProvider()


class TestOpenAIProvider:
    """Testes para OpenAI Provider"""

    @patch("cogitura.providers.ai_providers.OpenAI")
    def test_openai_initialization(self, mock_openai):
        """Testa inicialização do OpenAI provider"""
        provider = OpenAIProvider(api_key="test_key")
        assert provider.model == "gpt-4"
        mock_openai.assert_called_once_with(api_key="test_key")

    @patch("cogitura.providers.ai_providers.OpenAI")
    def test_openai_generate_sentence(self, mock_openai):
        """Testa geração de sentença com OpenAI"""
        # Mock do cliente OpenAI
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="This is a test sentence."))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(api_key="test_key")
        sentence = provider.generate_sentence()

        assert sentence == "This is a test sentence."
        mock_client.chat.completions.create.assert_called_once()

    @patch("cogitura.providers.ai_providers.OpenAI")
    def test_openai_custom_model(self, mock_openai):
        """Testa uso de modelo customizado"""
        provider = OpenAIProvider(api_key="test_key", model="gpt-3.5-turbo")
        assert provider.model == "gpt-3.5-turbo"


class TestAnthropicProvider:
    """Testes para Anthropic Provider"""

    @patch("cogitura.providers.ai_providers.Anthropic")
    def test_anthropic_initialization(self, mock_anthropic):
        """Testa inicialização do Anthropic provider"""
        provider = AnthropicProvider(api_key="test_key")
        assert provider.model == "claude-3-opus-20240229"
        mock_anthropic.assert_called_once_with(api_key="test_key")

    @patch("cogitura.providers.ai_providers.Anthropic")
    def test_anthropic_generate_sentence(self, mock_anthropic):
        """Testa geração de sentença com Anthropic"""
        # Mock do cliente Anthropic
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a Claude response.")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = AnthropicProvider(api_key="test_key")
        sentence = provider.generate_sentence()

        assert sentence == "This is a Claude response."
        mock_client.messages.create.assert_called_once()


class TestGeminiProvider:
    """Testes para Google Gemini Provider"""

    @patch("cogitura.providers.ai_providers.genai")
    def test_gemini_initialization(self, mock_genai):
        """Testa inicialização do Gemini provider"""
        provider = GeminiProvider(api_key="test_key")
        assert provider.model_name == "gemini-pro"
        mock_genai.configure.assert_called_once_with(api_key="test_key")

    @patch("cogitura.providers.ai_providers.genai")
    def test_gemini_generate_sentence(self, mock_genai):
        """Testa geração de sentença com Gemini"""
        # Mock do modelo Gemini
        mock_model = Mock()
        mock_response = Mock(text="This is a Gemini response.")
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider(api_key="test_key")
        sentence = provider.generate_sentence()

        assert sentence == "This is a Gemini response."
        mock_model.generate_content.assert_called_once()


class TestLocalModelProvider:
    """Testes para Local Model Provider"""

    @patch("cogitura.providers.ai_providers.AutoTokenizer")
    @patch("cogitura.providers.ai_providers.AutoModelForCausalLM")
    def test_local_model_initialization(self, mock_model_class, mock_tokenizer_class):
        """Testa inicialização do provider de modelo local"""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        provider = LocalModelProvider(model_path="test/model")

        assert provider.model_path == "test/model"
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("cogitura.providers.ai_providers.AutoTokenizer")
    @patch("cogitura.providers.ai_providers.AutoModelForCausalLM")
    def test_local_model_generate_sentence(self, mock_model_class, mock_tokenizer_class):
        """Testa geração de sentença com modelo local"""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
        mock_tokenizer.decode.return_value = "This is a local model response."
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = Mock()
        mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
        mock_model_class.from_pretrained.return_value = mock_model

        provider = LocalModelProvider(model_path="test/model")
        sentence = provider.generate_sentence()

        assert isinstance(sentence, str)
        mock_model.generate.assert_called_once()


class TestGetProvider:
    """Testes para função get_provider"""

    @patch("cogitura.providers.ai_providers.OpenAI")
    def test_get_openai_provider(self, mock_openai):
        """Testa obtenção de provider OpenAI"""
        config = {"api_key": "test_key"}
        provider = get_provider("openai", config)

        assert isinstance(provider, OpenAIProvider)

    @patch("cogitura.providers.ai_providers.Anthropic")
    def test_get_anthropic_provider(self, mock_anthropic):
        """Testa obtenção de provider Anthropic"""
        config = {"api_key": "test_key"}
        provider = get_provider("anthropic", config)

        assert isinstance(provider, AnthropicProvider)

    @patch("cogitura.providers.ai_providers.genai")
    def test_get_gemini_provider(self, mock_genai):
        """Testa obtenção de provider Gemini"""
        config = {"api_key": "test_key"}
        provider = get_provider("gemini", config)

        assert isinstance(provider, GeminiProvider)

    @patch("cogitura.providers.ai_providers.AutoTokenizer")
    @patch("cogitura.providers.ai_providers.AutoModelForCausalLM")
    def test_get_local_provider(self, mock_model_class, mock_tokenizer_class):
        """Testa obtenção de provider local"""
        config = {"model_path": "test/model"}
        provider = get_provider("local", config)

        assert isinstance(provider, LocalModelProvider)

    def test_get_provider_invalid(self):
        """Testa erro ao solicitar provider inválido"""
        with pytest.raises(ValueError, match="Unknown AI provider"):
            get_provider("invalid_provider", {})
