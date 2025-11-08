"""
Configuração para testes pytest
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configura ambiente de teste"""
    # Define variáveis de ambiente para teste
    os.environ["AI_PROVIDER"] = "openai"
    os.environ["OPENAI_API_KEY"] = "test_key"
    os.environ["ELASTICSEARCH_HOST"] = "localhost"
    os.environ["ELASTICSEARCH_PORT"] = "9200"

    yield

    # Cleanup após testes


@pytest.fixture
def temp_audio_dir(tmp_path):
    """Cria diretório temporário para áudios"""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    return audio_dir


@pytest.fixture
def temp_model_dir(tmp_path):
    """Cria diretório temporário para modelos"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def sample_sentences():
    """Retorna lista de sentenças de exemplo"""
    return [
        "hello world",
        "this is a test",
        "python programming",
        "artificial intelligence",
        "machine learning",
    ]


@pytest.fixture
def sample_words():
    """Retorna lista de palavras de exemplo"""
    return ["hello", "world", "test", "python", "programming"]
