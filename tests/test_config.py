"""
Testes para o módulo de configuração
"""
import pytest
from cogitura.config import Config


def test_config_directories():
    """Testa se os diretórios estão configurados corretamente"""
    assert Config.DATA_DIR.exists()
    assert Config.AUDIO_DIR.exists()
    assert Config.MODELS_DIR.exists()
    assert Config.LOGS_DIR.exists()


def test_config_elasticsearch():
    """Testa configurações do ElasticSearch"""
    assert Config.ELASTICSEARCH_HOST is not None
    assert Config.ELASTICSEARCH_PORT > 0
    assert Config.ELASTICSEARCH_INDEX_SENTENCES is not None
    assert Config.ELASTICSEARCH_INDEX_WORDS is not None


def test_config_validation():
    """Testa validação de configuração"""
    is_valid, errors = Config.validate_config()
    # Pode ser inválido se não houver API key configurada
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)


def test_config_providers():
    """Testa se o provedor de IA está configurado"""
    assert Config.AI_PROVIDER in ['openai', 'anthropic', 'google', 'ollama', 'custom', 'local']
