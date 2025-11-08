"""
Testes básicos de importação
"""
import pytest


def test_import_config():
    """Testa importação do módulo de configuração"""
    from cogitura.config import Config
    assert Config is not None


def test_import_logger():
    """Testa importação do logger"""
    from cogitura.logger import log
    assert log is not None


def test_import_utils():
    """Testa importação de utilitários"""
    from cogitura.utils import hash_text, split_sentence_into_words
    assert hash_text is not None
    assert split_sentence_into_words is not None


def test_import_providers():
    """Testa importação de provedores"""
    from cogitura.providers import get_provider
    assert get_provider is not None


def test_import_core():
    """Testa importação dos módulos core"""
    from cogitura.core.sentence_generator import SentenceGenerator
    from cogitura.core.tts_processor import TTSProcessor
    from cogitura.core.database_manager import DatabaseManager
    
    assert SentenceGenerator is not None
    assert TTSProcessor is not None
    assert DatabaseManager is not None


def test_package_version():
    """Testa versão do pacote"""
    import cogitura
    assert hasattr(cogitura, '__version__')
    assert cogitura.__version__ == "0.1.0"
