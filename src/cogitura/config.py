"""
Configurações globais do projeto Cogitura
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()


class Config:
    """Classe de configuração centralizada"""

    # Diretórios base
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    AUDIO_DIR = DATA_DIR / "audio"
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"

    # ElasticSearch
    ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    ELASTICSEARCH_INDEX_SENTENCES = os.getenv("ELASTICSEARCH_INDEX_SENTENCES", "cogitura_sentences")
    ELASTICSEARCH_INDEX_WORDS = os.getenv("ELASTICSEARCH_INDEX_WORDS", "cogitura_words")

    # AI Provider
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "100"))
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # Anthropic
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "100"))

    # Google (Gemini)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-pro")

    # Ollama (Local)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

    # Custom API
    CUSTOM_API_URL = os.getenv("CUSTOM_API_URL")
    CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")

    # TTS
    TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
    TTS_SLOW = os.getenv("TTS_SLOW", "false").lower() == "true"
    TTS_OUTPUT_DIR = Path(os.getenv("TTS_OUTPUT_DIR", str(AUDIO_DIR)))

    # Training
    TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "32"))
    TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "50"))
    TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", "0.001"))
    MODEL_OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", str(MODELS_DIR / "trained")))

    # Data Collection
    MAX_SENTENCES_TO_GENERATE = int(os.getenv("MAX_SENTENCES_TO_GENERATE", "10000"))
    MIN_SENTENCE_LENGTH = int(os.getenv("MIN_SENTENCE_LENGTH", "5"))
    MAX_SENTENCE_LENGTH = int(os.getenv("MAX_SENTENCE_LENGTH", "20"))
    BATCH_SIZE_GENERATION = int(os.getenv("BATCH_SIZE_GENERATION", "100"))

    # Testing
    TEST_SAMPLE_SIZE = int(os.getenv("TEST_SAMPLE_SIZE", "1000"))
    TEST_OUTPUT_DIR = Path(os.getenv("TEST_OUTPUT_DIR", str(DATA_DIR / "test_results")))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = Path(os.getenv("LOG_FILE", str(LOGS_DIR / "cogitura.log")))

    @classmethod
    def create_directories(cls):
        """Cria os diretórios necessários se não existirem"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Cria arquivos .gitkeep
        (cls.AUDIO_DIR / ".gitkeep").touch(exist_ok=True)
        (cls.MODELS_DIR / ".gitkeep").touch(exist_ok=True)
        (cls.TEST_OUTPUT_DIR / ".gitkeep").touch(exist_ok=True)

    @classmethod
    def validate_config(cls) -> tuple[bool, list[str]]:
        """
        Valida se a configuração está correta

        Returns:
            Tupla (is_valid, errors) onde is_valid é booleano e errors é lista de mensagens
        """
        errors = []

        # Valida provider de IA
        if cls.AI_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY não configurada")
        elif cls.AI_PROVIDER == "anthropic" and not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY não configurada")
        elif cls.AI_PROVIDER == "google" and not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY não configurada")
        elif cls.AI_PROVIDER == "custom" and not cls.CUSTOM_API_URL:
            errors.append("CUSTOM_API_URL não configurada")

        return len(errors) == 0, errors


# Cria os diretórios na inicialização
Config.create_directories()
