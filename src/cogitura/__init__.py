"""
Cogitura - AI Creating AI Research Project

Este pacote contém a implementação completa do projeto Cogitura,
uma pesquisa sobre IAs criando outras IAs através de treinamento
de reconhecimento de voz.
"""

__version__ = "0.1.0"
__author__ = "TheusHen"

from cogitura.core.database_manager import DatabaseManager
from cogitura.core.evaluator import Evaluator
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.trainer import Trainer
from cogitura.core.tts_processor import TTSProcessor

__all__ = [
    "SentenceGenerator",
    "TTSProcessor",
    "DatabaseManager",
    "Trainer",
    "Evaluator",
]
