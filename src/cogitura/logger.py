"""
Sistema de logging configurável para o projeto Cogitura
"""
import sys
from pathlib import Path
from loguru import logger
from cogitura.config import Config


def setup_logger():
    """Configura o logger do projeto"""
    # Remove configuração padrão
    logger.remove()
    
    # Adiciona handler para console
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=Config.LOG_LEVEL,
        colorize=True,
    )
    
    # Adiciona handler para arquivo
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        Config.LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=Config.LOG_LEVEL,
        rotation="10 MB",
        retention="1 week",
        compression="zip",
    )
    
    return logger


# Instância global do logger
log = setup_logger()
