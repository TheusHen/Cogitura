"""
Utilitários gerais para o projeto Cogitura
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def hash_text(text: str) -> str:
    """
    Gera hash SHA256 de um texto

    Args:
        text: Texto para gerar hash

    Returns:
        Hash hexadecimal do texto
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_sentence_into_words(sentence: str) -> List[str]:
    """
    Divide uma frase em palavras

    Args:
        sentence: Frase para dividir

    Returns:
        Lista de palavras (lowercase, sem pontuação)
    """
    import re

    # Remove pontuação e converte para lowercase
    words = re.findall(r"\b[a-z]+\b", sentence.lower())
    return words


def sanitize_filename(filename: str) -> str:
    """
    Remove caracteres inválidos de um nome de arquivo

    Args:
        filename: Nome do arquivo para sanitizar

    Returns:
        Nome de arquivo sanitizado
    """
    import re

    # Remove caracteres especiais
    filename = re.sub(r"[^\w\s-]", "", filename)
    # Substitui espaços por underscores
    filename = re.sub(r"[\s]+", "_", filename)
    return filename


def save_json(data: Any, filepath: Path) -> None:
    """
    Salva dados em formato JSON

    Args:
        data: Dados para salvar
        filepath: Caminho do arquivo
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: Path) -> Any:
    """
    Carrega dados de um arquivo JSON

    Args:
        filepath: Caminho do arquivo

    Returns:
        Dados carregados
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_audio_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estima a duração de áudio para um texto

    Args:
        text: Texto para estimar
        words_per_minute: Palavras por minuto (velocidade de fala)

    Returns:
        Duração estimada em segundos
    """
    words = len(text.split())
    minutes = words / words_per_minute
    return minutes * 60


def format_bytes(bytes_size: int) -> str:
    """
    Formata tamanho em bytes para formato legível

    Args:
        bytes_size: Tamanho em bytes

    Returns:
        String formatada (ex: "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def calculate_accuracy(predictions: List[str], targets: List[str]) -> float:
    """
    Calcula a acurácia entre predições e targets

    Args:
        predictions: Lista de predições
        targets: Lista de valores reais

    Returns:
        Acurácia (0.0 a 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(predictions)


def calculate_wer(predictions: List[str], targets: List[str]) -> float:
    """
    Calcula Word Error Rate (WER)

    Args:
        predictions: Lista de predições
        targets: Lista de valores reais

    Returns:
        WER (0.0 a 1.0+)
    """
    from difflib import SequenceMatcher

    total_words = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()

        total_words += len(target_words)

        # Calcula distância de edição
        matcher = SequenceMatcher(None, pred_words, target_words)
        errors = len(target_words) - sum(block[2] for block in matcher.get_matching_blocks())
        total_errors += errors

    return total_errors / total_words if total_words > 0 else 0.0
