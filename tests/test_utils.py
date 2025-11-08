"""
Testes para utilitários
"""
import pytest

from cogitura.utils import (
    calculate_accuracy,
    calculate_wer,
    hash_text,
    sanitize_filename,
    split_sentence_into_words,
)


def test_hash_text():
    """Testa geração de hash"""
    text = "hello world"
    hash1 = hash_text(text)
    hash2 = hash_text(text)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 gera 64 caracteres hex

    # Textos diferentes devem gerar hashes diferentes
    hash3 = hash_text("different text")
    assert hash1 != hash3


def test_split_sentence_into_words():
    """Testa divisão de sentenças em palavras"""
    sentence = "Hello, World! This is a test."
    words = split_sentence_into_words(sentence)

    assert words == ["hello", "world", "this", "is", "a", "test"]

    # Testa sentença vazia
    assert split_sentence_into_words("") == []

    # Testa sentença com pontuação
    sentence = "It's a beautiful day!"
    words = split_sentence_into_words(sentence)
    assert "it" in words or "s" in words  # Depende da implementação


def test_sanitize_filename():
    """Testa sanitização de nomes de arquivo"""
    filename = "my file!@#$%^&*.txt"
    sanitized = sanitize_filename(filename)

    # Não deve conter caracteres especiais
    assert "@" not in sanitized
    assert "#" not in sanitized
    assert "%" not in sanitized

    # Deve manter caracteres alfanuméricos
    assert "my" in sanitized
    assert "file" in sanitized


def test_calculate_accuracy():
    """Testa cálculo de acurácia"""
    predictions = ["hello", "world", "test"]
    targets = ["hello", "world", "test"]

    accuracy = calculate_accuracy(predictions, targets)
    assert accuracy == 1.0

    predictions = ["hello", "word", "test"]
    targets = ["hello", "world", "test"]

    accuracy = calculate_accuracy(predictions, targets)
    assert accuracy == pytest.approx(2 / 3)


def test_calculate_wer():
    """Testa cálculo de WER"""
    predictions = ["hello world"]
    targets = ["hello world"]

    wer = calculate_wer(predictions, targets)
    assert wer == 0.0

    predictions = ["hello word"]
    targets = ["hello world"]

    wer = calculate_wer(predictions, targets)
    assert wer > 0.0
