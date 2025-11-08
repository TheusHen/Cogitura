"""
Testes para módulo de avaliação
"""
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest

from src.cogitura.core.evaluator import Evaluator


class TestEvaluator:
    """Testes para classe Evaluator"""

    def test_initialization(self):
        """Testa inicialização do avaliador"""
        config = {"results_path": "/tmp/results"}

        evaluator = Evaluator(config)

        assert evaluator.results_path.endswith("results")

    @patch("src.cogitura.core.evaluator.Trainer")
    def test_load_model(self, mock_trainer_class):
        """Testa carregamento de modelo para avaliação"""
        mock_trainer = Mock()
        mock_model = Mock()
        mock_trainer.load_model.return_value = mock_model
        mock_trainer_class.return_value = mock_trainer

        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        model = evaluator.load_model("/path/to/model.pth")

        assert model == mock_model

    def test_calculate_accuracy(self):
        """Testa cálculo de acurácia"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        predictions = ["hello", "world", "test", "data"]
        ground_truth = ["hello", "world", "test", "date"]

        accuracy = evaluator.calculate_accuracy(predictions, ground_truth)

        assert accuracy == 0.75  # 3 corretos de 4

    def test_calculate_wer(self):
        """Testa cálculo de Word Error Rate"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        reference = "hello world test"
        hypothesis = "hello word test"

        wer = evaluator.calculate_wer(reference, hypothesis)

        # 1 erro (world -> word) em 3 palavras = 33.33%
        assert 0.30 <= wer <= 0.35

    def test_calculate_cer(self):
        """Testa cálculo de Character Error Rate"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        reference = "hello"
        hypothesis = "helo"

        cer = evaluator.calculate_cer(reference, hypothesis)

        # 1 caractere faltando em 5 = 20%
        assert 0.15 <= cer <= 0.25

    @patch("src.cogitura.core.evaluator.DatabaseManager")
    def test_evaluate_on_dataset(self, mock_db_manager):
        """Testa avaliação em dataset completo"""
        mock_db = Mock()
        mock_db.get_all_sentences.return_value = ["Hello world", "This is a test"]
        mock_db_manager.return_value = mock_db

        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        mock_model = Mock()
        mock_model.predict.return_value = ["Hello world", "This is a test"]

        results = evaluator.evaluate_on_dataset(mock_model, mock_db)

        assert "accuracy" in results
        assert "wer" in results or "total_sentences" in results

    def test_generate_confusion_matrix(self):
        """Testa geração de matriz de confusão"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        predictions = ["a", "b", "c", "a"]
        ground_truth = ["a", "b", "b", "a"]

        confusion = evaluator.generate_confusion_matrix(predictions, ground_truth)

        assert confusion is not None

    @patch("builtins.open", new_callable=mock_open)
    def test_save_results(self, mock_file):
        """Testa salvamento de resultados"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        results = {"accuracy": 0.95, "wer": 0.05, "total_samples": 1000}

        evaluator.save_results(results, "test_results.json")

        mock_file.assert_called()

    def test_calculate_metrics_summary(self):
        """Testa cálculo de resumo de métricas"""
        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        predictions = ["hello", "world", "test"]
        ground_truth = ["hello", "word", "test"]

        metrics = evaluator.calculate_metrics_summary(predictions, ground_truth)

        assert "accuracy" in metrics
        assert "correct_predictions" in metrics
        assert "total_predictions" in metrics
        assert metrics["correct_predictions"] == 2
        assert metrics["total_predictions"] == 3

    @patch("src.cogitura.core.evaluator.librosa")
    def test_evaluate_audio_prediction(self, mock_librosa):
        """Testa avaliação de predição de áudio"""
        mock_librosa.load.return_value = ([0.1, 0.2], 22050)

        config = {"results_path": "/tmp/results"}
        evaluator = Evaluator(config)

        mock_model = Mock()
        mock_model.predict.return_value = "hello"

        prediction = evaluator.evaluate_audio_prediction(mock_model, "/path/to/audio.mp3")

        assert prediction == "hello"
        mock_librosa.load.assert_called_once()
