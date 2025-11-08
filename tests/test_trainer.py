"""
Testes para módulo de treinamento
"""
from pathlib import Path
from unittest.mock import Mock, patch

from cogitura.core.trainer import Trainer


class TestTrainer:
    """Testes para classe Trainer"""

    def test_initialization(self):
        """Testa inicialização do trainer"""
        config = {"model_output_path": "/tmp/models", "batch_size": 16, "epochs": 10}

        trainer = Trainer(config)

        assert trainer.model_output_path == Path("/tmp/models")
        assert trainer.batch_size == 16
        assert trainer.epochs == 10

    @patch("cogitura.core.trainer.DatabaseManager")
    def test_load_training_data(self, mock_db_manager):
        """Testa carregamento de dados de treinamento"""
        mock_db = Mock()
        mock_db.get_all_words.return_value = [
            {"word": "hello", "audio_path": "/audio/hello.mp3"},
            {"word": "world", "audio_path": "/audio/world.mp3"},
        ]
        mock_db_manager.return_value = mock_db

        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        data = trainer.load_training_data(mock_db)

        assert len(data) == 2
        assert data[0]["word"] == "hello"

    @patch("cogitura.core.trainer.librosa")
    def test_prepare_audio_features(self, mock_librosa):
        """Testa preparação de features de áudio"""
        mock_librosa.load.return_value = ([0.1, 0.2, 0.3], 22050)
        mock_librosa.feature.mfcc.return_value = [[0.1, 0.2], [0.3, 0.4]]

        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        features = trainer.prepare_audio_features("/path/to/audio.mp3")

        assert features is not None
        mock_librosa.load.assert_called_once()

    def test_create_model_architecture(self):
        """Testa criação da arquitetura do modelo"""
        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        model = trainer.create_model(input_shape=(13, 100), num_classes=50)

        assert model is not None

    @patch("cogitura.core.trainer.torch.save")
    def test_save_model(self, mock_save):
        """Testa salvamento do modelo"""
        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        mock_model = Mock()
        trainer.save_model(mock_model, "test_model.pth")

        mock_save.assert_called_once()

    @patch("cogitura.core.trainer.torch.load")
    def test_load_model(self, mock_load):
        """Testa carregamento do modelo"""
        mock_model = Mock()
        mock_load.return_value = mock_model

        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        loaded_model = trainer.load_model("test_model.pth")

        assert loaded_model == mock_model
        mock_load.assert_called_once()

    @patch("cogitura.core.trainer.DatabaseManager")
    @patch("cogitura.core.trainer.librosa")
    def test_train_epoch(self, mock_librosa, mock_db_manager):
        """Testa treinamento de uma época"""
        mock_librosa.load.return_value = ([0.1, 0.2], 22050)
        mock_librosa.feature.mfcc.return_value = [[0.1], [0.2]]

        mock_db = Mock()
        mock_db.get_all_words.return_value = [{"word": "test", "audio_path": "/audio/test.mp3"}]
        mock_db_manager.return_value = mock_db

        config = {"model_output_path": "/tmp/models", "batch_size": 1, "epochs": 1}
        trainer = Trainer(config)

        # Mock do modelo
        mock_model = Mock()
        mock_optimizer = Mock()

        loss = trainer.train_epoch(
            mock_model, mock_optimizer, [{"word": "test", "audio_path": "/audio/test.mp3"}]
        )

        assert isinstance(loss, (int, float)) or loss is not None

    def test_prepare_dataset_split(self):
        """Testa divisão do dataset em treino/validação/teste"""
        config = {"model_output_path": "/tmp/models"}
        trainer = Trainer(config)

        data = [{"word": f"word{i}", "audio_path": f"/audio/word{i}.mp3"} for i in range(100)]

        train, val, test = trainer.split_dataset(data, train_ratio=0.7, val_ratio=0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100
