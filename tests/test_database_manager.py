"""
Testes para módulo de gerenciamento de banco de dados
"""
from unittest.mock import Mock, patch


from cogitura.core.database_manager import DatabaseManager


class TestDatabaseManager:
    """Testes para classe DatabaseManager"""

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_initialization(self, mock_es):
        """Testa inicialização do gerenciador de banco"""
        mock_client = Mock()
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}

        db_manager = DatabaseManager(config)

        assert db_manager.es == mock_client
        mock_es.assert_called_once()

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_create_indices(self, mock_es):
        """Testa criação de índices no Elasticsearch"""
        mock_client = Mock()
        mock_client.indices.exists.return_value = False
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)
        db_manager.create_indices()

        # Deve criar índices de sentenças e palavras
        assert mock_client.indices.create.call_count >= 2

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_save_sentence(self, mock_es):
        """Testa salvamento de sentença"""
        mock_client = Mock()
        mock_client.index.return_value = {"_id": "test123"}
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        sentence_id = db_manager.save_sentence("This is a test sentence.")

        assert sentence_id == "test123"
        mock_client.index.assert_called_once()

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_sentence_exists(self, mock_es):
        """Testa verificação de existência de sentença"""
        mock_client = Mock()
        mock_client.search.return_value = {
            "hits": {"total": {"value": 1}, "hits": [{"_id": "existing123"}]}
        }
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        exists = db_manager.sentence_exists("This is a test sentence.")

        assert exists is True
        mock_client.search.assert_called_once()

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_sentence_not_exists(self, mock_es):
        """Testa verificação quando sentença não existe"""
        mock_client = Mock()
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        exists = db_manager.sentence_exists("Non-existent sentence.")

        assert exists is False

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_save_word_with_audio(self, mock_es):
        """Testa salvamento de palavra com áudio"""
        mock_client = Mock()
        mock_client.index.return_value = {"_id": "word123"}
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        word_id = db_manager.save_word(word="hello", audio_path="/path/to/hello.mp3")

        assert word_id == "word123"
        mock_client.index.assert_called_once()

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_word_exists(self, mock_es):
        """Testa verificação de existência de palavra"""
        mock_client = Mock()
        mock_client.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{"_id": "word123", "_source": {"audio_path": "/path/to/word.mp3"}}],
            }
        }
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        exists, audio_path = db_manager.word_exists("hello")

        assert exists is True
        assert audio_path == "/path/to/word.mp3"

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_word_not_exists(self, mock_es):
        """Testa verificação quando palavra não existe"""
        mock_client = Mock()
        mock_client.search.return_value = {"hits": {"total": {"value": 0}, "hits": []}}
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        exists, audio_path = db_manager.word_exists("nonexistent")

        assert exists is False
        assert audio_path is None

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_get_all_sentences(self, mock_es):
        """Testa recuperação de todas as sentenças"""
        mock_client = Mock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"text": "Sentence 1"}},
                    {"_source": {"text": "Sentence 2"}},
                    {"_source": {"text": "Sentence 3"}},
                ]
            }
        }
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        sentences = db_manager.get_all_sentences()

        assert len(sentences) == 3
        assert sentences[0] == "Sentence 1"

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_get_all_words(self, mock_es):
        """Testa recuperação de todas as palavras"""
        mock_client = Mock()
        mock_client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"word": "hello", "audio_path": "/path/hello.mp3"}},
                    {"_source": {"word": "world", "audio_path": "/path/world.mp3"}},
                ]
            }
        }
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        words = db_manager.get_all_words()

        assert len(words) == 2
        assert words[0]["word"] == "hello"

    @patch("src.cogitura.core.database_manager.Elasticsearch")
    def test_get_statistics(self, mock_es):
        """Testa obtenção de estatísticas do banco"""
        mock_client = Mock()
        mock_client.count.side_effect = [{"count": 1000}, {"count": 5000}]  # sentences  # words
        mock_es.return_value = mock_client

        config = {"host": "localhost", "port": 9200}
        db_manager = DatabaseManager(config)

        stats = db_manager.get_statistics()

        assert stats["total_sentences"] == 1000
        assert stats["total_words"] == 5000
