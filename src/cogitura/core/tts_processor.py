"""
Processador de Text-to-Speech usando gTTS
"""
from pathlib import Path
from typing import Dict, List, Optional

from gtts import gTTS
from tqdm import tqdm

from cogitura.config import Config
from cogitura.logger import log
from cogitura.utils import hash_text, sanitize_filename


class TTSProcessor:
    """Processa texto para áudio usando gTTS"""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        language: Optional[str] = None,
        slow: Optional[bool] = None,
    ):
        """
        Inicializa o processador TTS

        Args:
            output_dir: Diretório de saída para arquivos de áudio
            language: Código do idioma (ex: 'en', 'pt', 'es')
            slow: Se deve falar devagar
        """
        self.output_dir = output_dir or Config.TTS_OUTPUT_DIR
        self.language = language or Config.TTS_LANGUAGE
        self.slow = slow if slow is not None else Config.TTS_SLOW
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            f"TTSProcessor inicializado - idioma: {self.language}, diretório: {self.output_dir}"
        )

    def text_to_speech(self, text: str, output_filename: Optional[str] = None) -> Path:
        """
        Converte texto em áudio

        Args:
            text: Texto para converter
            output_filename: Nome do arquivo de saída (opcional, será gerado se não fornecido)

        Returns:
            Caminho do arquivo de áudio gerado
        """
        if not text.strip():
            raise ValueError("Texto não pode ser vazio")

        # Gera nome de arquivo se não fornecido
        if output_filename is None:
            # Usa hash do texto como nome de arquivo
            text_hash = hash_text(text)[:16]
            output_filename = f"{text_hash}.mp3"
        else:
            output_filename = sanitize_filename(output_filename)
            if not output_filename.endswith(".mp3"):
                output_filename += ".mp3"

        output_path = self.output_dir / output_filename

        # Verifica se já existe
        if output_path.exists():
            log.debug(f"Arquivo já existe: {output_path}")
            return output_path

        try:
            # Gera áudio
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            tts.save(str(output_path))
            log.debug(f"Áudio gerado: {output_path}")
            return output_path
        except Exception as e:
            log.error(f"Erro ao gerar áudio para '{text}': {e}")
            raise

    def word_to_speech(self, word: str) -> Path:
        """
        Converte uma palavra em áudio

        Args:
            word: Palavra para converter

        Returns:
            Caminho do arquivo de áudio
        """
        word = word.lower().strip()
        filename = f"word_{word}.mp3"
        return self.text_to_speech(word, filename)

    def sentence_to_speech(self, sentence: str, sentence_id: Optional[str] = None) -> Path:
        """
        Converte uma sentença em áudio

        Args:
            sentence: Sentença para converter
            sentence_id: ID da sentença (opcional)

        Returns:
            Caminho do arquivo de áudio
        """
        if sentence_id:
            filename = f"sentence_{sentence_id}.mp3"
        else:
            filename = None
        return self.text_to_speech(sentence, filename)

    def batch_process_words(self, words: List[str], show_progress: bool = True) -> Dict[str, Path]:
        """
        Processa múltiplas palavras em lote

        Args:
            words: Lista de palavras para processar
            show_progress: Se deve mostrar barra de progresso

        Returns:
            Dicionário mapeando palavra -> caminho do arquivo de áudio
        """
        log.info(f"Processando {len(words)} palavras para TTS")
        results = {}

        iterator = words
        if show_progress:
            iterator = tqdm(words, desc="Gerando TTS para palavras")

        for word in iterator:
            try:
                audio_path = self.word_to_speech(word)
                results[word] = audio_path
            except Exception as e:
                log.error(f"Erro ao processar palavra '{word}': {e}")
                continue

        log.info(f"TTS gerado para {len(results)} palavras")
        return results

    def batch_process_sentences(
        self, sentences: List[str], show_progress: bool = True
    ) -> Dict[str, Path]:
        """
        Processa múltiplas sentenças em lote

        Args:
            sentences: Lista de sentenças para processar
            show_progress: Se deve mostrar barra de progresso

        Returns:
            Dicionário mapeando sentença -> caminho do arquivo de áudio
        """
        log.info(f"Processando {len(sentences)} sentenças para TTS")
        results = {}

        iterator = enumerate(sentences)
        if show_progress:
            iterator = tqdm(list(iterator), desc="Gerando TTS para sentenças")

        for idx, sentence in iterator:
            try:
                sentence_id = hash_text(sentence)[:16]
                audio_path = self.sentence_to_speech(sentence, sentence_id)
                results[sentence] = audio_path
            except Exception as e:
                log.error(f"Erro ao processar sentença '{sentence}': {e}")
                continue

        log.info(f"TTS gerado para {len(results)} sentenças")
        return results

    def get_statistics(self) -> dict:
        """
        Retorna estatísticas sobre arquivos de áudio gerados

        Returns:
            Dicionário com estatísticas
        """
        audio_files = list(self.output_dir.glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in audio_files)

        return {
            "total_files": len(audio_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "output_directory": str(self.output_dir),
        }
