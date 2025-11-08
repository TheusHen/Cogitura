"""
Gerador de Sentenças - Fase 1 do projeto
Gera sentenças usando IA e extrai palavras
"""
import re
from typing import Any, Dict, List, Optional

from cogitura.config import Config
from cogitura.logger import log
from cogitura.providers.ai_providers import get_provider


class SentenceGenerator:
    """Gerencia geração de sentenças e extração de palavras.

    Compatível com os testes (métodos generate, generate_batch, generate_with_retry)
    e mantém API expandida (generate_sentence, generate_sentences).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.provider_name = self.config.get("ai_provider") or Config.AI_PROVIDER
        self.provider_config = self.config.get("provider_config", {})
        self.max_retries = self.config.get("max_retries", 3)
        # Carrega provider imediatamente para testes que esperam chamada única
        self.provider = get_provider(self.provider_name, self.provider_config)
        log.info(f"SentenceGenerator inicializado com provider: {self.provider_name}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        """
        Gera uma sentença única

        Args:
            prompt: Prompt opcional

        Returns:
            Sentença gerada
        """
        # Provider já carregado no __init__ para compatibilidade
        for attempt in range(self.max_retries):
            try:
                sentence = self.provider.generate_sentence(prompt)
                log.debug(f"Sentença gerada: {sentence}")
                return sentence
            except Exception as e:
                log.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("Failed to generate sentence after max retries")

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        """
        Gera múltiplas sentenças

        Args:
            count: Número de sentenças
            prompt: Prompt opcional

        Returns:
            Lista de sentenças
        """
        # Estratégia simples: chamadas individuais para corresponder aos testes
        sentences: List[str] = []
        for _ in range(count):
            sentences.append(self.generate_sentence(prompt))
        log.info(f"{len(sentences)} sentenças geradas (batch)")
        return sentences

    def extract_words(self, sentence: str) -> List[str]:
        """
        Extrai palavras de uma sentença

        Args:
            sentence: Sentença para extrair palavras

        Returns:
            Lista de palavras únicas
        """
        # Remove pontuação e caracteres especiais
        cleaned = re.sub(r"[^\w\s]", "", sentence)

        # Divide em palavras e converte para minúsculas
        words = cleaned.lower().split()

        # Remove duplicatas mantendo a ordem
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        log.debug(f"{len(unique_words)} palavras únicas extraídas de: {sentence}")

        return unique_words

    # Métodos compatíveis com testes -------------------------------------
    def generate(self, prompt: Optional[str] = None) -> str:
        return self.generate_sentence(prompt)

    def generate_batch(self, count: int, prompt: Optional[str] = None) -> List[str]:
        return self.generate_sentences(count, prompt)

    def generate_with_retry(self, prompt: Optional[str] = None) -> str:
        # Reutiliza lógica de retries de generate_sentence
        return self.generate_sentence(prompt)
