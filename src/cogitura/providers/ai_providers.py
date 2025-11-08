"""
Provedores de IA para geração de sentenças
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests

# Expor símbolos no nível do módulo para facilitar patch nos testes
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
try:
    from anthropic import Anthropic  # type: ignore
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore

from cogitura.config import Config
from cogitura.logger import log


class AIProvider(ABC):
    """Classe base abstrata para provedores de IA"""

    @abstractmethod
    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        """
        Gera uma sentença usando o provedor de IA

        Args:
            prompt: Prompt opcional para guiar a geração

        Returns:
            Sentença gerada
        """
        pass

    @abstractmethod
    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        """
        Gera múltiplas sentenças

        Args:
            count: Número de sentenças para gerar
            prompt: Prompt opcional para guiar a geração

        Returns:
            Lista de sentenças geradas
        """
        pass


class OpenAIProvider(AIProvider):
    """Provedor usando OpenAI API"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        api_key = api_key or Config.OPENAI_API_KEY
        if OpenAI is None:
            raise ImportError("OpenAI SDK not available")
        self.client = OpenAI(api_key=api_key)
        # Valor padrão esperado pelos testes
        self.model = model or "gpt-4"
        self.max_tokens = Config.OPENAI_MAX_TOKENS
        self.temperature = Config.OPENAI_TEMPERATURE
        log.info(f"OpenAI provider inicializado com modelo {self.model}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates simple English sentences.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            sentence = response.choices[0].message.content.strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com OpenAI: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        if prompt is None:
            prompt = f"Generate {count} different random English sentences with common words. Return only the sentences, one per line:"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates simple English sentences.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens * count,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content.strip()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            log.debug(f"{len(sentences)} sentenças geradas")
            return sentences[:count]
        except Exception as e:
            log.error(f"Erro ao gerar sentenças com OpenAI: {e}")
            raise


class AnthropicProvider(AIProvider):
    """Provedor usando Anthropic (Claude) API"""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        if Anthropic is None:
            raise ImportError("Anthropic SDK not available")
        api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=api_key)
        # Valor padrão esperado pelos testes
        self.model = model or "claude-3-opus-20240229"
        self.max_tokens = Config.ANTHROPIC_MAX_TOKENS
        log.info(f"Anthropic provider inicializado com modelo {self.model}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            sentence = response.content[0].text.strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com Anthropic: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        if prompt is None:
            prompt = f"Generate {count} different random English sentences with common words. Return only the sentences, one per line:"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens * count,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text.strip()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            log.debug(f"{len(sentences)} sentenças geradas")
            return sentences[:count]
        except Exception as e:
            log.error(f"Erro ao gerar sentenças com Anthropic: {e}")
            raise


class GoogleProvider(AIProvider):
    """Provedor usando Google Gemini API"""

    def __init__(self, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-generativeai not available")
        api_key = api_key or Config.GOOGLE_API_KEY
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(Config.GOOGLE_MODEL)
        log.info(f"Google provider inicializado com modelo {Config.GOOGLE_MODEL}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            response = self.model.generate_content(prompt)
            sentence = response.text.strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com Google: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        if prompt is None:
            prompt = f"Generate {count} different random English sentences with common words. Return only the sentences, one per line:"

        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            log.debug(f"{len(sentences)} sentenças geradas")
            return sentences[:count]
        except Exception as e:
            log.error(f"Erro ao gerar sentenças com Google: {e}")
            raise


class GeminiProvider(AIProvider):
    """Provedor usando Google Gemini API (alias para GoogleProvider)"""

    def __init__(self, api_key: Optional[str] = None):
        if genai is None:
            raise ImportError("google-generativeai not available")
        api_key = api_key or Config.GOOGLE_API_KEY
        genai.configure(api_key=api_key)
        self.model_name = "gemini-pro"
        self.model = genai.GenerativeModel(self.model_name)
        log.info(f"Gemini provider inicializado com modelo {self.model_name}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            response = self.model.generate_content(prompt)
            sentence = response.text.strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com Gemini: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        if prompt is None:
            prompt = f"Generate {count} different random English sentences with common words. Return only the sentences, one per line:"

        try:
            response = self.model.generate_content(prompt)
            content = response.text.strip()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            log.debug(f"{len(sentences)} sentenças geradas")
            return sentences[:count]
        except Exception as e:
            log.error(f"Erro ao gerar sentenças com Gemini: {e}")
            raise


class OllamaProvider(AIProvider):
    """Provedor usando Ollama (modelos locais)"""

    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        log.info(f"Ollama provider inicializado com modelo {self.model} em {self.base_url}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            sentence = response.json()["response"].strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com Ollama: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        if prompt is None:
            prompt = f"Generate {count} different random English sentences with common words. Return only the sentences, one per line:"

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            content = response.json()["response"].strip()
            sentences = [s.strip() for s in content.split("\n") if s.strip()]
            log.debug(f"{len(sentences)} sentenças geradas")
            return sentences[:count]
        except Exception as e:
            log.error(f"Erro ao gerar sentenças com Ollama: {e}")
            raise


class CustomProvider(AIProvider):
    """Provedor usando API customizada"""

    def __init__(self):
        self.api_url = Config.CUSTOM_API_URL
        self.api_key = Config.CUSTOM_API_KEY
        log.info(f"Custom provider inicializado com URL {self.api_url}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence with common words:"

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            response = requests.post(self.api_url, json={"prompt": prompt}, headers=headers)
            response.raise_for_status()
            sentence = response.json()["text"].strip()
            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com Custom API: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        sentences = []
        for _ in range(count):
            sentences.append(self.generate_sentence(prompt))
        return sentences


class LocalModelProvider(AIProvider):
    """Provedor usando modelos locais do HuggingFace"""

    def __init__(self, model_path: str):
        import torch

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers not available")

        self.model_path = model_path
        log.info(f"Carregando modelo local de {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        log.info(f"Modelo local carregado no device: {self.device}")

    def generate_sentence(self, prompt: Optional[str] = None) -> str:
        if prompt is None:
            prompt = "Generate a random English sentence:"

        try:
            import torch

            # Tokenize (tests podem mockar tokenizer retornando dict simples)
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            if hasattr(tokenized, "to"):
                inputs = tokenized.to(self.device)
            else:
                # Constrói tensores manualmente
                input_ids = tokenized.get("input_ids")
                if input_ids is None:
                    raise ValueError("Tokenizer mock returned no input_ids")
                inputs = {"input_ids": torch.tensor(input_ids).to(self.device)}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from output
            if sentence.startswith(prompt):
                sentence = sentence[len(prompt) :].strip()

            log.debug(f"Sentença gerada: {sentence}")
            return sentence
        except Exception as e:
            log.error(f"Erro ao gerar sentença com modelo local: {e}")
            raise

    def generate_sentences(self, count: int, prompt: Optional[str] = None) -> List[str]:
        sentences = []
        for _ in range(count):
            sentence = self.generate_sentence(prompt)
            sentences.append(sentence)
        return sentences


def get_provider(
    provider_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None
) -> AIProvider:
    """
    Factory function para obter o provedor configurado

    Args:
        provider_name: Nome do provedor (openai, anthropic, gemini, google, local, ollama, custom)
        config: Configuração do provedor (api_key, model_path, etc)

    Returns:
        Instância do provedor de IA configurado
    """
    if config is None:
        config = {}

    if provider_name is None:
        provider_name = Config.AI_PROVIDER.lower()
    else:
        provider_name = provider_name.lower()

    if provider_name == "openai":
        return OpenAIProvider(api_key=config.get("api_key"))
    elif provider_name == "anthropic":
        return AnthropicProvider(api_key=config.get("api_key"))
    elif provider_name in ["google", "gemini"]:
        return GeminiProvider(api_key=config.get("api_key"))
    elif provider_name == "local":
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for local provider")
        return LocalModelProvider(model_path=model_path)
    elif provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "custom":
        return CustomProvider()
    else:
        raise ValueError(f"Unknown AI provider: {provider_name}")
