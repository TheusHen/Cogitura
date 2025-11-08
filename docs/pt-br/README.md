# Cogitura - Documentação Completa

## Índice

1. [Introdução](#introdução)
2. [Arquitetura](#arquitetura)
3. [Guia de Instalação](#guia-de-instalação)
4. [Configuração](#configuração)
5. [Uso](#uso)
6. [Referência da API](#referência-da-api)
7. [Contribuindo](#contribuindo)

## Introdução

Cogitura é um projeto de pesquisa que explora a questão: **IAs podem criar outras IAs?**

O projeto implementa um pipeline completo onde uma IA generativa cria dados de treinamento para uma IA de reconhecimento de voz, demonstrando como a inteligência artificial pode ser usada para inicializar e treinar outros sistemas de IA.

### Objetivos da Pesquisa

- Investigar a viabilidade de dados de treinamento gerados por IA
- Comparar desempenho entre diferentes provedores de IA
- Analisar qualidade e diversidade das sentenças geradas
- Medir eficácia do modelo de reconhecimento de voz treinado
- Documentar descobertas para a comunidade de pesquisa

### Características Principais

- Suporte a múltiplos provedores de IA (OpenAI, Anthropic, Google, Ollama, Custom)
- Geração automática de text-to-speech usando gTTS
- Gerenciamento de dados baseado em ElasticSearch
- Treinamento de reconhecimento de voz com PyTorch
- Métricas de avaliação abrangentes
- Exportação fácil para Hugging Face

## Arquitetura

### Visão Geral do Sistema

```
┌─────────────────┐
│ Provedores IA   │
│(OpenAI, etc.)   │
└────────┬────────┘
         │ gera
         ▼
┌─────────────────┐
│   Sentenças     │
└────────┬────────┘
         │ divide em
         ▼
┌─────────────────┐     ┌──────────────┐
│    Palavras     │────▶│     gTTS     │
└────────┬────────┘     └──────┬───────┘
         │                     │
         │ armazena            │ gera
         ▼                     ▼
┌─────────────────┐     ┌──────────────┐
│ ElasticSearch   │     │Arquivos Áudio│
└────────┬────────┘     └──────┬───────┘
         │                     │
         └──────────┬──────────┘
                    │ treina
                    ▼
         ┌─────────────────────┐
         │   Modelo Speech-    │
         │      to-Text        │
         └──────────┬──────────┘
                    │ avalia
                    ▼
         ┌─────────────────────┐
         │    Resultados e     │
         │      Análise        │
         └─────────────────────┘
```

### Componentes

#### 1. Gerador de Sentenças
- Conecta ao provedor de IA configurado
- Gera sentenças diversas em inglês
- Valida tamanho e qualidade das sentenças
- Extrai palavras únicas

#### 2. Processador TTS
- Converte texto em fala usando gTTS
- Cache de arquivos de áudio para evitar regeneração
- Suporta múltiplos idiomas
- Processamento em lote para eficiência

#### 3. Gerenciador de Banco de Dados
- Gerencia índices do ElasticSearch
- Previne entradas duplicadas
- Rastreia estatísticas de uso de palavras
- Permite consultas eficientes

#### 4. Treinador de Modelo
- Usa arquitetura Wav2Vec2
- Suporta aceleração por GPU
- Implementa checkpointing
- Fornece métricas de treinamento

#### 5. Avaliador de Modelo
- Calcula métricas de acurácia
- Computa Word Error Rate (WER)
- Gera relatórios detalhados
- Exporta resultados para análise

## Guia de Instalação

### Requisitos do Sistema

- **Sistema Operacional**: Linux, macOS, ou Windows com WSL2
- **Python**: 3.9 ou superior
- **RAM**: 8GB mínimo, 16GB recomendado
- **Armazenamento**: 10GB para dados e modelos
- **GPU**: GPU NVIDIA com suporte CUDA (opcional, para treinamento mais rápido)

### Instalação Passo a Passo

#### 1. Instalar Dependências do Sistema

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.9 python3-pip python3-venv docker.io docker-compose git
```

**macOS:**
```bash
brew install python@3.9 docker docker-compose git
```

**Windows (WSL2):**
```bash
# Instale WSL2 primeiro, depois:
sudo apt update
sudo apt install -y python3.9 python3-pip python3-venv docker.io docker-compose git
```

#### 2. Clonar Repositório

```bash
git clone https://github.com/TheusHen/Cogitura.git
cd Cogitura
```

#### 3. Criar Ambiente Virtual

```bash
python3.9 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

#### 4. Instalar Dependências Python

```bash
pip install --upgrade pip
pip install -e .
```

#### 5. Configurar Ambiente

```bash
cp .env.example .env
nano .env  # Ou use seu editor preferido
```

Configure pelo menos um provedor de IA:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=sua_chave_api_aqui
```

#### 6. Iniciar ElasticSearch

```bash
docker-compose up -d
```

Aguarde o ElasticSearch iniciar (cerca de 30 segundos), depois verifique:

```bash
curl http://localhost:9200
```

Você deve ver uma resposta JSON com informações do cluster.

#### 7. Verificar Instalação

```bash
cogitura config-check
```

## Configuração

### Variáveis de Ambiente

#### Seleção de Provedor de IA

```env
# Escolha: openai, anthropic, google, ollama, custom
AI_PROVIDER=openai
```

#### Configuração OpenAI

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4  # ou gpt-3.5-turbo
OPENAI_MAX_TOKENS=100
OPENAI_TEMPERATURE=0.7
```

#### Configuração Anthropic

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-opus-20240229
ANTHROPIC_MAX_TOKENS=100
```

#### Configuração Google

```env
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-pro
```

#### Configuração Ollama

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

Primeiro instale o Ollama:
```bash
# Linux
curl https://ollama.ai/install.sh | sh
ollama pull llama2

# macOS
brew install ollama
ollama pull llama2
```

#### Configuração API Customizada

```env
CUSTOM_API_URL=http://localhost:8000/generate
CUSTOM_API_KEY=opcional
```

Sua API deve aceitar requisições POST com corpo JSON:
```json
{
  "prompt": "Gere uma sentença aleatória em inglês:"
}
```

E retornar:
```json
{
  "text": "A sentença gerada"
}
```

## Uso

### Interface de Linha de Comando

#### Verificar Configuração

```bash
cogitura config-check
```

#### Gerar Sentenças (Fase 1)

```bash
# Gerar 1000 sentenças
cogitura generate --count 1000 --save-db --generate-tts

# Opções:
#   --count, -c: Número de sentenças a gerar
#   --batch-size, -b: Tamanho do lote para geração
#   --save-db: Salvar no ElasticSearch
#   --generate-tts: Gerar arquivos de áudio
```

#### Treinar Modelo (Fase 2)

```bash
# Treinar com configurações padrão
cogitura train

# Treinar com configurações customizadas
cogitura train --epochs 100 --batch-size 64 --model facebook/wav2vec2-large-960h
```

#### Avaliar Modelo (Fase 3)

```bash
# Avaliar modelo treinado
cogitura evaluate ./models/trained/final_model

# Avaliar com tamanho de amostra
cogitura evaluate ./models/trained/final_model --sample-size 500
```

#### Gerenciamento de Banco de Dados

```bash
# Mostrar estatísticas do banco
cogitura db-stats

# Limpar todos os dados (use com cautela!)
cogitura db-clear
```

### API Python

#### Gerar Sentenças

```python
from cogitura.core.sentence_generator import SentenceGenerator

generator = SentenceGenerator()

# Gerar sentença única
sentence = generator.generate_single_sentence()
print(sentence)

# Gerar múltiplas sentenças
sentences = generator.generate_multiple(
    total_count=1000,
    batch_size=100,
    show_progress=True
)

# Obter estatísticas
stats = generator.get_statistics()
print(f"Geradas {stats['total_sentences']} sentenças")
print(f"Palavras únicas: {stats['unique_words']}")
```

#### Processar Text-to-Speech

```python
from cogitura.core.tts_processor import TTSProcessor

tts = TTSProcessor(language='en')

# Converter palavra única
audio_path = tts.word_to_speech("hello")
print(f"Áudio salvo em: {audio_path}")

# Processar palavras em lote
words = ["hello", "world", "python"]
word_audios = tts.batch_process_words(words, show_progress=True)

# Processar sentenças em lote
sentences = ["Hello world", "Python is great"]
sentence_audios = tts.batch_process_sentences(sentences)
```

## Referência da API

Veja [Referência da API](API.md) para documentação detalhada de todas as classes e métodos.

## Contribuindo

Contribuições são bem-vindas! Veja [CONTRIBUTING.md](../CONTRIBUTING.md) para diretrizes.

## Licença

Licença MIT - veja [LICENSE](../../LICENSE) para detalhes.
