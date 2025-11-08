# Cogitura

> **IAs podem criar outras IAs?** Um projeto de pesquisa para explorar a criação de modelos de reconhecimento de voz usando dados gerados por IAs.

[![Tests](https://github.com/TheusHen/Cogitura/workflows/tests/badge.svg)](https://github.com/TheusHen/Cogitura/actions)
[![Coverage](https://codecov.io/gh/TheusHen/Cogitura/branch/main/graph/badge.svg)](https://codecov.io/gh/TheusHen/Cogitura)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Visão Geral

Cogitura é um projeto de pesquisa que explora a questão: **IAs podem criar outras IAs?** 

O projeto implementa um pipeline completo onde:
1. Uma IA generativa (OpenAI, Anthropic, Google, Ollama ou customizada) gera sentenças em inglês
2. Cada sentença é dividida em palavras e convertida em áudio via gTTS
3. Os dados são armazenados no ElasticSearch (sentenças e palavras únicas com seus respectivos áudios)
4. Um modelo de Speech-to-Text é treinado com esses dados
5. O modelo é avaliado e os resultados são analisados

## 🆕 Novidades (2025)

### Dicionários Online
Módulo de fontes de dicionários com scraping leve e APIs públicas:

- **Wiktionary** (HTML scraping mínimo)
- **Datamuse API** (sem chave requerida)
- **Free Dictionary API** (acesso público)
- **Wordnik API** (opcional, via `WORDNIK_API_KEY`)
- **WordNet** (NLTK, quando disponível)

```python
from cogitura.core.dictionary_sources import fetch_definitions

# Busca definições de múltiplas fontes
defs = fetch_definitions('test', sources=['wiktionary', 'datamuse'])
print(defs['wiktionary'][:3])
```

**Características:**
- ✅ User-Agent customizado
- ✅ Timeout e backoff automático para rate limiting
- ✅ Tratamento de erros robusto (retorna listas vazias)
- ✅ Totalmente testado com mocks


## Testes

### Executar Testes

```bash
# Todos os testes
PYTHONPATH=. pytest

# Com cobertura
PYTHONPATH=. pytest --cov=src/cogitura --cov-report=html

# Testes específicos
PYTHONPATH=. pytest tests/test_ai_providers.py -v

# Testes paralelos (mais rápido)
PYTHONPATH=. pytest -n auto
```

### Status dos Testes (2025)
- ✅ **78/78 testes passando**
- ✅ CI/CD automatizado
- ✅ Testes de integração com mocks

## CI/CD

### GitHub Actions Workflows

O projeto possui workflows automatizados:

- **Tests**: Testes em Python 3.9, 3.10, 3.11, 3.12
- **Lint**: Flake8, Black, isort
- **Type Check**: MyPy
- **Coverage**: Codecov integration
- **Docker**: Build e publicação de imagens
- **Security**: CodeQL e SonarCloud



## Estrutura do Projeto

```
Cogitura/
├── src/cogitura/          # Código fonte principal
│   ├── core/              # Módulos principais
│   │   ├── sentence_generator.py    # Geração de sentenças com IA
│   │   ├── tts_processor.py         # Text-to-Speech
│   │   ├── database_manager.py      # Gerenciamento ElasticSearch
│   │   ├── trainer.py               # Treinamento de modelos
│   │   ├── evaluator.py             # Avaliação e métricas
│   │   └── dictionary_sources.py    # 🆕 Fontes de dicionário
│   ├── providers/         # Provedores de IA
│   │   └── ai_providers.py          # OpenAI, Anthropic, Gemini, etc
│   ├── config.py          # Configurações
│   ├── logger.py          # Sistema de logs
│   ├── utils.py           # Utilitários
│   └── cli.py             # Interface CLI
├── docs/                  # Documentação completa
│   ├── en/                # Documentação em Inglês
│   ├── pt-br/             # Documentação em Português
│   └── es/                # Documentação em Espanhol
├── tests/                 # Testes unitários (78 testes)
├── config/                # Arquivos de configuração
├── data/                  # Dados e áudios gerados
├── models/                # Modelos treinados
├── scripts/               # Scripts utilitários
└── docker-compose.yml     # Configuração do ElasticSearch
```

## Pré-requisitos

- Python 3.9+
- Docker e Docker Compose
- CUDA (opcional, para treinamento com GPU)

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/TheusHen/Cogitura.git
cd Cogitura
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências

```bash
pip install -e .
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure suas chaves de API:

```env
# Escolha seu provedor de IA
AI_PROVIDER=openai  # openai, anthropic, google, ollama, custom

# Configure a chave correspondente
OPENAI_API_KEY=sua_chave_aqui
# ou
ANTHROPIC_API_KEY=sua_chave_aqui
# ou
GOOGLE_API_KEY=sua_chave_aqui
```

### 5. Inicie o ElasticSearch

```bash
docker-compose up -d
```

Aguarde alguns segundos e verifique se está rodando:

```bash
curl http://localhost:9200
```

## Uso Rápido

### Verificar Configuração

```bash
cogitura config-check
```

### Fase 1: Gerar Sentenças e TTS

```bash
# Gera 1000 sentenças, salva no DB e gera TTS
cogitura generate --count 1000 --save-db --generate-tts
```

### Fase 2: Treinar Modelo

```bash
# Treina modelo de Speech-to-Text
cogitura train --epochs 50
```

### Fase 3: Avaliar Modelo

```bash
# Avalia modelo treinado
cogitura evaluate ./models/trained/final_model --sample-size 500
```

### Verificar Estatísticas do Banco

```bash
cogitura db-stats
```

## Provedores de IA Suportados

O Cogitura oferece liberdade total para escolher qual IA usar:

### OpenAI (GPT-4, GPT-3.5)

```env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4
```

### Anthropic (Claude)

```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-opus-20240229
```

### Google (Gemini)

```env
AI_PROVIDER=google
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-pro
```

### Ollama (Modelos Locais)

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

Instale o Ollama: https://ollama.ai

### API Customizada

```env
AI_PROVIDER=custom
CUSTOM_API_URL=http://localhost:8000/generate
CUSTOM_API_KEY=opcional
```

## Pipeline Completo

### 1. Geração de Dados

```python
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.tts_processor import TTSProcessor
from cogitura.core.database_manager import DatabaseManager
from cogitura.utils import split_sentence_into_words

# Gera sentenças
generator = SentenceGenerator()
sentences = generator.generate_multiple(1000)

# Processa TTS
tts = TTSProcessor()
word_audios = tts.batch_process_words(list(generator.unique_words))
sentence_audios = tts.batch_process_sentences(sentences)

# Salva no DB
db = DatabaseManager()
for sentence in sentences:
    words = split_sentence_into_words(sentence)
    db.add_sentence(sentence, words, audio_path=str(sentence_audios[sentence]))

for word in generator.unique_words:
    db.add_word(word, audio_path=str(word_audios[word]))
```

### 2. Treinamento

```python
from cogitura.core.trainer import ModelTrainer
from pathlib import Path

# Carrega dados
db = DatabaseManager()
sentences_data = db.get_all_sentences()

audio_paths = [Path(s["audio_path"]) for s in sentences_data]
texts = [s["sentence"] for s in sentences_data]

# Treina
trainer = ModelTrainer()
train_loader, val_loader = trainer.prepare_data(audio_paths, texts)
history = trainer.train(train_loader, val_loader, epochs=50)
```

### 3. Avaliação

```python
from cogitura.core.evaluator import ModelEvaluator
from pathlib import Path

# Carrega modelo
evaluator = ModelEvaluator(Path("./models/trained/final_model"))

# Avalia
metrics = evaluator.evaluate_dataset(audio_paths, texts)
report = evaluator.generate_report(metrics)
print(report)
```

## Documentação Completa

- [Documentação em Português (PT-BR)](docs/pt-br/README.md)
- [Documentation in English](docs/en/README.md)
- [Documentación en Español](docs/es/README.md)

## Análise de Dados

Os resultados podem ser facilmente analisados usando:

- **Kibana**: http://localhost:5601 (visualização de dados do ElasticSearch)
- **DataSpell / JupyterLab**: Abra os arquivos CSV em `data/test_results/`
- **Jupyter Notebooks**: Scripts de análise em `scripts/analysis/`
- **TensorBoard**: Visualização de métricas de treinamento

### Métricas Disponíveis

- Word Error Rate (WER)
- Character Error Rate (CER)
- Accuracy
- Confusion Matrix
- Loss curves
- Audio waveforms

## Exportar para Hugging Face

```bash
# Exportar modelo treinado
python scripts/export_to_huggingface.py \
  --model-path ./models/trained/final_model \
  --repo-name seu-usuario/cogitura-model

# Com push automático
python scripts/export_to_huggingface.py \
  --model-path ./models/trained/final_model \
  --repo-name seu-usuario/cogitura-model \
  --push
```

## Estrutura de Dados

### ElasticSearch - Índice de Sentenças

```json
{
  "sentence": "the cat is on the table",
  "sentence_hash": "a1b2c3d4...",
  "word_count": 6,
  "words": ["the", "cat", "is", "on", "table"],
  "audio_path": "/path/to/audio.mp3",
  "created_at": "2025-01-15T10:00:00",
  "language": "en"
}
```

### ElasticSearch - Índice de Palavras

```json
{
  "word": "cat",
  "word_hash": "x1y2z3...",
  "audio_path": "/path/to/word_cat.mp3",
  "created_at": "2025-01-15T10:00:00",
  "language": "en",
  "usage_count": 15
}
```

## Contribuindo

Contribuições são bem-vindas! Este é um projeto de pesquisa aberto.

Veja [CONTRIBUTING.md](CONTRIBUTING.md) para diretrizes detalhadas.

### Processo Rápido

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Adicione testes para suas mudanças
4. Execute os testes: `PYTHONPATH=. pytest`
5. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
6. Push para a branch (`git push origin feature/AmazingFeature`)
7. Abra um Pull Request

### Áreas que Precisam de Ajuda

- [ ] Suporte para múltiplos idiomas (PT, ES, FR, DE)
- [ ] Interface web com dashboard
- [ ] Mais provedores de IA (Mistral, Cohere, etc)
- [ ] Otimizações de performance
- [ ] Documentação adicional
- [ ] Benchmarks comparativos

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Citação

Se você usar este projeto em sua pesquisa, por favor cite:

```bibtex
@software{cogitura2025,
  author = {TheusHen},
  title = {Cogitura: AI Creating AI Research Project},
  year = {2025},
  url = {https://github.com/TheusHen/Cogitura},
  note = {A research project exploring AI-generated training data for speech recognition}
}
```

## Roadmap 2025

### ✅ Concluído
- [x] Fase 1: Geração de sentenças e TTS
- [x] Fase 2: Treinamento de modelo
- [x] Fase 3: Avaliação e análise
- [x] Módulo de dicionários online (Wiktionary, Datamuse, etc)
- [x] Sistema de testes completo (78 testes)
- [x] CI/CD com GitHub Actions
- [x] Documentação multilíngue

### 🚧 Em Desenvolvimento
- [ ] Interface web com dashboard interativo
- [ ] Suporte para múltiplos idiomas (PT-BR, ES, FR, DE)
- [ ] Integração com Mistral AI e Cohere
- [ ] Sistema de cache distribuído
- [ ] Benchmarks automatizados

### 🔮 Planejado
- [ ] Exportação automática para Hugging Face

## Contato e Suporte

- **GitHub**: [@TheusHen](https://github.com/TheusHen)
- **Issues**: [GitHub Issues](https://github.com/TheusHen/Cogitura/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TheusHen/Cogitura/discussions)

Link do Projeto: [https://github.com/TheusHen/Cogitura](https://github.com/TheusHen/Cogitura)

## Agradecimentos

- **OpenAI, Anthropic, Google** pelo acesso às APIs de IA
- **Hugging Face** pela biblioteca Transformers e plataforma
- **Elastic** pela plataforma ElasticSearch
- **Comunidade Python** e todo ecossistema open source
- **Contribuidores** que ajudam a melhorar o projeto

---

<div align="center">

**⭐ Se este projeto foi útil, considere dar uma estrela!**

Feito com ❤️ por [TheusHen](https://github.com/TheusHen)

© 2025 Cogitura Project

</div>
