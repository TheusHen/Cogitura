# Guia de Início Rápido - Cogitura

> Última atualização: Janeiro 2025 | Versão 0.2.0

## ⚡ Instalação em 5 Minutos

### 1. Pré-requisitos
```bash
# Verifique Python 3.9+
python --version

# Verifique Docker
docker --version
```

### 2. Clone e Configure
```bash
git clone https://github.com/TheusHen/Cogitura.git
cd Cogitura

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -e .
```

### 3. Configure Variáveis de Ambiente
```bash
cp .env.example .env
nano .env  # ou use seu editor favorito
```

Exemplo mínimo do `.env`:
```env
# Escolha seu provedor de IA
AI_PROVIDER=openai

# Configure a chave correspondente
OPENAI_API_KEY=sk-seu-token-aqui
```

### 4. Inicie ElasticSearch
```bash
# Inicia o container
docker-compose up -d

# Aguarde ~30 segundos para inicialização
sleep 30

# Teste a conexão
curl http://localhost:9200
```

### 5. Execute seu Primeiro Experimento ✨
```bash
# Verifica configuração
cogitura config-check

# Gera 100 sentenças com TTS
cogitura generate --count 100 --save-db --generate-tts

# Veja os resultados
cogitura db-stats
```

🎉 **Pronto!** Você acabou de gerar seu primeiro dataset de treinamento criado por IA.


## 📋 Comandos Essenciais

```bash
# Verificar configuração
cogitura config-check

# Gerar dados (1000 sentenças com áudio)
cogitura generate --count 1000 --save-db --generate-tts

# Treinar modelo (50 épocas)
cogitura train --epochs 50

# Avaliar modelo treinado
cogitura evaluate ./models/trained/final_model --sample-size 500

# Ver estatísticas do banco
cogitura db-stats

# Limpar todos os dados (CUIDADO!)
cogitura db-clear
```

## 🐍 Uso Python Básico

### Geração de Sentenças

```python
from cogitura.core.sentence_generator import SentenceGenerator

# Inicializa o gerador
generator = SentenceGenerator()

# Gera múltiplas sentenças
sentences = generator.generate_sentences(count=100)

print(f"✅ Geradas {len(sentences)} sentenças")
print(f"📝 Palavras únicas: {len(generator.unique_words)}")

# Extrai palavras de uma sentença
words = generator.extract_words("Hello world, this is a test!")
print(f"Palavras: {words}")
```

### Usando Dicionários (🆕 2025)

```python
from cogitura.core.dictionary_sources import fetch_definitions

# Busca definições de múltiplas fontes
defs = fetch_definitions('python', sources=['wiktionary', 'datamuse', 'free_dictionary'])

for source, definitions in defs.items():
    print(f"\n{source}:")
    for d in definitions[:3]:  # Primeiras 3
        print(f"  - {d}")
```

### Pipeline Completo

```python
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.tts_processor import TTSProcessor
from cogitura.core.database_manager import DatabaseManager

# 1. Gera sentenças
generator = SentenceGenerator()
sentences = generator.generate_sentences(100)

# 2. Processa TTS
tts = TTSProcessor()
for sentence in sentences:
    audio_path = tts.process_sentence(sentence)
    print(f"Áudio: {audio_path}")

# 3. Salva no banco
db = DatabaseManager()
for sentence in sentences:
    words = generator.extract_words(sentence)
    db.add_sentence(sentence, words)

print(f"✅ Pipeline completo: {len(sentences)} sentenças processadas")
```

## 🤖 Provedores de IA Suportados

### OpenAI (GPT-4, GPT-3.5)
```env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4  # ou gpt-3.5-turbo
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
```bash
# Instale Ollama
curl https://ollama.ai/install.sh | sh

# Baixe um modelo
ollama pull llama2
# ou
ollama pull mistral
```

```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

### API Customizada
```env
AI_PROVIDER=custom
CUSTOM_API_URL=http://localhost:8000/generate
CUSTOM_API_KEY=opcional
```

## 🧪 Executar Testes

```bash
# Todos os testes (78 testes)
PYTHONPATH=. pytest

# Com relatório de cobertura
PYTHONPATH=. pytest --cov=src/cogitura --cov-report=html

# Abrir relatório HTML
open htmlcov/index.html
```

## 🐛 Troubleshooting

### ElasticSearch não conecta
```bash
# Reinicie os containers
docker-compose down
docker-compose up -d

# Aguarde inicialização
sleep 30

# Verifique logs
docker-compose logs elasticsearch

# Teste conexão
curl -X GET "localhost:9200/_cluster/health?pretty"
```

### Erro de API Key Inválida
```bash
# Verifique configuração
cogitura config-check

# Certifique-se que o .env existe e está correto
cat .env | grep API_KEY

# Teste API key manualmente
export OPENAI_API_KEY=sk-...
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### Módulo não encontrado
```bash
# Reinstale em modo editável
pip install -e .

# Ou use PYTHONPATH
export PYTHONPATH=.
```

### Testes falhando
```bash
# Certifique-se que está no ambiente virtual
which python

# Reinstale dependências de desenvolvimento
pip install -e ".[dev]"

# Execute com PYTHONPATH
PYTHONPATH=. pytest -v
```

## 📚 Próximos Passos

1. ✅ **Leia a documentação completa**: [README.md](README.md)
2. 🧪 **Explore os testes**: [TEST_RESULTS.md](TEST_RESULTS.md)
3. 🤝 **Contribua**: [CONTRIBUTING.md](CONTRIBUTING.md)
4. 📝 **Veja mudanças**: [CHANGELOG.md](CHANGELOG.md)

## 🆘 Ajuda

- **Issues**: [GitHub Issues](https://github.com/TheusHen/Cogitura/issues)
- **Discussions**: [GitHub Discussions](https://github.com/TheusHen/Cogitura/discussions)
- **Email**: Via GitHub profile

---

💡 **Dica**: Execute `cogitura --help` para ver todos os comandos disponíveis.

🌟 Se este projeto foi útil, considere dar uma estrela no GitHub!


### Erro de memória ao treinar
```env
# Reduza batch size no .env
TRAINING_BATCH_SIZE=16  # Era 32
```

## Próximos Passos

1. Leia a [documentação completa](docs/pt-br/README.md)
2. Explore os [scripts de exemplo](scripts/)
3. Execute o [pipeline completo](scripts/run_full_pipeline.py)
4. Contribua com o projeto!

## Suporte

- Issues: https://github.com/TheusHen/Cogitura/issues
- Documentação: [docs/](docs/)
