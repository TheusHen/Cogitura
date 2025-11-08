# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Planejado
- Interface web com dashboard
- Suporte para múltiplos idiomas completo
- API REST
- Deployment configs para Kubernetes

## [0.2.0] - 2025-01-15

### Adicionado
- Módulo `dictionary_sources` com 5 fontes de dicionários online
  - Wiktionary (scraping HTML)
  - Datamuse API
  - Free Dictionary API
  - Wordnik API (opcional)
  - WordNet (NLTK)
- 78 testes unitários completos (100% passando)
- Sistema de CI/CD robusto com GitHub Actions
- Workflows para lint, type checking, coverage
- Cobertura de código em 57%
- Documentação atualizada para 2025

### Melhorado
- Compatibilidade de API entre módulos e testes
- DatabaseManager com métodos adicionais de compatibilidade
- Trainer e Evaluator alinhados com expectativas de testes
- Providers de IA com melhor suporte a mocking
- Tratamento de erros mais robusto em todas as camadas

### Corrigido
- Estrutura quebrada no README
- Datas antigas (2024) atualizadas para 2025
- Importações circulares em módulos core
- Problemas de patching em testes
- Inconsistências em assinaturas de métodos

## [0.1.0] - 2024-11-08

### Adicionado
- Estrutura inicial do projeto
- Suporte para múltiplos provedores de IA (OpenAI, Anthropic, Google, Ollama, Custom)
- Gerador de sentenças com IA
- Processador TTS usando gTTS
- Gerenciador de banco de dados com ElasticSearch
- Módulo de treinamento de modelo Speech-to-Text
- Módulo de avaliação e métricas
- Interface CLI completa
- Documentação em múltiplas línguas (PT-BR, EN, ES)
- Scripts utilitários
- Testes unitários básicos
- Docker Compose para ElasticSearch
- Sistema de logging com Loguru
- Exportação para Hugging Face

### Características
- Pipeline completo de geração de dados até avaliação
- Suporte a GPU para treinamento
- Cache de arquivos de áudio
- Prevenção de duplicatas no banco de dados
- Estatísticas e análise de dados
- Checkpointing durante treinamento
- Relatórios detalhados de avaliação

[Unreleased]: https://github.com/TheusHen/Cogitura/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/TheusHen/Cogitura/releases/tag/v0.2.0
[0.1.0]: https://github.com/TheusHen/Cogitura/releases/tag/v0.1.0
