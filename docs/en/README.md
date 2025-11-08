# Cogitura - Complete Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Installation Guide](#installation-guide)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Contributing](#contributing)

## Introduction

Cogitura is a research project exploring the question: **Can AIs create other AIs?**

The project implements a complete pipeline where a generative AI creates training data for a speech recognition AI, demonstrating how artificial intelligence can be used to bootstrap and train other AI systems.

### Research Goals

- Investigate the viability of AI-generated training data
- Compare performance across different AI providers
- Analyze the quality and diversity of generated sentences
- Measure the effectiveness of the trained speech recognition model
- Document findings for the research community

### Key Features

- Multiple AI provider support (OpenAI, Anthropic, Google, Ollama, Custom)
- Automated text-to-speech generation using gTTS
- ElasticSearch-based data management
- PyTorch-based speech recognition training
- Comprehensive evaluation metrics
- Easy export to Hugging Face

## Architecture

### System Overview

```
┌─────────────────┐
│  AI Providers   │
│ (OpenAI, etc.)  │
└────────┬────────┘
         │ generates
         ▼
┌─────────────────┐
│   Sentences     │
└────────┬────────┘
         │ splits into
         ▼
┌─────────────────┐     ┌──────────────┐
│     Words       │────▶│     gTTS     │
└────────┬────────┘     └──────┬───────┘
         │                     │
         │ stores              │ generates
         ▼                     ▼
┌─────────────────┐     ┌──────────────┐
│ ElasticSearch   │     │  Audio Files │
└────────┬────────┘     └──────┬───────┘
         │                     │
         └──────────┬──────────┘
                    │ trains
                    ▼
         ┌─────────────────────┐
         │   Speech-to-Text    │
         │       Model         │
         └──────────┬──────────┘
                    │ evaluates
                    ▼
         ┌─────────────────────┐
         │      Results        │
         │    & Analysis       │
         └─────────────────────┘
```

### Components

#### 1. Sentence Generator
- Connects to configured AI provider
- Generates diverse English sentences
- Validates sentence length and quality
- Extracts unique words

#### 2. TTS Processor
- Converts text to speech using gTTS
- Caches audio files to avoid regeneration
- Supports multiple languages
- Batch processing for efficiency

#### 3. Database Manager
- Manages ElasticSearch indices
- Prevents duplicate entries
- Tracks word usage statistics
- Enables efficient querying

#### 4. Model Trainer
- Uses Wav2Vec2 architecture
- Supports GPU acceleration
- Implements checkpointing
- Provides training metrics

#### 5. Model Evaluator
- Calculates accuracy metrics
- Computes Word Error Rate (WER)
- Generates detailed reports
- Exports results for analysis

## Installation Guide

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL2
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for data and models
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)

### Step-by-Step Installation

#### 1. Install System Dependencies

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
# Install WSL2 first, then:
sudo apt update
sudo apt install -y python3.9 python3-pip python3-venv docker.io docker-compose git
```

#### 2. Clone Repository

```bash
git clone https://github.com/TheusHen/Cogitura.git
cd Cogitura
```

#### 3. Create Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 4. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -e .
```

#### 5. Configure Environment

```bash
cp .env.example .env
nano .env  # Or use your preferred editor
```

Configure at least one AI provider:

```env
AI_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
```

#### 6. Start ElasticSearch

```bash
docker-compose up -d
```

Wait for ElasticSearch to start (about 30 seconds), then verify:

```bash
curl http://localhost:9200
```

You should see a JSON response with cluster information.

#### 7. Verify Installation

```bash
cogitura config-check
```

## Configuration

### Environment Variables

#### AI Provider Selection

```env
# Choose: openai, anthropic, google, ollama, custom
AI_PROVIDER=openai
```

#### OpenAI Configuration

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo
OPENAI_MAX_TOKENS=100
OPENAI_TEMPERATURE=0.7
```

#### Anthropic Configuration

```env
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-opus-20240229
ANTHROPIC_MAX_TOKENS=100
```

#### Google Configuration

```env
GOOGLE_API_KEY=...
GOOGLE_MODEL=gemini-pro
```

#### Ollama Configuration

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
```

First install Ollama:
```bash
# Linux
curl https://ollama.ai/install.sh | sh
ollama pull llama2

# macOS
brew install ollama
ollama pull llama2
```

#### Custom API Configuration

```env
CUSTOM_API_URL=http://localhost:8000/generate
CUSTOM_API_KEY=optional
```

Your API should accept POST requests with JSON body:
```json
{
  "prompt": "Generate a random English sentence:"
}
```

And return:
```json
{
  "text": "The generated sentence"
}
```

#### ElasticSearch Configuration

```env
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX_SENTENCES=cogitura_sentences
ELASTICSEARCH_INDEX_WORDS=cogitura_words
```

#### TTS Configuration

```env
TTS_LANGUAGE=en  # en, pt, es, fr, de, etc.
TTS_SLOW=false
TTS_OUTPUT_DIR=./data/audio
```

#### Training Configuration

```env
TRAINING_BATCH_SIZE=32
TRAINING_EPOCHS=50
TRAINING_LEARNING_RATE=0.001
MODEL_OUTPUT_DIR=./models/trained
```

#### Data Collection Settings

```env
MAX_SENTENCES_TO_GENERATE=10000
MIN_SENTENCE_LENGTH=5
MAX_SENTENCE_LENGTH=20
BATCH_SIZE_GENERATION=100
```

## Usage

### Command Line Interface

#### Check Configuration

```bash
cogitura config-check
```

#### Generate Sentences (Phase 1)

```bash
# Generate 1000 sentences
cogitura generate --count 1000 --save-db --generate-tts

# Options:
#   --count, -c: Number of sentences to generate
#   --batch-size, -b: Batch size for generation
#   --save-db: Save to ElasticSearch
#   --generate-tts: Generate audio files
```

#### Train Model (Phase 2)

```bash
# Train with default settings
cogitura train

# Train with custom settings
cogitura train --epochs 100 --batch-size 64 --model facebook/wav2vec2-large-960h

# Options:
#   --model, -m: Base model for training
#   --epochs, -e: Number of training epochs
#   --batch-size, -b: Training batch size
```

#### Evaluate Model (Phase 3)

```bash
# Evaluate trained model
cogitura evaluate ./models/trained/final_model

# Evaluate with sample size
cogitura evaluate ./models/trained/final_model --sample-size 500

# Options:
#   --sample-size, -s: Number of samples to use for evaluation
```

#### Database Management

```bash
# Show database statistics
cogitura db-stats

# Clear all data (use with caution!)
cogitura db-clear
```

### Python API

#### Generate Sentences

```python
from cogitura.core.sentence_generator import SentenceGenerator

generator = SentenceGenerator()

# Generate single sentence
sentence = generator.generate_single_sentence()
print(sentence)

# Generate multiple sentences
sentences = generator.generate_multiple(
    total_count=1000,
    batch_size=100,
    show_progress=True
)

# Get statistics
stats = generator.get_statistics()
print(f"Generated {stats['total_sentences']} sentences")
print(f"Unique words: {stats['unique_words']}")
```

#### Process Text-to-Speech

```python
from cogitura.core.tts_processor import TTSProcessor

tts = TTSProcessor(language='en')

# Convert single word
audio_path = tts.word_to_speech("hello")
print(f"Audio saved to: {audio_path}")

# Batch process words
words = ["hello", "world", "python"]
word_audios = tts.batch_process_words(words, show_progress=True)

# Batch process sentences
sentences = ["Hello world", "Python is great"]
sentence_audios = tts.batch_process_sentences(sentences)
```

#### Manage Database

```python
from cogitura.core.database_manager import DatabaseManager

db = DatabaseManager()

# Add sentence
db.add_sentence(
    sentence="Hello world",
    words=["hello", "world"],
    audio_path="/path/to/audio.mp3"
)

# Add word
db.add_word(
    word="hello",
    audio_path="/path/to/word_hello.mp3"
)

# Check if exists
exists = db.word_exists("hello")

# Get statistics
stats = db.get_statistics()
print(stats)

# Retrieve all sentences
sentences = db.get_all_sentences(limit=100)
```

#### Train Model

```python
from cogitura.core.trainer import ModelTrainer
from pathlib import Path

trainer = ModelTrainer(model_name="facebook/wav2vec2-base-960h")

# Prepare data
audio_paths = [Path("audio1.mp3"), Path("audio2.mp3")]
texts = ["hello world", "python programming"]

train_loader, val_loader = trainer.prepare_data(audio_paths, texts)

# Train
history = trainer.train(
    train_loader,
    val_loader,
    epochs=50,
    learning_rate=0.001
)

# Save model
trainer.save_model(name="my_model")
```

#### Evaluate Model

```python
from cogitura.core.evaluator import ModelEvaluator
from pathlib import Path

evaluator = ModelEvaluator(Path("./models/trained/final_model"))

# Transcribe single audio
transcription = evaluator.transcribe_audio(Path("test.mp3"))
print(transcription)

# Evaluate dataset
audio_paths = [Path("test1.mp3"), Path("test2.mp3")]
references = ["hello world", "python programming"]

metrics = evaluator.evaluate_dataset(audio_paths, references)
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"WER: {metrics['wer']:.2%}")

# Generate report
report = evaluator.generate_report(metrics)
print(report)
```

## API Reference

See [API Reference](API.md) for detailed documentation of all classes and methods.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../../LICENSE) for details.
