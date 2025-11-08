#!/usr/bin/env python3
"""
🚀 GERAÇÃO MASSIVA OTIMIZADA - COGITURA PTBR-10K
Gera 10.000 sentenças com progresso detalhado em tempo real
"""
import os
import sys
import time
from pathlib import Path

# Adiciona o projeto ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.database_manager import DatabaseManager
from cogitura.core.tts_processor import TTSProcessor
from cogitura.utils import split_sentence_into_words

# Configurações
TARGET_SENTENCES = 10000
BATCH_SIZE = 100  # Sentenças por batch (maior = mais eficiente com Ollama)

def main():
    print("🚀 COGITURA PTBR-10K - GERAÇÃO MASSIVA OTIMIZADA")
    print("=" * 70)
    print(f"🎯 Meta: {TARGET_SENTENCES:,} sentenças em português")
    print(f"📦 Batch size: {BATCH_SIZE} sentenças/batch")
    print(f"🔢 Total batches: {TARGET_SENTENCES // BATCH_SIZE}")
    print("=" * 70)
    print()
    
    total_batches = TARGET_SENTENCES // BATCH_SIZE
    start_time = time.time()
    total_sentences_generated = 0
    total_words_generated = 0
    
    # Inicializa componentes uma vez
    generator = SentenceGenerator()
    db = DatabaseManager()
    tts = TTSProcessor()
    
    # Executa batches sequencialmente
    for batch_num in range(1, total_batches + 1):
        batch_start = time.time()
        
        try:
            # Gera sentenças
            print(f"🤖 Batch {batch_num}/{total_batches}: Gerando {BATCH_SIZE} sentenças...", end="", flush=True)
            sentences = generator.generate_batch(BATCH_SIZE)
            print(f" ✅ ({len(sentences)} geradas)")
            
            # Salva no DB
            print(f"💾 Batch {batch_num}/{total_batches}: Salvando no ElasticSearch...", end="", flush=True)
            all_words = set()
            sentences_data = []
            
            for sentence in sentences:
                words = split_sentence_into_words(sentence)
                sentences_data.append({"sentence": sentence, "words": words})
                all_words.update(words)
            
            db.bulk_add_sentences(sentences_data)
            words_data = [{"word": word} for word in all_words]
            db.bulk_add_words(words_data)
            print(f" ✅ ({len(all_words)} palavras únicas)")
            
            # Gera TTS
            print(f"🎵 Batch {batch_num}/{total_batches}: Gerando áudio TTS...", end="", flush=True)
            tts.batch_process_words(list(all_words), show_progress=False)
            tts.batch_process_sentences(sentences, show_progress=False)
            print(" ✅")
            
            # Estatísticas do batch
            batch_time = time.time() - batch_start
            total_sentences_generated += len(sentences)
            total_words_generated += len(all_words)
            
            elapsed = time.time() - start_time
            progress = (batch_num / total_batches) * 100
            avg_time_per_batch = elapsed / batch_num
            eta_seconds = avg_time_per_batch * (total_batches - batch_num)
            eta_minutes = eta_seconds / 60
            sentences_per_sec = total_sentences_generated / elapsed
            
            print(f"📊 Batch {batch_num}/{total_batches} COMPLETO em {batch_time:.1f}s")
            print(f"   ├─ Progresso: {progress:.1f}%")
            print(f"   ├─ Total acumulado: {total_sentences_generated:,} sentenças, {total_words_generated:,} palavras")
            print(f"   ├─ Velocidade: {sentences_per_sec:.1f} sentenças/seg")
            print(f"   └─ ETA: {eta_minutes:.1f} minutos")
            print()
            
        except Exception as e:
            print(f" ❌ ERRO: {e}")
            print(f"⚠️  Batch {batch_num} falhou, continuando...")
            print()
    
    # Estatísticas finais
    total_time = time.time() - start_time
    print()
    print("=" * 70)
    print("🎉 GERAÇÃO COMPLETA!")
    print("=" * 70)
    print(f"✅ Sentenças geradas: {total_sentences_generated:,}")
    print(f"✅ Palavras únicas: {total_words_generated:,}")
    print(f"⏱️  Tempo total: {total_time/60:.1f} minutos ({total_time:.1f}s)")
    print(f"⚡ Velocidade: {total_sentences_generated/total_time:.1f} sentenças/segundo")
    print(f"💪 Throughput: {(total_sentences_generated * 60)/total_time:.0f} sentenças/minuto")
    print("=" * 70)
    print()
    print("🚀 Pronto para Fase 2: Treinamento!")
    print(f"   Comando: cogitura train --model facebook/wav2vec2-base --epochs 10")

if __name__ == "__main__":
    main()
