"""
Script para executar pipeline completo
"""
import argparse
from pathlib import Path
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.tts_processor import TTSProcessor
from cogitura.core.database_manager import DatabaseManager
from cogitura.core.trainer import ModelTrainer
from cogitura.core.evaluator import ModelEvaluator
from cogitura.utils import split_sentence_into_words
from cogitura.logger import log
from cogitura.config import Config


def run_full_pipeline(
    num_sentences: int = 1000,
    train_epochs: int = 50,
    eval_sample_size: int = 500
):
    """
    Executa pipeline completo do Cogitura
    
    Args:
        num_sentences: Número de sentenças a gerar
        train_epochs: Número de épocas de treinamento
        eval_sample_size: Tamanho da amostra para avaliação
    """
    log.info("=== INICIANDO PIPELINE COMPLETO DO COGITURA ===")
    
    # Fase 1: Geração de Dados
    log.info("\n=== FASE 1: GERAÇÃO DE DADOS ===")
    
    generator = SentenceGenerator()
    log.info(f"Gerando {num_sentences} sentenças...")
    sentences = generator.generate_multiple(num_sentences, show_progress=True)
    
    stats = generator.get_statistics()
    log.info(f"Sentenças geradas: {stats['total_sentences']}")
    log.info(f"Palavras únicas: {stats['unique_words']}")
    
    # TTS
    log.info("Gerando TTS para palavras...")
    tts = TTSProcessor()
    word_audios = tts.batch_process_words(list(generator.unique_words), show_progress=True)
    
    log.info("Gerando TTS para sentenças...")
    sentence_audios = tts.batch_process_sentences(sentences, show_progress=True)
    
    # Salvar no DB
    log.info("Salvando dados no ElasticSearch...")
    db = DatabaseManager()
    
    sentences_data = []
    for sentence in sentences:
        words = split_sentence_into_words(sentence)
        sentences_data.append({
            "sentence": sentence,
            "words": words,
            "audio_path": str(sentence_audios.get(sentence, ""))
        })
    
    db.bulk_add_sentences(sentences_data)
    
    words_data = [
        {"word": word, "audio_path": str(word_audios.get(word, ""))}
        for word in generator.unique_words
    ]
    db.bulk_add_words(words_data)
    
    log.info("Dados salvos com sucesso!")
    
    # Fase 2: Treinamento
    log.info("\n=== FASE 2: TREINAMENTO DO MODELO ===")
    
    # Prepara dados
    audio_paths = [Path(sentence_audios[s]) for s in sentences if s in sentence_audios]
    texts = [s for s in sentences if s in sentence_audios]
    
    log.info(f"Dados de treinamento: {len(audio_paths)} amostras")
    
    trainer = ModelTrainer()
    train_loader, val_loader = trainer.prepare_data(audio_paths, texts)
    
    log.info(f"Iniciando treinamento por {train_epochs} épocas...")
    history = trainer.train(train_loader, val_loader, epochs=train_epochs)
    
    log.info("Treinamento concluído!")
    
    # Fase 3: Avaliação
    log.info("\n=== FASE 3: AVALIAÇÃO DO MODELO ===")
    
    model_path = Config.MODEL_OUTPUT_DIR / "final_model"
    evaluator = ModelEvaluator(model_path)
    
    # Usa amostra para avaliação
    eval_audio_paths = audio_paths[:eval_sample_size]
    eval_texts = texts[:eval_sample_size]
    
    log.info(f"Avaliando modelo em {len(eval_audio_paths)} amostras...")
    metrics = evaluator.evaluate_dataset(eval_audio_paths, eval_texts)
    
    report = evaluator.generate_report(metrics)
    print("\n" + report)
    
    log.info("\n=== PIPELINE COMPLETO CONCLUÍDO ===")
    log.info(f"Acurácia Final: {metrics['accuracy']:.2%}")
    log.info(f"WER Final: {metrics['wer']:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Executar pipeline completo")
    parser.add_argument("--sentences", type=int, default=1000, help="Número de sentenças")
    parser.add_argument("--epochs", type=int, default=50, help="Épocas de treinamento")
    parser.add_argument("--eval-size", type=int, default=500, help="Tamanho da amostra de avaliação")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        num_sentences=args.sentences,
        train_epochs=args.epochs,
        eval_sample_size=args.eval_size
    )


if __name__ == "__main__":
    main()
