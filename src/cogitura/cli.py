"""
Interface de linha de comando para o Cogitura
"""
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from cogitura.config import Config
from cogitura.core.database_manager import DatabaseManager
from cogitura.core.evaluator import Evaluator
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.trainer import Trainer
from cogitura.core.tts_processor import TTSProcessor
from cogitura.utils import split_sentence_into_words

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Cogitura - AI Creating AI Research Project"""
    Config.create_directories()


@main.command()
@click.option("--count", "-c", default=100, help="Número de sentenças para gerar")
@click.option("--batch-size", "-b", default=None, type=int, help="Tamanho do lote")
@click.option("--save-db", is_flag=True, help="Salvar no ElasticSearch")
@click.option("--generate-tts", is_flag=True, help="Gerar TTS para sentenças e palavras")
def generate(count, batch_size, save_db, generate_tts):
    """Gera sentenças usando IA (Fase 1)"""
    console.print(f"[bold green]Gerando {count} sentenças...[/bold green]")

    # Valida configuração
    is_valid, errors = Config.validate_config()
    if not is_valid:
        console.print("[bold red]Erro na configuração:[/bold red]")
        for error in errors:
            console.print(f"  - {error}")
        return

    # Gera sentenças
    generator = SentenceGenerator()
    sentences = generator.generate_batch(count)

    # Estatísticas simples
    console.print(f"\n[bold]Estatísticas:[/bold]")
    console.print(f"  Sentenças geradas: {len(sentences)}")

    # Salva no DB
    if save_db:
        console.print("\n[bold]Salvando no ElasticSearch...[/bold]")
        db = DatabaseManager()

        sentences_data = []
        all_words = set()
        for sentence in sentences:
            words = split_sentence_into_words(sentence)
            sentences_data.append({"sentence": sentence, "words": words})
            all_words.update(words)

        db.bulk_add_sentences(sentences_data)

        # Adiciona palavras únicas
        words_data = [{"word": word} for word in all_words]
        db.bulk_add_words(words_data)

        console.print("[bold green]Dados salvos com sucesso![/bold green]")

    # Gera TTS
    if generate_tts:
        console.print("\n[bold]Gerando TTS...[/bold]")
        tts = TTSProcessor()

        # Extrai palavras únicas das sentenças
        all_words = set()
        for sentence in sentences:
            words = split_sentence_into_words(sentence)
            all_words.update(words)

        # TTS para palavras
        console.print(f"Gerando áudio para {len(all_words)} palavras únicas...")
        tts.batch_process_words(list(all_words), show_progress=False)

        # TTS para sentenças
        console.print(f"Gerando áudio para {len(sentences)} sentenças...")
        tts.batch_process_sentences(sentences, show_progress=False)

        console.print("[bold green]TTS gerado com sucesso![/bold green]")


@main.command()
@click.option(
    "--model", "-m", default="facebook/wav2vec2-base-960h", help="Modelo base para treinamento"
)
@click.option("--epochs", "-e", default=None, type=int, help="Número de épocas")
@click.option("--batch-size", "-b", default=None, type=int, help="Tamanho do lote")
def train(model, epochs, batch_size):
    """Treina modelo de Speech-to-Text (Fase 2)"""
    console.print("[bold green]Iniciando treinamento...[/bold green]")

    # Busca dados do DB
    console.print("Carregando dados do ElasticSearch...")
    db = DatabaseManager()
    
    # Busca sentenças com seus metadados completos
    query = {"query": {"match_all": {}}, "size": 10000}
    result = db.es.search(index=db.sentences_index, body=query)
    sentences_data = [hit["_source"] for hit in result.get("hits", {}).get("hits", [])]

    if not sentences_data:
        console.print("[bold red]Nenhum dado encontrado no banco de dados![/bold red]")
        console.print("Execute primeiro: cogitura generate --save-db --generate-tts")
        return

    # Prepara paths
    audio_paths = []
    texts = []

    for item in sentences_data:
        # Gera o path do áudio baseado no hash da sentença
        sentence_text = item.get("sentence") or item.get("text")
        if sentence_text:
            # Usa o hash da sentença (primeiros 16 caracteres)
            sentence_hash = item.get("sentence_hash", "")[:16]
            
            audio_file = Path(Config.AUDIO_DIR) / f"sentence_{sentence_hash}mp3.mp3"
            if audio_file.exists():
                audio_paths.append(audio_file)
                texts.append(sentence_text.strip())

    console.print(f"Encontradas {len(audio_paths)} amostras com áudio")

    if len(audio_paths) < 10:
        console.print("[bold yellow]Aviso: Poucos dados para treinamento![/bold yellow]")
        
    if len(audio_paths) == 0:
        console.print("[bold red]Nenhum áudio encontrado![/bold red]")
        console.print(f"Verifique o diretório: {Config.AUDIO_DIR}")
        return

    # Treina
    config = {"model_name": model, "epochs": epochs, "batch_size": batch_size}
    trainer = Trainer(config=config)
    train_loader, val_loader = trainer.prepare_data(audio_paths, texts)

    trainer.train(train_loader, val_loader, epochs=epochs)

    console.print("[bold green]Treinamento concluído![/bold green]")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--sample-size", "-s", default=None, type=int, help="Tamanho da amostra de teste")
def evaluate(model_path, sample_size):
    """Avalia modelo treinado (Fase 3)"""
    console.print("[bold green]Avaliando modelo...[/bold green]")

    # Busca dados de teste
    console.print("Carregando dados do ElasticSearch...")
    db = DatabaseManager()
    sentences_data = db.get_all_sentences(limit=sample_size)

    if not sentences_data:
        console.print("[bold red]Nenhum dado encontrado![/bold red]")
        return

    # Prepara dados
    audio_paths = []
    texts = []

    for item in sentences_data:
        if item.get("audio_path"):
            audio_paths.append(Path(item["audio_path"]))
            texts.append(item["sentence"])

    console.print(f"Avaliando em {len(audio_paths)} amostras")

    # Avalia
    evaluator = Evaluator(Path(model_path))
    metrics = evaluator.evaluate_dataset(audio_paths, texts)

    # Gera e exibe relatório
    report = evaluator.generate_report(metrics)
    console.print(report)


@main.command()
def db_stats():
    """Mostra estatísticas do banco de dados"""
    db = DatabaseManager()
    stats = db.get_statistics()

    table = Table(title="Estatísticas do ElasticSearch")
    table.add_column("Métrica", style="cyan")
    table.add_column("Valor", style="green")

    table.add_row("Total de Sentenças", str(stats["total_sentences"]))
    table.add_row("Total de Palavras", str(stats["total_words"]))
    table.add_row("Host", stats["elasticsearch_host"])
    table.add_row("Índice de Sentenças", stats["sentences_index"])
    table.add_row("Índice de Palavras", stats["words_index"])

    console.print(table)


@main.command()
@click.confirmation_option(prompt="Tem certeza que deseja limpar todos os dados?")
def db_clear():
    """Limpa todos os dados do banco de dados"""
    db = DatabaseManager()
    db.clear_all_data()
    console.print("[bold green]Dados removidos com sucesso![/bold green]")


@main.command()
def config_check():
    """Verifica a configuração do projeto"""
    is_valid, errors = Config.validate_config()

    if is_valid:
        console.print("[bold green]✓ Configuração válida![/bold green]")
    else:
        console.print("[bold red]✗ Erros na configuração:[/bold red]")
        for error in errors:
            console.print(f"  - {error}")

    # Mostra configurações
    table = Table(title="Configurações Atuais")
    table.add_column("Configuração", style="cyan")
    table.add_column("Valor", style="yellow")

    table.add_row("AI Provider", Config.AI_PROVIDER)
    table.add_row("TTS Language", Config.TTS_LANGUAGE)
    table.add_row("ElasticSearch", f"{Config.ELASTICSEARCH_HOST}:{Config.ELASTICSEARCH_PORT}")
    table.add_row("Diretório de Áudio", str(Config.AUDIO_DIR))
    table.add_row("Diretório de Modelos", str(Config.MODELS_DIR))

    console.print(table)


if __name__ == "__main__":
    main()
