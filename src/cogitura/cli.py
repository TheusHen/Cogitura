"""
Interface de linha de comando para o Cogitura
"""
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from cogitura.config import Config
from cogitura.core.database_manager import DatabaseManager
from cogitura.core.evaluator import ModelEvaluator
from cogitura.core.sentence_generator import SentenceGenerator
from cogitura.core.trainer import ModelTrainer
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
    sentences = generator.generate_multiple(count, batch_size, show_progress=True)

    # Estatísticas
    stats = generator.get_statistics()
    console.print("\n[bold]Estatísticas:[/bold]")
    console.print(f"  Sentenças geradas: {stats['total_sentences']}")
    console.print(f"  Palavras únicas: {stats['unique_words']}")
    console.print(f"  Média de palavras por sentença: {stats['avg_words_per_sentence']:.2f}")

    # Salva no DB
    if save_db:
        console.print("\n[bold]Salvando no ElasticSearch...[/bold]")
        db = DatabaseManager()

        sentences_data = []
        for sentence in sentences:
            words = split_sentence_into_words(sentence)
            sentences_data.append({"sentence": sentence, "words": words})

        db.bulk_add_sentences(sentences_data)

        # Adiciona palavras únicas
        words_data = [{"word": word} for word in stats["words_list"]]
        db.bulk_add_words(words_data)

        console.print("[bold green]Dados salvos com sucesso![/bold green]")

    # Gera TTS
    if generate_tts:
        console.print("\n[bold]Gerando TTS...[/bold]")
        tts = TTSProcessor()

        # TTS para palavras
        tts.batch_process_words(list(generator.unique_words), show_progress=True)

        # TTS para sentenças
        tts.batch_process_sentences(sentences, show_progress=True)

        tts_stats = tts.get_statistics()
        console.print(
            "[bold green]TTS gerado: "
            f"{tts_stats['total_files']} arquivos, "
            f"{tts_stats['total_size_mb']:.2f} MB[/bold green]"
        )


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
    sentences_data = db.get_all_sentences()

    if not sentences_data:
        console.print("[bold red]Nenhum dado encontrado no banco de dados![/bold red]")
        console.print("Execute primeiro: cogitura generate --save-db --generate-tts")
        return

    # Prepara paths
    audio_paths = []
    texts = []

    for item in sentences_data:
        if item.get("audio_path"):
            audio_paths.append(Path(item["audio_path"]))
            texts.append(item["sentence"])

    console.print(f"Encontradas {len(audio_paths)} amostras com áudio")

    if len(audio_paths) < 10:
        console.print("[bold yellow]Aviso: Poucos dados para treinamento![/bold yellow]")

    # Treina
    trainer = ModelTrainer(model_name=model)
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
    evaluator = ModelEvaluator(Path(model_path))
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
