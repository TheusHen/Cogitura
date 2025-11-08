"""
Script para análise de dados do ElasticSearch
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from cogitura.core.database_manager import DatabaseManager
from cogitura.logger import log


def analyze_data(output_dir: Path = None):
    """
    Analisa dados do ElasticSearch e gera visualizações
    
    Args:
        output_dir: Diretório para salvar gráficos
    """
    if output_dir is None:
        output_dir = Path("./data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Conectando ao ElasticSearch...")
    db = DatabaseManager()
    
    # Carrega dados
    log.info("Carregando sentenças...")
    sentences_data = db.get_all_sentences()
    
    log.info("Carregando palavras...")
    words_data = db.get_all_words()
    
    # Cria DataFrames
    df_sentences = pd.DataFrame(sentences_data)
    df_words = pd.DataFrame(words_data)
    
    log.info(f"Sentenças: {len(df_sentences)}, Palavras: {len(df_words)}")
    
    # Análise de sentenças
    plt.figure(figsize=(12, 6))
    
    # Distribuição de tamanho de sentenças
    plt.subplot(1, 2, 1)
    df_sentences['word_count'].hist(bins=30, edgecolor='black')
    plt.xlabel('Número de Palavras')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Tamanho das Sentenças')
    plt.grid(axis='y', alpha=0.5)
    
    # Top 20 palavras mais usadas
    plt.subplot(1, 2, 2)
    top_words = df_words.nlargest(20, 'usage_count')
    plt.barh(top_words['word'], top_words['usage_count'])
    plt.xlabel('Frequência de Uso')
    plt.title('Top 20 Palavras Mais Usadas')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plot_path = output_dir / 'data_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    log.info(f"Gráfico salvo em: {plot_path}")
    
    # Salva estatísticas
    stats = {
        'total_sentences': len(df_sentences),
        'total_words': len(df_words),
        'avg_sentence_length': df_sentences['word_count'].mean(),
        'median_sentence_length': df_sentences['word_count'].median(),
        'max_sentence_length': df_sentences['word_count'].max(),
        'min_sentence_length': df_sentences['word_count'].min(),
        'avg_word_usage': df_words['usage_count'].mean(),
        'most_used_word': df_words.nlargest(1, 'usage_count')['word'].values[0],
        'most_used_word_count': df_words.nlargest(1, 'usage_count')['usage_count'].values[0]
    }
    
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=== Estatísticas do Dataset ===\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    log.info(f"Estatísticas salvas em: {stats_path}")
    
    # Salva CSVs
    df_sentences.to_csv(output_dir / 'sentences.csv', index=False)
    df_words.to_csv(output_dir / 'words.csv', index=False)
    log.info("CSVs salvos")
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Análise de dados")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretório de saída")
    
    args = parser.parse_args()
    
    stats = analyze_data(args.output_dir)
    
    print("\n=== Estatísticas ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
