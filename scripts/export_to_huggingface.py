"""
Script para exportar modelo treinado para o Hugging Face Hub
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from cogitura.logger import log


def export_to_huggingface(
    model_path: Path,
    repo_name: str,
    token: str,
    organization: str = None,
    private: bool = False
):
    """
    Exporta modelo para Hugging Face Hub
    
    Args:
        model_path: Caminho do modelo local
        repo_name: Nome do repositório no HF
        token: Token de acesso do Hugging Face
        organization: Organização (opcional)
        private: Se o repo deve ser privado
    """
    log.info(f"Exportando modelo {model_path} para Hugging Face")
    
    # Cria API
    api = HfApi()
    
    # Nome completo do repo
    full_repo_name = f"{organization}/{repo_name}" if organization else repo_name
    
    # Cria repositório
    try:
        create_repo(
            repo_id=full_repo_name,
            token=token,
            private=private,
            repo_type="model"
        )
        log.info(f"Repositório criado: {full_repo_name}")
    except Exception as e:
        log.warning(f"Repositório pode já existir: {e}")
    
    # Upload dos arquivos
    log.info("Fazendo upload dos arquivos...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=full_repo_name,
        repo_type="model",
        token=token
    )
    
    log.info(f"Modelo exportado com sucesso para: https://huggingface.co/{full_repo_name}")


def main():
    parser = argparse.ArgumentParser(description="Exportar modelo para Hugging Face")
    parser.add_argument("--model-path", type=Path, required=True, help="Caminho do modelo")
    parser.add_argument("--repo-name", required=True, help="Nome do repositório")
    parser.add_argument("--token", required=True, help="Token do Hugging Face")
    parser.add_argument("--organization", help="Organização (opcional)")
    parser.add_argument("--private", action="store_true", help="Repositório privado")
    
    args = parser.parse_args()
    
    export_to_huggingface(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        organization=args.organization,
        private=args.private
    )


if __name__ == "__main__":
    main()
