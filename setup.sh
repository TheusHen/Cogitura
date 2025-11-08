#!/bin/bash

# Script de inicialização do Cogitura
# Este script configura o ambiente e valida a instalação

set -e

echo "======================================"
echo "   Cogitura - Setup Inicial"
echo "======================================"
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para printar com cor
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Verifica Python
echo "Verificando Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python $PYTHON_VERSION encontrado"
else
    print_error "Python 3 não encontrado. Instale Python 3.9+"
    exit 1
fi

# Verifica Docker
echo "Verificando Docker..."
if command -v docker &> /dev/null; then
    print_success "Docker encontrado"
else
    print_warning "Docker não encontrado. ElasticSearch não poderá ser iniciado"
fi

# Verifica se venv existe
echo ""
echo "Configurando ambiente virtual..."
if [ ! -d "venv" ]; then
    echo "Criando ambiente virtual..."
    python3 -m venv venv
    print_success "Ambiente virtual criado"
else
    print_success "Ambiente virtual já existe"
fi

# Ativa venv
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Instala dependências
echo ""
echo "Instalando dependências..."
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
print_success "Dependências instaladas"

# Cria .env se não existir
echo ""
if [ ! -f ".env" ]; then
    echo "Criando arquivo .env..."
    cp .env.example .env
    print_warning "Configure suas chaves de API em .env"
else
    print_success "Arquivo .env já existe"
fi

# Inicia ElasticSearch
echo ""
if command -v docker &> /dev/null; then
    echo "Iniciando ElasticSearch..."
    docker-compose up -d > /dev/null 2>&1
    
    echo "Aguardando ElasticSearch iniciar (30s)..."
    sleep 30
    
    # Testa conexão
    if curl -s http://localhost:9200 > /dev/null; then
        print_success "ElasticSearch está rodando"
    else
        print_warning "ElasticSearch pode não estar pronto ainda"
    fi
fi

# Verifica configuração
echo ""
echo "Verificando configuração..."
if python -c "from cogitura.config import Config; is_valid, errors = Config.validate_config(); exit(0 if is_valid else 1)" 2>/dev/null; then
    print_success "Configuração válida"
else
    print_warning "Configure suas chaves de API no arquivo .env"
fi

echo ""
echo "======================================"
echo "   Setup Concluído!"
echo "======================================"
echo ""
echo "Próximos passos:"
echo "  1. Configure o arquivo .env com suas chaves de API"
echo "  2. Execute: source venv/bin/activate"
echo "  3. Execute: cogitura config-check"
echo "  4. Comece a usar: cogitura generate --count 100 --save-db --generate-tts"
echo ""
echo "Documentação: docs/pt-br/README.md"
echo "Guia rápido: QUICKSTART.md"
echo ""
