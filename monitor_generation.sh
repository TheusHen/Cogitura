#!/bin/bash

# Monitor de Geração - Cogitura PTBR-10K
# Monitora progresso da geração massiva de sentenças

clear
echo "🤖 COGITURA PTBR-10K - Monitor de Geração"
echo "=========================================="
echo ""

while true; do
    # Estatísticas do ElasticSearch
    TOTAL_SENTENCES=$(curl -s "http://localhost:9200/cogitura_sentences/_count" 2>/dev/null | grep -o '"count":[0-9]*' | cut -d':' -f2)
    
    if [ -z "$TOTAL_SENTENCES" ]; then
        TOTAL_SENTENCES=0
    fi
    
    # Progresso percentual
    PROGRESS=$(echo "scale=2; ($TOTAL_SENTENCES / 10000) * 100" | bc 2>/dev/null)
    
    # Arquivos de áudio
    AUDIO_FILES=$(ls /workspaces/Cogitura/data/audio/*.mp3 2>/dev/null | wc -l)
    
    # Último batch no log
    LAST_BATCH=$(tail -20 /tmp/massive_generation.log 2>/dev/null | grep "Batch" | tail -1)
    
    # Exibe status
    echo -ne "\r\033[K" # Limpa linha
    echo "📊 PROGRESSO ATUAL"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Sentenças geradas: $TOTAL_SENTENCES / 10,000"
    echo "📈 Progresso: $PROGRESS%"
    echo "🎵 Arquivos de áudio: $AUDIO_FILES"
    echo "📦 $LAST_BATCH"
    echo ""
    echo "💾 Últimas atividades:"
    tail -5 /tmp/massive_generation.log 2>/dev/null | grep -v "^$"
    echo ""
    echo "🔄 Atualizando a cada 10 segundos... (Ctrl+C para sair)"
    
    # Se chegou em 10k, para
    if [ "$TOTAL_SENTENCES" -ge 10000 ]; then
        echo ""
        echo "🎉 META ATINGIDA! 10.000 sentenças geradas!"
        echo "✅ Pronto para Fase 2: Treinamento"
        break
    fi
    
    sleep 10
    clear
    echo "🤖 COGITURA PTBR-10K - Monitor de Geração"
    echo "=========================================="
    echo ""
done
