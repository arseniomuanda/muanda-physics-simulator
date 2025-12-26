#!/bin/bash

# Script para criar um novo ciclo de trabalho
# Uso: ./new-cycle.sh [número] [descrição]

if [ -z "$1" ]; then
    echo "❌ Erro: Número do ciclo não fornecido"
    echo ""
    echo "Uso: ./new-cycle.sh NUMERO [DESCRIÇÃO]"
    echo ""
    echo "Exemplo: ./new-cycle.sh 002 \"Machine Learning Integration\""
    exit 1
fi

CYCLE_NUM=$(printf "%03d" "$1")
CYCLE_DESC="${2:-Development cycle}"
BRANCH_NAME="cycle/${CYCLE_NUM}"
CYCLE_FILE="CYCLE_${CYCLE_NUM}.md"

# Navegar para o diretório do projeto
cd "$(dirname "$0")"

# Verificar se estamos em main
CURRENT_BRANCH=$(GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Você não está na branch main. Mudando para main..."
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git checkout main
fi

# Verificar se a branch já existe
if GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git show-ref --verify --quiet refs/heads/"$BRANCH_NAME"; then
    echo "⚠️  Branch $BRANCH_NAME já existe. Mudando para ela..."
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git checkout "$BRANCH_NAME"
    exit 0
fi

# Criar nova branch
echo "🌿 Criando nova branch: $BRANCH_NAME"
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git checkout -b "$BRANCH_NAME"

# Criar arquivo de documentação do ciclo
cat > "$CYCLE_FILE" << EOF
# 🔄 Cycle ${CYCLE_NUM} - ${CYCLE_DESC}

**Branch:** \`${BRANCH_NAME}\`  
**Start Date:** $(date +%Y-%m-%d)  
**Status:** 🟢 Active

## 🎯 Objetivo

${CYCLE_DESC}

## 📋 Atividades Planejadas

- [ ] Definir tarefas específicas
- [ ] Implementar funcionalidades
- [ ] Testar e validar
- [ ] Documentar resultados

## 🔬 Foco Técnico

Este ciclo foca em:
- [A definir]

## 📝 Notas

- [Adicionar notas durante o desenvolvimento]

## 🎯 Resultados Esperados

- [A definir]

## 📊 Progresso

- ✅ Branch criada
- 🔄 Em andamento...

---

**Próximo Ciclo:** \`cycle/$(printf "%03d" $((CYCLE_NUM + 1)))\` (a ser criado após conclusão deste)
EOF

echo "📝 Arquivo de documentação criado: $CYCLE_FILE"
echo ""
echo "✅ Ciclo ${CYCLE_NUM} criado com sucesso!"
echo "📂 Branch: $BRANCH_NAME"
echo "📄 Documentação: $CYCLE_FILE"
echo ""
echo "💡 Dica: Edite $CYCLE_FILE para adicionar objetivos e atividades específicas"
