#!/bin/bash

# Script para publicar o repositório muanda-physics-simulator no GitHub
# Uso: ./publish.sh [seu_username_github]

REPO_NAME="muanda-physics-simulator"
DESCRIPTION="Advanced physics simulation engine for materials under extreme conditions - thermal expansion, equations of state, plasma physics, and stress testing"

# Verificar se o username foi fornecido
if [ -z "$1" ]; then
    echo "❌ Erro: Username do GitHub não fornecido"
    echo ""
    echo "Uso: ./publish.sh SEU_USERNAME_GITHUB"
    echo ""
    echo "Exemplo: ./publish.sh arseniomuanda"
    echo ""
    echo "Ou crie o repositório manualmente em: https://github.com/new"
    echo "Nome sugerido: $REPO_NAME"
    exit 1
fi

USERNAME="$1"
REPO_URL="https://github.com/${USERNAME}/${REPO_NAME}.git"

echo "🚀 Publicando repositório: $REPO_NAME"
echo "📦 URL: $REPO_URL"
echo ""

# Navegar para o diretório do projeto
cd "$(dirname "$0")"

# Verificar se já existe remote
if GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote get-url origin &>/dev/null; then
    echo "⚠️  Remote 'origin' já existe. Atualizando..."
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote set-url origin "$REPO_URL"
else
    echo "➕ Adicionando remote 'origin'..."
    GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote add origin "$REPO_URL"
fi

# Garantir que estamos na branch main
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git branch -M main

echo ""
echo "📤 Fazendo push para GitHub..."
echo ""

# Fazer push
if GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git push -u origin main; then
    echo ""
    echo "✅ Repositório publicado com sucesso!"
    echo ""
    echo "🌐 Acesse em: https://github.com/${USERNAME}/${REPO_NAME}"
    echo ""
else
    echo ""
    echo "❌ Erro ao fazer push. Possíveis causas:"
    echo "   1. Repositório ainda não foi criado no GitHub"
    echo "   2. Problemas de autenticação"
    echo ""
    echo "💡 Solução:"
    echo "   1. Acesse https://github.com/new"
    echo "   2. Crie o repositório: $REPO_NAME"
    echo "   3. Execute este script novamente"
    echo ""
    exit 1
fi
