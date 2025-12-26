# 🚀 Publicar Repositório no GitHub

## Nome do Repositório Sugerido
**`muanda-physics-simulator`**

## Instruções para Publicar

### Opção 1: Via Interface Web do GitHub (Recomendado)

1. Acesse https://github.com/new
2. Preencha os dados:
   - **Repository name**: `muanda-physics-simulator`
   - **Description**: `Advanced physics simulation engine for materials under extreme conditions - thermal expansion, equations of state, plasma physics, and stress testing`
   - **Visibility**: Public
   - **NÃO** marque "Initialize this repository with a README" (já temos um)
3. Clique em "Create repository"
4. Execute os comandos abaixo no terminal:

```bash
cd /mnt/c/ArsenioMuanda
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote add origin https://github.com/SEU_USUARIO/muanda-physics-simulator.git
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git branch -M main
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git push -u origin main
```

**Substitua `SEU_USUARIO` pelo seu username do GitHub!**

### Opção 2: Via GitHub CLI (se instalado)

```bash
# Instalar GitHub CLI (se necessário)
sudo apt install gh

# Autenticar
gh auth login

# Criar repositório e fazer push
cd /mnt/c/ArsenioMuanda
gh repo create muanda-physics-simulator --public --source=. --remote=origin --description "Advanced physics simulation engine for materials under extreme conditions"
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git push -u origin main
```

### Opção 3: Script Automático

Execute o script abaixo (substitua SEU_USUARIO):

```bash
#!/bin/bash
USERNAME="SEU_USUARIO"  # SUBSTITUA AQUI!
REPO_NAME="muanda-physics-simulator"
REPO_URL="https://github.com/${USERNAME}/${REPO_NAME}.git"

cd /mnt/c/ArsenioMuanda

# Adicionar remote
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote add origin ${REPO_URL} 2>/dev/null || \
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git remote set-url origin ${REPO_URL}

# Push
GIT_DISCOVERY_ACROSS_FILESYSTEM=1 git push -u origin main

echo "✅ Repositório publicado em: ${REPO_URL}"
```

## Após Publicar

O repositório estará disponível em:
**https://github.com/SEU_USUARIO/muanda-physics-simulator**

## Próximos Passos Sugeridos

- [ ] Adicionar arquivo `.gitignore` para Python
- [ ] Adicionar LICENSE (MIT recomendado)
- [ ] Configurar GitHub Actions para testes (opcional)
- [ ] Adicionar badges no README após publicação
- [ ] Criar releases/tags para versões principais
