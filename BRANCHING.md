# 🌿 Convenção de Branches

## Estrutura de Branches

### Branch Principal
- **`main`**: Branch principal com código estável e testado

### Branches de Ciclo de Trabalho
Cada ciclo de atividades de desenvolvimento/pesquisa será uma nova branch seguindo o padrão:

**Formato:** `cycle/XXX` ou `cycle-XXX`

Onde `XXX` é um número sequencial de 3 dígitos (001, 002, 003, ...)

#### Exemplos:
- `cycle/001` - Primeiro ciclo de atividades
- `cycle/002` - Segundo ciclo de atividades
- `cycle/003` - Terceiro ciclo de atividades

### Convenções

1. **Nomenclatura**: Use sempre números com 3 dígitos (001, não 1)
2. **Sequencial**: Cada novo ciclo incrementa o número
3. **Descrição**: Cada ciclo deve ter um objetivo claro
4. **Merge**: Após completar um ciclo, fazer merge para `main` via Pull Request

### Workflow

#### Criar Novo Ciclo (Recomendado)

Use o script `new-cycle.sh` para criar novos ciclos automaticamente:

```bash
# Criar ciclo 002
./new-cycle.sh 002 "Machine Learning Integration"

# Criar ciclo 003 (sem descrição)
./new-cycle.sh 003
```

#### Workflow Manual

```bash
# Criar nova branch de ciclo
git checkout main
git checkout -b cycle/001

# Trabalhar no ciclo
# ... fazer commits ...

# Ao finalizar, fazer merge para main
git checkout main
git merge cycle/001
git push origin main

# Criar próximo ciclo
./new-cycle.sh 002 "Descrição do próximo ciclo"
```

### Documentação de Ciclos

Cada ciclo deve ter:
- Objetivo claro
- Lista de tarefas/atividades
- Resultados esperados
- Notas sobre o que foi aprendido/descoberto

### Histórico de Ciclos

| Ciclo | Branch | Objetivo | Status | Data |
|-------|--------|----------|--------|------|
| 001 | `cycle/001` | [A definir] | 🟢 Ativo | 2024-12-26 |

---

**Nota**: Esta convenção permite rastrear claramente cada fase de desenvolvimento e facilita a organização do trabalho em ciclos iterativos.
