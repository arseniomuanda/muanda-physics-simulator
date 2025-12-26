# Simulação de Matéria — Versão GA (v2)

Resumo
- Esta versão contém a simulação original e um Algoritmo Genético para otimizar fatores de agrupamento por nível.
- Nova funcionalidade: Visualização com matplotlib para plotar evolução da matéria (tamanho vs energia em escala log).

Uso rápido
- Executar a simulação original (sem GA):

```powershell
python simulacao_materia_v2_genetic.py
```

- Executar o GA (exemplo; salva relatórios):

```powershell
python simulacao_materia_v2_genetic.py --run-ga --pop 30 --gens 25 --save-json ga_reports.json --save-csv ga_reports.csv --verbose
```

- Visualizar evolução (após simulação):

```python
# No REPL ou no final do script:
todas_particulas = pfs + qls + pns + atomos + estruturas_cristalinas + pedacos_ferro + substancia_tamanho_formiga
plotar_evolucao_muanda(todas_particulas)
```

O que é salvo
- `ga_reports.json`: relatório detalhado por geração (timestamps, duração, melhoria, snapshot de genes, status).
- `ga_reports.csv`: versão tabular do relatório para análise rápida.

Campos-chave por geração
- `geracao_nome`: nome legível da geração
- `inicio_utc`, `fim_utc`: timestamps UTC
- `duracao_s`: duração da avaliação da geração em segundos
- `cromossomas_avaliados`: número de cromossomas avaliados nessa geração
- `status`: `running`, `success` ou `completed`
- `sucesso`: boolean indicando se a geração atingiu o threshold
- `improvement_abs`, `improvement_rel`: melhoria absoluta e relativa do fitness
- `genes_snapshot`: primeiros genes do indivíduo líder

Próximos passos sugeridos
- Ajustar bounds dos genes ou incluir novos genes (ex.: número de componentes por nível).
- Aumentar a população / gerações para otimização mais robusta.
- Integrar salvamento automático dos melhores indivíduos (checkpoints).

Arquivos relacionados
- `technical_description.md`: Descrição técnica em inglês para apresentação/publicação.

Autor: Eng. Arsénio Muanda — adaptado para GA
