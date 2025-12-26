# Protocolo de interação — guia rápido

## Propósito
Definir como os próximos prompts devem ser formulados e interpretados para alinhar o agente ao nosso objetivo atual: acelerar aprendizagem, manter reproducibilidade e obter respostas executáveis e verificáveis.

## Objetivos do estudo
- Produzir instruções claras e reproduzíveis.
- Gerar saídas em formatos previsíveis (Markdown, código, checklist).
- Facilitar avaliação automática e humana.

## Regras gerais
- Use português claro e direto.
- Declare o objetivo da interação na primeira frase.
- Forneça contexto mínimo necessário (arquivos, estado, restrições).
- Especifique o formato de saída (ex.: "Resposta em Markdown", "arquivo guide.md").
- Limite de comprimento quando necessário (ex.: máximo 200 palavras).
- Inclua critérios de aceitação (o que valida a resposta).

## Estrutura recomendada de um prompt (template)
- Contexto: breve histórico e estado atual.
- Tarefa: ação específica a executar.
- Entrada: dados, caminhos de arquivo, exemplos.
- Saída esperada: formato e conteúdo mínimo.
- Restrições: prazos, tamanho, estilo, segurança.
- Critérios de aceitação: como validar resultado.

## Exemplo de prompt
Contexto: arquivo vazio em /c:/ArsenioMuanda/guide.md.  
Tarefa: criar um guia de uso de prompts para este projeto.  
Entrada: linguagem PT-BR, formato Markdown.  
Saída esperada: seção "Objetivo", "Regras", "Template" (máx. 300 palavras).  
Restrições: tom impessoal, conciso.  
Critérios de aceitação: guia <= 300 palavras, contém as seções pedidas.

## Critérios de avaliação
- Completude: contém todas as seções solicitadas.
- Conformidade de formato: Markdown válido.
- Clareza: instruções aplicáveis sem ambiguidade.
- Tamanho dentro do limite especificado.

## Checklist rápido para cada prompt
- [ ] Objetivo declarado?
- [ ] Contexto suficiente?
- [ ] Formato de saída especificado?
- [ ] Restrições definidas?
- [ ] Critérios de aceitação claros?

## Notas para o agente
- Priorize respostas acionáveis e testáveis.
- Peça esclarecimento se contexto for insuficiente.
- Produza saídas reproduzíveis e com exemplos quando relevante.
- Mantenha tom impessoal e conciso.
