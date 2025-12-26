# Versão 1 — simulação original (pré-Algoritmo Genético)
# Cópia da versão original de `simulacao_materia.py` antes de integrarmos o GA.

import math
import random

class Particula:
    """
    Classe base para representar uma partícula em qualquer nível de escala.
    Possui um tamanho relativo e uma 'energia vibracional' simulada.
    """
    def __init__(self, nome, tamanho_relativo, energia_vibracional=0.0):
        self.nome = nome
        self.tamanho_relativo = tamanho_relativo
        self.energia_vibracional = energia_vibracional
        self.componentes = [] # Para armazenar partículas de nível inferior

    def adicionar_componente(self, componente):
        self.componentes.append(componente)

    def calcular_energia_total_componentes(self):
        return sum(c.energia_vibracional for c in self.componentes)

    def __str__(self):
        return (f"{self.nome} (Tamanho Rel.: {self.tamanho_relativo:.2e}, "
                f"Energia Vibr.: {self.energia_vibracional:.2e} Joules)")


def simular_agrupamento(nome_nivel, n_componentes, tamanho_base, energia_base, fator_tamanho, fator_energia, componentes_base=None):
    """
    Função genérica para simular o agrupamento de partículas.
    Retorna uma lista de novas partículas de um nível superior.
    """
    novas_particulas = []
    num_criacoes = 0
    if componentes_base:
        # Criar novas partículas a partir dos componentes fornecidos
        num_criacoes = math.ceil(len(componentes_base) / n_componentes)
    else:
        # Se não há componentes base, cria do zero (ex: Partículas Fundamentais)
        num_criacoes = n_componentes # Vamos criar n_componentes de partículas fundamentais no primeiro nível

    for i in range(num_criacoes):
        nova_particula = Particula(f"{nome_nivel}_{i+1}",
                                   tamanho_base * fator_tamanho,
                                   energia_base * fator_energia * (1 + random.uniform(-0.1, 0.1))) # Pequena variação na energia

        if componentes_base:
            # Agrupar os componentes no novo nível
            for j in range(n_componentes):
                if componentes_base: # Garantir que ainda há componentes para adicionar
                    nova_particula.adicionar_componente(componentes_base.pop(0)) # Remove o componente da lista base

            # A energia vibracional pode ser uma função dos componentes
            # Aqui, como sugerido na teoria, a energia forte (representada pela energia_vibracional base)
            # permite a vibração e o agrupamento, e o novo agrupamento tem uma nova energia.
            # Podemos fazer a energia do novo nível ser a média ou soma dos componentes + um fator de ligação
            nova_particula.energia_vibracional = (nova_particula.calcular_energia_total_componentes() * 0.5 + # Uma parte da energia dos componentes
                                                  energia_base * fator_energia) # Mais uma energia de ligação
        
        novas_particulas.append(nova_particula)
    return novas_particulas


# --- Parâmetros da Simulação ---
# Valores arbitrários para demonstrar a escala. Na física real, são muito diferentes.
TAMANHO_FORMIGA_METROS = 4.0 * 10**-3  # 4 milímetros
PARTICULA_FUNDAMENTAL_TAMANHO_BASE = 1.0 * 10**-20 # Um tamanho muito pequeno para a PF
ENERGIA_FUNDAMENTAL_BASE = 1.0 * 10**-25 # Energia de vibração base (em Joules, arbitrário)

# --- Níveis de Agrupamento ---

print("--- Iniciando Simulação de Escalas da Matéria (Teoria Eng. Arsénio Muanda) ---")
print(f"Objetivo: Construir até a escala de uma formiga ({TAMANHO_FORMIGA_METROS:.1e} metros)\n")

# Nível 1: Partículas Fundamentais (PF)
print("1. Criando Partículas Fundamentais (PF)...")
# Vamos simular um número inicial de PFs para começar a construir
num_pfs_iniciais = 1000 # Um número grande para ter material suficiente
pfs = []
for i in range(num_pfs_iniciais):
    # Cada PF tem um tamanho e uma energia vibracional inicial (representando a "energia forte" em sua escala)
    pfs.append(Particula(f"PF_{i+1}", PARTICULA_FUNDAMENTAL_TAMANHO_BASE, ENERGIA_FUNDAMENTAL_BASE * (1 + random.uniform(-0.1, 0.1))))
print(f"  Criadas {len(pfs)} Partículas Fundamentais. Exemplo: {pfs[0]}\n")

# Nível 2: Quarks/Léptons Simplificados (QLS) - 3 PFs formam um QLS
print("2. Agrupando PFs em Quarks/Léptons Simplificados (QLS)...")
qls_componentes_por_qls = 3
qls = simular_agrupamento("QLS", qls_componentes_por_qls,
                           pfs[0].tamanho_relativo,
                           pfs[0].energia_vibracional,
                           fator_tamanho=5, # QLS é 5x maior que PF
                           fator_energia=1.5, # Energia aumenta um pouco
                           componentes_base=pfs)
print(f"  Criados {len(qls)} QLS. Exemplo: {qls[0]}\n")

# Nível 3: Prótons/Nêutrons Simplificados (PNS) - 3 QLS formam um PNS
print("3. Agrupando QLS em Prótons/Nêutrons Simplificados (PNS)...")
pns_componentes_por_pns = 3
pns = simular_agrupamento("PNS", pns_componentes_por_pns,
                           qls[0].tamanho_relativo,
                           qls[0].energia_vibracional,
                           fator_tamanho=10, # PNS é 10x maior que QLS
                           fator_energia=2.0, # Energia aumenta
                           componentes_base=qls)
print(f"  Criados {len(pns)} PNS. Exemplo: {pns[0]}\n")

# Nível 4: Átomos Simplificados (AS - Ex: Ferro) - Vamos usar 26 PNS (para simular Ferro com 26 prótons/nêutrons no núcleo)
print("4. Agrupando PNS em Átomos Simplificados (AS - Ferro)...")
as_componentes_por_as = 26 # Simplificação para o núcleo de Ferro
atomos = simular_agrupamento("Atomos_Ferro", as_componentes_por_as,
                             pns[0].tamanho_relativo,
                             pns[0].energia_vibracional,
                             fator_tamanho=1000, # Átomo é muito maior que o núcleo
                             fator_energia=10.0, # Grande salto de energia de ligação
                             componentes_base=pns)
print(f"  Criados {len(atomos)} Átomos de Ferro. Exemplo: {atomos[0]}\n")

# Nível 5: Estrutura Cristalina de Ferro Simplificada (ECS) - Vários átomos formam uma micro-estrutura
print("5. Agrupando Átomos em Estrutura Cristalina Simplificada (ECS)...")
ecs_componentes_por_ecs = 100 # Exemplo: 100 átomos formam uma célula unitária simplificada
estruturas_cristalinas = simular_agrupamento("Estrutura_Cristalina", ecs_componentes_por_ecs,
                                              atomos[0].tamanho_relativo,
                                              atomos[0].energia_vibracional,
                                              fator_tamanho=100, # ECS é 100x maior que o átomo
                                              fator_energia=5.0,
                                              componentes_base=atomos)
print(f"  Criadas {len(estruturas_cristalinas)} Estruturas Cristalinas. Exemplo: {estruturas_cristalinas[0]}\n")

# Nível 6: Pedaços de Ferro (PFerro) - Agrupamento de Estruturas Cristalinas
print("6. Agrupando ECS em Pedaços de Ferro (PFerro)...")
pferro_componentes_por_pferro = 1000 # Mil estruturas cristalinas formam um pedacinho
pedacos_ferro = simular_agrupamento("Pedaco_Ferro", pferro_componentes_por_pferro,
                                      estruturas_cristalinas[0].tamanho_relativo,
                                      estruturas_cristalinas[0].energia_vibracional,
                                      fator_tamanho=1000, # Pedacinho é 1000x maior que a ECS
                                      fator_energia=2.0,
                                      componentes_base=estruturas_cristalinas)
print(f"  Criados {len(pedacos_ferro)} Pedaços de Ferro. Exemplo: {pedacos_ferro[0]}\n")

# Nível 7: Substância do Tamanho de uma Formiga (STF)
print("7. Agrupando Pedaços de Ferro para atingir o tamanho de uma Formiga...")
substancia_tamanho_formiga = []
ultimo_tamanho_acumulado = 0.0
contador_stf = 0

# Vamos continuar agrupando até o tamanho total ser maior ou igual ao de uma formiga
while ultimo_tamanho_acumulado < TAMANHO_FORMIGA_METROS and pedacos_ferro:
    componentes_para_formiga = min(len(pedacos_ferro), 1000) # Pega até 1000 pedaços por vez
    if componentes_para_formiga == 0:
        break # Sem mais pedaços para adicionar

    novas_unidades_formiga = simular_agrupamento(f"Formiga_Part_{contador_stf+1}",
                                                  componentes_para_formiga,
                                                  pedacos_ferro[0].tamanho_relativo,
                                                  pedacos_ferro[0].energia_vibracional,
                                                  fator_tamanho=10, # Cada agrupamento cresce 10x
                                                  fator_energia=1.1,
                                                  componentes_base=pedacos_ferro[:componentes_para_formiga])
    
    # Remove os componentes usados da lista original
    del pedacos_ferro[:componentes_para_formiga]

    substancia_tamanho_formiga.extend(novas_unidades_formiga)
    
    # Recalcular o tamanho acumulado (simplificação: somar o tamanho de todas as partes da formiga)
    ultimo_tamanho_acumulado = sum(s.tamanho_relativo for s in substancia_tamanho_formiga)
    contador_stf += 1
    
    print(f"  Agrupando... Tamanho atual acumulado: {ultimo_tamanho_acumulado:.2e} m (Meta: {TAMANHO_FORMIGA_METROS:.2e} m)")

print(f"\n--- Simulação Concluída ---")
if ultimo_tamanho_acumulado >= TAMANHO_FORMIGA_METROS:
    print(f"  Sucesso! Atingimos a escala de uma formiga com tamanho total de aproximadamente {ultimo_tamanho_acumulado:.2e} metros.")
else:
    print(f"  Não foi possível atingir o tamanho de uma formiga com os componentes iniciais. Tamanho final: {ultimo_tamanho_acumulado:.2e} metros.")

print(f"  Exemplo de uma parte da substância do tamanho de uma formiga: {substancia_tamanho_formiga[0] if substancia_tamanho_formiga else 'N/A'}")
print("\nEste algoritmo ilustra os princípios de agrupamento de partículas em escalas crescentes,")
print("mantendo a autoria da teoria de base ao Eng. Arsénio Muanda.")
