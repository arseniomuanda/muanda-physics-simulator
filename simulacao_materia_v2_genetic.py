# Versão 2 — simulação com Algoritmo Genético integrado
# Arquivo gerado a partir de `simulacao_materia.py` (versão atual)
# Esta versão contém a simulação original + implementação do GA (Chromosome, GeneticAlgorithm)

# IMPORTANTE: Este arquivo é uma cópia da versão atual do projeto.

# Para executar:
# python simulacao_materia_v2_genetic.py

# (Conteúdo copiado do `simulacao_materia.py` atual)

import math
import random
import time
import argparse
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt

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
TAMANHO_FORMIGA_METROS = 4.0 * 10**-3  # 4 milímetros
PARTICULA_FUNDAMENTAL_TAMANHO_BASE = 1.0 * 10**-20 # Um tamanho muito pequeno para a PF
ENERGIA_FUNDAMENTAL_BASE = 1.0 * 10**-25 # Energia de vibração base (em Joules, arbitrário)

# --- Níveis de Agrupamento ---

print("--- Iniciando Simulação de Escalas da Matéria (Teoria Eng. Arsénio Muanda) ---")
print(f"Objetivo: Construir até a escala de uma formiga ({TAMANHO_FORMIGA_METROS:.1e} metros)\n")

# Nível 1: Partículas Fundamentais (PF)
print("1. Criando Partículas Fundamentais (PF)...")
num_pfs_iniciais = 1000
pfs = []
for i in range(num_pfs_iniciais):
    pfs.append(Particula(f"PF_{i+1}", PARTICULA_FUNDAMENTAL_TAMANHO_BASE, ENERGIA_FUNDAMENTAL_BASE * (1 + random.uniform(-0.1, 0.1))))
print(f"  Criadas {len(pfs)} Partículas Fundamentais. Exemplo: {pfs[0]}\n")

print("2. Agrupando PFs em Quarks/Léptons Simplificados (QLS)...")
qls_componentes_por_qls = 3
qls = simular_agrupamento("QLS", qls_componentes_por_qls,
                           pfs[0].tamanho_relativo,
                           pfs[0].energia_vibracional,
                           fator_tamanho=5,
                           fator_energia=1.5,
                           componentes_base=pfs)
print(f"  Criados {len(qls)} QLS. Exemplo: {qls[0]}\n")

print("3. Agrupando QLS em Prótons/Nêutrons Simplificados (PNS)...")
pns_componentes_por_pns = 3
pns = simular_agrupamento("PNS", pns_componentes_por_pns,
                           qls[0].tamanho_relativo,
                           qls[0].energia_vibracional,
                           fator_tamanho=10,
                           fator_energia=2.0,
                           componentes_base=qls)
print(f"  Criados {len(pns)} PNS. Exemplo: {pns[0]}\n")

print("4. Agrupando PNS em Átomos Simplificados (AS - Ferro)...")
as_componentes_por_as = 26
atomos = simular_agrupamento("Atomos_Ferro", as_componentes_por_as,
                             pns[0].tamanho_relativo,
                             pns[0].energia_vibracional,
                             fator_tamanho=1000,
                             fator_energia=10.0,
                             componentes_base=pns)
print(f"  Criados {len(atomos)} Átomos de Ferro. Exemplo: {atomos[0]}\n")

print("5. Agrupando Átomos em Estrutura Cristalina Simplificada (ECS)...")
ecs_componentes_por_ecs = 100
estruturas_cristalinas = simular_agrupamento("Estrutura_Cristalina", ecs_componentes_por_ecs,
                                              atomos[0].tamanho_relativo,
                                              atomos[0].energia_vibracional,
                                              fator_tamanho=100,
                                              fator_energia=5.0,
                                              componentes_base=atomos)
print(f"  Criadas {len(estruturas_cristalinas)} Estruturas Cristalinas. Exemplo: {estruturas_cristalinas[0]}\n")

print("6. Agrupando ECS em Pedaços de Ferro (PFerro)...")
pferro_componentes_por_pferro = 1000
pedacos_ferro = simular_agrupamento("Pedaco_Ferro", pferro_componentes_por_pferro,
                                      estruturas_cristalinas[0].tamanho_relativo,
                                      estruturas_cristalinas[0].energia_vibracional,
                                      fator_tamanho=1000,
                                      fator_energia=2.0,
                                      componentes_base=estruturas_cristalinas)
print(f"  Criados {len(pedacos_ferro)} Pedaços de Ferro. Exemplo: {pedacos_ferro[0]}\n")

print("7. Agrupando Pedaços de Ferro para atingir o tamanho de uma Formiga...")
substancia_tamanho_formiga = []
ultimo_tamanho_acumulado = 0.0
contador_stf = 0

while ultimo_tamanho_acumulado < TAMANHO_FORMIGA_METROS and pedacos_ferro:
    componentes_para_formiga = min(len(pedacos_ferro), 1000)
    if componentes_para_formiga == 0:
        break

    novas_unidades_formiga = simular_agrupamento(f"Formiga_Part_{contador_stf+1}",
                                                  componentes_para_formiga,
                                                  pedacos_ferro[0].tamanho_relativo,
                                                  pedacos_ferro[0].energia_vibracional,
                                                  fator_tamanho=10,
                                                  fator_energia=1.1,
                                                  componentes_base=pedacos_ferro[:componentes_para_formiga])
    
    del pedacos_ferro[:componentes_para_formiga]

    substancia_tamanho_formiga.extend(novas_unidades_formiga)
    
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


# -----------------------------
# Algoritmo Genético Genérico
# -----------------------------
class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = None

    def copy(self):
        c = Chromosome(self.genes[:])
        c.fitness = self.fitness
        return c


class GeneticAlgorithm:
    def __init__(self, gene_bounds, fitness_func, pop_size=30, crossover_rate=0.8, mutation_rate=0.1, generations=50, success_threshold=0.9):
        """gene_bounds: list of (min, max) for each gene
        fitness_func: function(genes) -> fitness (higher is better)
        """
        self.gene_bounds = gene_bounds
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.success_threshold = success_threshold
        self.reports = []
        self.population = []

    def _random_gene(self, bound):
        return random.uniform(bound[0], bound[1])

    def initialize(self):
        self.population = []
        for _ in range(self.pop_size):
            genes = [self._random_gene(b) for b in self.gene_bounds]
            self.population.append(Chromosome(genes))

    def evaluate(self):
        for ind in self.population:
            if ind.fitness is None:
                ind.fitness = self.fitness_func(ind.genes)

    def tournament_selection(self, k=3):
        best = None
        for _ in range(k):
            cand = random.choice(self.population)
            if best is None or cand.fitness > best.fitness:
                best = cand
        return best.copy()

    def crossover(self, a, b):
        # single-point crossover for real-valued genes
        if random.random() > self.crossover_rate:
            return a.copy(), b.copy()
        point = random.randrange(1, len(a.genes))
        child1_genes = a.genes[:point] + b.genes[point:]
        child2_genes = b.genes[:point] + a.genes[point:]
        return Chromosome(child1_genes), Chromosome(child2_genes)

    def mutate(self, ind):
        for i in range(len(ind.genes)):
            if random.random() < self.mutation_rate:
                lo, hi = self.gene_bounds[i]
                # small gaussian perturbation with clamping
                span = (hi - lo)
                perturb = random.gauss(0, 0.1 * span)
                newv = ind.genes[i] + perturb
                # if out of bounds, re-randomize
                if newv < lo or newv > hi:
                    newv = random.uniform(lo, hi)
                ind.genes[i] = newv

    def run(self, verbose=False):
        self.initialize()
        self.evaluate()
        best_history = []
        # previous best for computing improvement
        prev_best = max(self.population, key=lambda x: x.fitness).fitness if self.population else 0.0
        for gen in range(self.generations):
            newpop = []
            # Elitism: keep best
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite = self.population[0].copy()
            newpop.append(elite)

            while len(newpop) < self.pop_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                child1.fitness = None
                child2.fitness = None
                newpop.extend([child1, child2])

            self.population = newpop[:self.pop_size]
            # Count how many chromosomes need evaluation in this generation
            chromosomes_to_eval = sum(1 for ind in self.population if ind.fitness is None)
            gen_start_ts = datetime.utcnow()
            start_time = time.time()
            self.evaluate()
            duration = time.time() - start_time
            gen_end_ts = datetime.utcnow()

            best = max(self.population, key=lambda x: x.fitness)
            improvement_abs = best.fitness - prev_best
            improvement_rel = (improvement_abs / (abs(prev_best) + 1e-12)) if prev_best != 0 else float('inf')
            progresso = float(gen + 1) / float(self.generations) * 100.0

            # Build human-readable description and status
            genes_snapshot = best.genes[:] if len(best.genes) <= 8 else best.genes[:8]
            description = {
                'best_fitness': round(best.fitness, 9),
                'improvement_abs': improvement_abs,
                'improvement_rel': improvement_rel,
                'genes_snapshot': [round(g, 6) for g in genes_snapshot]
            }

            sucesso = bool(best.fitness >= self.success_threshold)
            # Status: 'success' if reached threshold, otherwise 'running' or 'completed' at final gen
            status = 'running'
            if sucesso:
                status = 'success'
            elif gen == self.generations - 1:
                status = 'completed'

            report = {
                'geracao_nome': f"Gen_{gen+1}",
                'inicio_utc': gen_start_ts.isoformat() + 'Z',
                'fim_utc': gen_end_ts.isoformat() + 'Z',
                'duracao_s': duration,
                'cromossomas_avaliados': chromosomes_to_eval,
                'descricao': description,
                'status': status,
                'sucesso': sucesso,
                'insucesso': not sucesso,
                'improvement_abs': improvement_abs,
                'improvement_rel': improvement_rel,
                'avanco_pct': progresso
            }
            self.reports.append(report)

            best_history.append((gen, best.fitness, best.genes[:]))
            prev_best = max(prev_best, best.fitness)

            if verbose:
                print(f"GA gen {gen+1}/{self.generations}: best fitness = {best.fitness:.6f}, duration={duration:.3f}s, evaluated={chromosomes_to_eval}, status={status}")

            # Early stop if success
            if sucesso:
                if verbose:
                    print(f"Success threshold reached at generation {gen+1} (fitness={best.fitness:.6f}). Stopping early.")
                break

        # Return best individual, history and per-generation reports
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0], best_history, self.reports



# -----------------------------
# Função de fitness conectada à simulação
# -----------------------------
def simulate_with_factors(genes, target_size=TAMANHO_FORMIGA_METROS, verbose=False):
    """
    genes: lista de 12 genes correspondendo aos fatores de tamanho e energia
    para os níveis: QLS, PNS, ATOMOS, ECS, PFerro, Formiga (2 genes por nível)
    Retorna fitness baseado na proximidade do tamanho conseguido ao target_size.
    """
    # Mapear genes
    if len(genes) < 12:
        raise ValueError("Esperado 12 genes (2 por nível para 6 níveis).")

    it = iter(genes)
    nivel_nomes = ["QLS", "PNS", "ATOMO", "ECS", "PFerro", "Formiga"]
    fatores = {}
    for nome in nivel_nomes:
        fatores[nome] = {
            'fator_tamanho': next(it),
            'fator_energia': next(it)
        }

    # Simulação reduzida e rápida (números menores para performance na avaliação)
    num_pfs = 200
    pfs_local = [Particula(f"PF_{i+1}", PARTICULA_FUNDAMENTAL_TAMANHO_BASE, ENERGIA_FUNDAMENTAL_BASE * (1 + random.uniform(-0.05, 0.05))) for i in range(num_pfs)]

    # Nível 2: QLS (3 PFs)
    qls = simular_agrupamento("QLS", 3, pfs_local[0].tamanho_relativo, pfs_local[0].energia_vibracional,
                              fatores['QLS']['fator_tamanho'], fatores['QLS']['fator_energia'], componentes_base=pfs_local)

    if not qls:
        return 0.0

    # Nível 3: PNS (3 QLS)
    pns = simular_agrupamento("PNS", 3, qls[0].tamanho_relativo, qls[0].energia_vibracional,
                              fatores['PNS']['fator_tamanho'], fatores['PNS']['fator_energia'], componentes_base=qls)

    if not pns:
        return 0.0

    # Nível 4: Átomos (26 PNS)
    atomos_local = simular_agrupamento("Atomos", 26, pns[0].tamanho_relativo, pns[0].energia_vibracional,
                                       fatores['ATOMO']['fator_tamanho'], fatores['ATOMO']['fator_energia'], componentes_base=pns)

    if not atomos_local:
        return 0.0

    # Nível 5: ECS (100 átomos)
    ecs_local = simular_agrupamento("ECS", 10, atomos_local[0].tamanho_relativo, atomos_local[0].energia_vibracional,
                                    fatores['ECS']['fator_tamanho'], fatores['ECS']['fator_energia'], componentes_base=atomos_local)

    if not ecs_local:
        return 0.0

    # Nível 6: Pedaços de Ferro (agrupar algumas ECS)
    pferro_local = simular_agrupamento("PFerro", 10, ecs_local[0].tamanho_relativo, ecs_local[0].energia_vibracional,
                                       fatores['PFerro']['fator_tamanho'], fatores['PFerro']['fator_energia'], componentes_base=ecs_local)

    if not pferro_local:
        return 0.0

    # Nível Formiga: Agrupar pedaços de ferro até tentar alcançar target
    substancia = []
    ultimo_tamanho = 0.0
    iter_count = 0
    # Limitar iterações para manter avaliação rápida
    while ultimo_tamanho < target_size and pferro_local and iter_count < 5:
        comps = min(len(pferro_local), 50)
        novas = simular_agrupamento("Formiga_Part", comps, pferro_local[0].tamanho_relativo, pferro_local[0].energia_vibracional,
                                    fatores['Formiga']['fator_tamanho'], fatores['Formiga']['fator_energia'], componentes_base=pferro_local[:comps])
        del pferro_local[:comps]
        substancia.extend(novas)
        ultimo_tamanho = sum(s.tamanho_relativo for s in substancia)
        iter_count += 1

    # Fitness: quanto mais perto do target_size, melhor. Normalizar para (0,1]
    error = abs(target_size - ultimo_tamanho)
    fitness = 1.0 / (1.0 + error)
    if verbose:
        print(f"Simulação curta: tamanho={ultimo_tamanho:.2e}, error={error:.2e}, fitness={fitness:.6f}")
    return fitness


def run_ga_example():
    # Definir bounds para os 12 genes (2 por nível)
    # fator_tamanho (min, max), fator_energia (min, max)
    gene_bounds = []
    # Para cada nível: QLS, PNS, ATOMO, ECS, PFerro, Formiga
    for _ in range(6):
        gene_bounds.append((1.1, 100.0))   # fator_tamanho
        gene_bounds.append((0.5, 20.0))    # fator_energia

    ga = GeneticAlgorithm(gene_bounds, simulate_with_factors, pop_size=30, crossover_rate=0.8, mutation_rate=0.2, generations=25)
    best, history, reports = ga.run(verbose=True)
    print("\n--- Resultado do Algoritmo Genético ---")
    print(f"Melhor fitness: {best.fitness:.6f}")
    print(f"Melhores genes: {[round(g,4) for g in best.genes]}")
    # Mostrar simulação final com os melhores genes
    final_size = 0.0
    try:
        # Re-executar com verbose para obter tamanho
        simulate_with_factors(best.genes, verbose=True)
    except Exception:
        pass
    # Imprimir sumário dos relatórios por geração
    print('\nRelatório por geração (resumo):')
    for r in reports:
        print(f"{r['geracao_nome']}: duracao={r['duracao_s']:.3f}s, avaliadas={r['cromossomas_avaliados']}, sucesso={r['sucesso']}, avanco={r['avanco_pct']:.1f}%")


def save_reports_json(reports, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def save_reports_csv(reports, path):
    # Flatten reports into CSV rows
    fieldnames = ['geracao_nome', 'inicio_utc', 'fim_utc', 'duracao_s', 'cromossomas_avaliados',
                  'status', 'sucesso', 'insucesso', 'improvement_abs', 'improvement_rel', 'avanco_pct',
                  'best_fitness', 'genes_snapshot']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            desc = r.get('descricao', {})
            row = {
                'geracao_nome': r.get('geracao_nome'),
                'inicio_utc': r.get('inicio_utc'),
                'fim_utc': r.get('fim_utc'),
                'duracao_s': r.get('duracao_s'),
                'cromossomas_avaliados': r.get('cromossomas_avaliados'),
                'status': r.get('status'),
                'sucesso': r.get('sucesso'),
                'insucesso': r.get('insucesso'),
                'improvement_abs': r.get('improvement_abs'),
                'improvement_rel': r.get('improvement_rel'),
                'avanco_pct': r.get('avanco_pct'),
                'best_fitness': desc.get('best_fitness') if isinstance(desc, dict) else None,
                'genes_snapshot': ';'.join(str(x) for x in desc.get('genes_snapshot', [])) if isinstance(desc, dict) else ''
            }
            writer.writerow(row)


def run_ga_cli(pop_size=30, generations=25, crossover=0.8, mutation=0.2, success_threshold=0.9, save_json=None, save_csv=None, verbose=True):
    gene_bounds = []
    for _ in range(6):
        gene_bounds.append((1.1, 100.0))
        gene_bounds.append((0.5, 20.0))

    ga = GeneticAlgorithm(gene_bounds, simulate_with_factors, pop_size=pop_size, crossover_rate=crossover, mutation_rate=mutation, generations=generations, success_threshold=success_threshold)
    best, history, reports = ga.run(verbose=verbose)

    print("\n--- Resultado do Algoritmo Genético ---")
    print(f"Melhor fitness: {best.fitness:.6f}")
    print(f"Melhores genes: {[round(g,4) for g in best.genes]}")

    if save_json:
        save_reports_json(reports, save_json)
        print(f"Relatórios salvos em JSON: {save_json}")
    if save_csv:
        save_reports_csv(reports, save_csv)
        print(f"Relatórios salvos em CSV: {save_csv}")


import matplotlib.pyplot as plt

def plotar_evolucao_muanda(resultados_simulacao):
    """
    Gera um gráfico comparando o Tamanho vs Energia Vibracional.
    resultados_simulacao: Lista de objetos Particula de diferentes níveis.
    """
    tamanhos = [p.tamanho_relativo for p in resultados_simulacao]
    energias = [p.energia_vibracional for p in resultados_simulacao]
    nomes = [p.nome.split('_')[0] for p in resultados_simulacao]

    plt.figure(figsize=(12, 7))
    
    # Criando o gráfico com escala logarítmica em ambos os eixos
    plt.scatter(tamanhos, energias, alpha=0.6, edgecolors='w', s=100, c=energias, cmap='viridis')
    
    # Adicionando anotações para cada nível
    for i, nome in enumerate(nomes):
        if i % (len(nomes)//5 or 1) == 0: # Evita sobreposição de nomes
            plt.annotate(nome, (tamanhos[i], energias[i]), xytext=(5,5), textcoords='offset points', fontsize=9)

    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Evolução da Matéria: Escala vs Energia (Teoria Eng. Arsénio Muanda)', fontsize=14)
    plt.xlabel('Tamanho Relativo (metros) - Escala Log', fontsize=12)
    plt.ylabel('Energia Vibracional (Joules) - Escala Log', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.show()

# Exemplo de como chamar após a simulação:
# todas_particulas = pfs + qls + pns + atomos + estruturas_cristalinas + pedacos_ferro + substancia_tamanho_formiga
# plotar_evolucao_muanda(todas_particulas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runner para o Algoritmo Genético (simulação).')
    parser.add_argument('--run-ga', action='store_true', help='Executar o GA (default: não).')
    parser.add_argument('--pop', type=int, default=30, help='Tamanho da população')
    parser.add_argument('--gens', type=int, default=25, help='Número de gerações')
    parser.add_argument('--crossover', type=float, default=0.8, help='Taxa de crossover')
    parser.add_argument('--mutation', type=float, default=0.2, help='Taxa de mutação')
    parser.add_argument('--success-threshold', type=float, default=0.9, help='Threshold de sucesso (fitness)')
    parser.add_argument('--save-json', type=str, default=None, help='Caminho para salvar relatórios em JSON')
    parser.add_argument('--save-csv', type=str, default=None, help='Caminho para salvar relatórios em CSV')
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    args = parser.parse_args()
    if args.run_ga:
        run_ga_cli(pop_size=args.pop, generations=args.gens, crossover=args.crossover, mutation=args.mutation, success_threshold=args.success_threshold, save_json=args.save_json, save_csv=args.save_csv, verbose=args.verbose)
    else:
        print('\nPara executar o GA com esta versão, use:')
        print('  python simulacao_materia_v2_genetic.py --run-ga --pop 30 --gens 25 --save-json ga_reports.json --save-csv ga_reports.csv --verbose')
