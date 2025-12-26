# Algoritmo de Simulação de Escala de Matéria
# Baseado na teoria de Agrupamento e Vibração de Partículas
# Autor da Teoria: Eng. Arsénio Muanda
# Desenvolvido em Python por [Seu Nome ou o Nome que Desejar para o Colaborador, se aplicável]

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
    def __init__(self, gene_bounds, fitness_func, pop_size=30, crossover_rate=0.8, mutation_rate=0.1, generations=50):
        """gene_bounds: list of (min, max) for each gene
        fitness_func: function(genes) -> fitness (higher is better)
        """
        self.gene_bounds = gene_bounds
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
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
            self.evaluate()
            best = max(self.population, key=lambda x: x.fitness)
            best_history.append((gen, best.fitness, best.genes[:]))
            if verbose and gen % max(1, self.generations // 10) == 0:
                print(f"GA gen {gen}: best fitness = {best.fitness:.6f}")

        # Return best individual and history
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0], best_history


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
    best, history = ga.run(verbose=True)
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


if __name__ == '__main__':
    print('\nPara executar o exemplo do Algoritmo Genético, chame `run_ga_example()` ou execute o script diretamente com modificação.')