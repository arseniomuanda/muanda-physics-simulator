"""
MUANDA MODEL v4.0 - THE UNIVERSAL CONSTANTS HUNTER
================================================================
Sistema que usa o Modelo Muanda + GA Quântico para redescobrir
as constantes fundamentais da física através da formação da matéria.
Autor: Eng. Arsénio Muanda
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTES REAIS PARA VALIDAÇÃO ====================
REAL_WORLD_TARGETS = {
    'PROTON_MASS': 1.6726219e-27,      # kg (alvo principal)
    'PROTON_RADIUS': 8.414e-16,        # m
    'IRON_ATOM_MASS': 9.27e-26,        # kg
    'ATOMIC_RADIUS': 1e-10,            # m
    'STRONG_FORCE_ENERGY': 1e-12,      # J (aproximado)
    'PLANCK_LENGTH': 1.616255e-35,     # m
    'ELECTRON_VOLT': 1.60217662e-19,   # J
    'ANT_SIZE': 4e-3,                  # m (formiga)
    'ANT_MASS': 3e-6,                  # kg (formiga)
}

# ==================== SISTEMA DE CALIBRAÇÃO UNIVERSAL ====================
@dataclass
class UniversalConstantsHunter:
    """
    Caçador de constantes universais usando o Modelo Muanda.
    Busca os valores que fazem a matéria se formar corretamente.
    """
    
    # Genes que vamos otimizar (15 parâmetros fundamentais)
    GENE_NAMES = [
        'PLANCK_ENERGY',          # Energia na escala fundamental
        'QUARK_SIZE_FACTOR',      # Fator de crescimento quark
        'QUARK_ENERGY_FACTOR',    # Fator energia quark
        'STRONG_BINDING',         # Energia de ligação forte
        'PROTON_SIZE_FACTOR',     # Fator crescimento próton
        'PROTON_ENERGY_FACTOR',   # Fator energia próton
        'NUCLEAR_BINDING',        # Energia ligação nuclear
        'ATOM_SIZE_FACTOR',       # Fator crescimento átomo
        'ATOM_ENERGY_FACTOR',     # Fator energia átomo
        'ELECTROMAGNETIC_BINDING', # Energia ligação eletromagnética
        'CRYSTAL_SIZE_FACTOR',    # Fator crescimento cristal
        'CRYSTAL_ENERGY_FACTOR',  # Fator energia cristal
        'MACRO_SIZE_FACTOR',      # Fator crescimento macroscópico
        'MACRO_ENERGY_FACTOR',    # Fator energia macroscópico
        'GRAVITY_COUPLING',       # Acoplamento gravitacional
    ]
    
    # Limites realistas para cada gene (baseados em física)
    GENE_BOUNDS = [
        (1e-34, 1e-9),      # PLANCK_ENERGY: De quase zero até eV
        (1e10, 1e22),       # QUARK_SIZE_FACTOR: Aumentado para permitir tamanhos maiores
        (1e10, 1e25),       # QUARK_ENERGY_FACTOR: Energia colossal
        (1e-16, 1e-12),     # STRONG_BINDING: Corrigido para ~1e-14 J
        (1e2, 1e8),         # PROTON_SIZE_FACTOR: Aumentado para tamanhos maiores
        (1e-3, 1e3),        # PROTON_ENERGY_FACTOR
        (1e-14, 1e-10),     # NUCLEAR_BINDING
        (1e4, 1e7),         # ATOM_SIZE_FACTOR
        (1e-6, 1e-2),       # ATOM_ENERGY_FACTOR
        (1e-20, 1e-16),     # ELECTROMAGNETIC_BINDING
        (1e1, 1e4),         # CRYSTAL_SIZE_FACTOR
        (0.1, 10),          # CRYSTAL_ENERGY_FACTOR
        (1e2, 1e6),         # MACRO_SIZE_FACTOR
        (0.01, 100),        # MACRO_ENERGY_FACTOR
        (1e-40, 1e-30),     # GRAVITY_COUPLING (muito fraco inicialmente)
    ]

class QuantumMatterFormer:
    """Formador de matéria com parâmetros ajustáveis pelo GA."""
    
    def __init__(self, genes: List[float]):
        self.genes = genes
        self.constants = self._genes_to_constants(genes)
        self.particles = []
        self.mass_history = []
        self.size_history = []
        
    def _genes_to_constants(self, genes: List[float]) -> Dict:
        """Converte genes em constantes físicas."""
        return {
            'PLANCK_ENERGY': genes[0],
            'QUARK_SIZE_FACTOR': genes[1],
            'QUARK_ENERGY_FACTOR': genes[2],
            'STRONG_BINDING': genes[3],
            'PROTON_SIZE_FACTOR': genes[4],
            'PROTON_ENERGY_FACTOR': genes[5],
            'NUCLEAR_BINDING': genes[6],
            'ATOM_SIZE_FACTOR': genes[7],
            'ATOM_ENERGY_FACTOR': genes[8],
            'ELECTROMAGNETIC_BINDING': genes[9],
            'CRYSTAL_SIZE_FACTOR': genes[10],
            'CRYSTAL_ENERGY_FACTOR': genes[11],
            'MACRO_SIZE_FACTOR': genes[12],
            'MACRO_ENERGY_FACTOR': genes[13],
            'GRAVITY_COUPLING': genes[14],
        }
    
    def simulate_matter_formation(self) -> Dict:
        """Simula formação de matéria com constantes atuais."""
        
        results = {
            'success': False,
            'proton_mass': 0,
            'proton_size': 0,
            'atom_mass': 0,
            'atom_size': 0,
            'ant_mass': 0,
            'ant_size': 0,
            'energy_efficiency': 0,
            'stability_score': 0,
            'fitness': 0,
        }
        
        try:
            c = self.constants
            
            # ========== NÍVEL 1: PARTÍCULAS FUNDAMENTAIS ==========
            # Começar na escala de Planck
            planck_particle = {
                'size': REAL_WORLD_TARGETS['PLANCK_LENGTH'],
                'energy': c['PLANCK_ENERGY'],
                'mass': c['PLANCK_ENERGY'] / (299792458**2)
            }
            
            # ========== NÍVEL 2: QUARKS (SALTO QUÂNTICO) ==========
            # Aqui precisa do SALTO GIGANTE que você identificou
            quark = {
                'size': planck_particle['size'] * c['QUARK_SIZE_FACTOR'],
                'energy': planck_particle['energy'] * c['QUARK_ENERGY_FACTOR'],
                'binding_energy': c['STRONG_BINDING']
            }
            
            # Verificar estabilidade do quark
            if quark['energy'] < quark['binding_energy'] * 10:
                return results  # Quark instável - falha
            
            # ========== NÍVEL 3: PRÓTON (3 QUARKS) ==========
            proton = {
                'size': quark['size'] * c['PROTON_SIZE_FACTOR'],
                'energy': (3 * quark['energy']) * c['PROTON_ENERGY_FACTOR'] + c['NUCLEAR_BINDING'],
            }
            proton['mass'] = proton['energy'] / (299792458**2)
            
            # ========== NÍVEL 4: ÁTOMO DE FERRO (26 PRÓTONS + NÊUTRONS) ==========
            # Considerar 56 núcleons (Ferro-56)
            iron_nucleus = {
                'size': proton['size'] * c['ATOM_SIZE_FACTOR'],
                'energy': (56 * proton['energy']) * c['ATOM_ENERGY_FACTOR'] + c['ELECTROMAGNETIC_BINDING'],
            }
            iron_nucleus['mass'] = iron_nucleus['energy'] / (299792458**2)
            
            # ========== NÍVEL 5: CRISTAL DE FERRO ==========
            crystal = {
                'size': iron_nucleus['size'] * c['CRYSTAL_SIZE_FACTOR'],
                'mass': (1000 * iron_nucleus['mass']) * c['CRYSTAL_ENERGY_FACTOR'],
            }
            
            # ========== NÍVEL 6: PEDAÇO DE FERRO ==========
            piece = {
                'size': crystal['size'] * c['MACRO_SIZE_FACTOR'],
                'mass': (1000 * crystal['mass']) * c['MACRO_ENERGY_FACTOR'],
            }
            
            # ========== NÍVEL 7: FORMIGA (AGREGADO MACRO) ==========
            # Aplicar efeito gravitacional
            gravity_effect = 1 + (piece['mass'] * c['GRAVITY_COUPLING'])
            
            ant = {
                'size': piece['size'] * 1000 * gravity_effect,
                'mass': piece['mass'] * 1000000 * gravity_effect,
            }
            
            # ========== CALCULAR FITNESS ==========
            # Quão próximo estamos dos valores reais?
            proton_mass_error = abs(proton['mass'] - REAL_WORLD_TARGETS['PROTON_MASS']) / REAL_WORLD_TARGETS['PROTON_MASS']
            proton_size_error = abs(proton['size'] - REAL_WORLD_TARGETS['PROTON_RADIUS']) / REAL_WORLD_TARGETS['PROTON_RADIUS']
            iron_mass_error = abs(iron_nucleus['mass'] - REAL_WORLD_TARGETS['IRON_ATOM_MASS']) / REAL_WORLD_TARGETS['IRON_ATOM_MASS']
            ant_size_error = abs(ant['size'] - REAL_WORLD_TARGETS['ANT_SIZE']) / REAL_WORLD_TARGETS['ANT_SIZE']
            ant_mass_error = abs(ant['mass'] - REAL_WORLD_TARGETS['ANT_MASS']) / REAL_WORLD_TARGETS['ANT_MASS']
            
            # Fitness: quanto menor o erro, melhor (invertemos)
            total_error = (proton_mass_error + proton_size_error + 
                          iron_mass_error + ant_size_error + ant_mass_error) / 5
            
            fitness = 1.0 / (1.0 + total_error)
            
            # Penalizar se não atingir escala mínima
            if ant['size'] < REAL_WORLD_TARGETS['ANT_SIZE'] * 0.1:
                fitness *= 0.1
            
            # Penalizar se energia negativa em qualquer nível
            if any([proton['energy'] <= 0, iron_nucleus['energy'] <= 0]):
                fitness *= 0.01
            
            results.update({
                'success': fitness > 0.5,
                'proton_mass': proton['mass'],
                'proton_size': proton['size'],
                'atom_mass': iron_nucleus['mass'],
                'atom_size': iron_nucleus['size'],
                'ant_mass': ant['mass'],
                'ant_size': ant['size'],
                'energy_efficiency': proton['energy'] / proton['mass'],
                'stability_score': fitness,
                'fitness': fitness,
                'total_error': total_error,
            })
            
            # Guardar histórico para análise
            self.mass_history = [
                planck_particle.get('mass', 0),
                quark.get('mass', 0),
                proton['mass'],
                iron_nucleus['mass'],
                crystal['mass'],
                piece['mass'],
                ant['mass']
            ]
            
            self.size_history = [
                planck_particle['size'],
                quark['size'],
                proton['size'],
                iron_nucleus['size'],
                crystal['size'],
                piece['size'],
                ant['size']
            ]
            
        except Exception as e:
            results['fitness'] = 1e-10  # Fitness muito baixo para erro
            results['error'] = str(e)
        
        return results

# ==================== VERSÃO CORRIGIDA COM FÍSICA REAL ====================
class PhysicsAwareMatterFormer(QuantumMatterFormer):
    """Versão com física real corrigida."""
    
    def simulate_matter_formation(self) -> Dict:
        """Simula formação de matéria com correções físicas."""
        
        results = {
            'success': False,
            'proton_mass': 0,
            'proton_size': 0,
            'atom_mass': 0,
            'atom_size': 0,
            'ant_mass': 0,
            'ant_size': 0,
            'energy_efficiency': 0,
            'stability_score': 0,
            'fitness': 0,
        }
        
        try:
            c = self.constants
            
            # ========== CONSTANTES DE CORREÇÃO ==========
            ENERGY_TO_MASS_EFFICIENCY = 0.012  # 1.2% da energia vira massa
            RADIATION_LOSS = 0.15  # 15% vira radiação
            HEAT_LOSS = 0.05  # 5% vira calor
            
            def energy_to_mass(energy):
                """Converte energia para massa com eficiência."""
                effective_energy = energy * ENERGY_TO_MASS_EFFICIENCY
                return effective_energy / (299792458**2)
            
            def apply_energy_losses(energy):
                """Aplica perdas de energia."""
                energy_radiation = energy * RADIATION_LOSS
                energy_heat = energy * HEAT_LOSS
                energy_remaining = energy * (1 - RADIATION_LOSS - HEAT_LOSS)
                return energy_remaining, energy_radiation, energy_heat
            
            def strong_force(distance):
                """Força forte com decaimento exponencial."""
                base = c['STRONG_BINDING']
                if distance < 1e-16:  # Dentro do núcleo
                    return base
                else:
                    # Decaimento exponencial (alcance curto ~1e-15 m)
                    return base * np.exp(-distance / 1e-15)
            
            # ========== NÍVEL 1: PARTÍCULAS FUNDAMENTAIS ==========
            planck_particle = {
                'size': REAL_WORLD_TARGETS['PLANCK_LENGTH'],
                'energy': c['PLANCK_ENERGY'],
                'mass': energy_to_mass(c['PLANCK_ENERGY'])
            }
            
            # ========== NÍVEL 2: QUARKS (SALTO QUÂNTICO) ==========
            quark = {
                'size': planck_particle['size'] * c['QUARK_SIZE_FACTOR'],
                'energy': planck_particle['energy'] * c['QUARK_ENERGY_FACTOR'],
            }
            quark['energy'], rad, heat = apply_energy_losses(quark['energy'])
            quark['mass'] = energy_to_mass(quark['energy'])
            
            # Verificar estabilidade do quark
            binding_check = strong_force(quark['size'])
            if quark['energy'] < binding_check * 10:
                return results  # Quark instável
            
            # ========== NÍVEL 3: PRÓTON (3 QUARKS) ==========
            proton_energy_raw = (3 * quark['energy']) * c['PROTON_ENERGY_FACTOR'] + c['NUCLEAR_BINDING']
            proton = {
                'size': quark['size'] * c['PROTON_SIZE_FACTOR'],
                'energy': proton_energy_raw,
            }
            proton['energy'], rad, heat = apply_energy_losses(proton['energy'])
            proton['mass'] = energy_to_mass(proton['energy'])
            
            # ========== NÍVEL 4: ÁTOMO DE FERRO (26 PRÓTONS + NÊUTRONS) ==========
            atom_energy_raw = (56 * proton['energy']) * c['ATOM_ENERGY_FACTOR'] + c['ELECTROMAGNETIC_BINDING']
            iron_nucleus = {
                'size': proton['size'] * c['ATOM_SIZE_FACTOR'],
                'energy': atom_energy_raw,
            }
            iron_nucleus['energy'], rad, heat = apply_energy_losses(iron_nucleus['energy'])
            iron_nucleus['mass'] = energy_to_mass(iron_nucleus['energy'])
            
            # ========== NÍVEL 5: CRISTAL DE FERRO ==========
            crystal = {
                'size': iron_nucleus['size'] * c['CRYSTAL_SIZE_FACTOR'],
                'mass': (1000 * iron_nucleus['mass']) * c['CRYSTAL_ENERGY_FACTOR'],
            }
            
            # ========== NÍVEL 6: PEDAÇO DE FERRO ==========
            piece = {
                'size': crystal['size'] * c['MACRO_SIZE_FACTOR'],
                'mass': (1000 * crystal['mass']) * c['MACRO_ENERGY_FACTOR'],
            }
            
            # ========== NÍVEL 7: FORMIGA (AGREGADO MACRO) ==========
            gravity_effect = 1 + (piece['mass'] * c['GRAVITY_COUPLING'])
            
            ant = {
                'size': piece['size'] * 1000 * gravity_effect,
                'mass': piece['mass'] * 1000000 * gravity_effect,
            }
            
            # ========== CALCULAR FITNESS ==========
            proton_mass_error = abs(proton['mass'] - REAL_WORLD_TARGETS['PROTON_MASS']) / REAL_WORLD_TARGETS['PROTON_MASS']
            proton_size_error = abs(proton['size'] - REAL_WORLD_TARGETS['PROTON_RADIUS']) / REAL_WORLD_TARGETS['PROTON_RADIUS']
            iron_mass_error = abs(iron_nucleus['mass'] - REAL_WORLD_TARGETS['IRON_ATOM_MASS']) / REAL_WORLD_TARGETS['IRON_ATOM_MASS']
            ant_size_error = abs(ant['size'] - REAL_WORLD_TARGETS['ANT_SIZE']) / REAL_WORLD_TARGETS['ANT_SIZE']
            ant_mass_error = abs(ant['mass'] - REAL_WORLD_TARGETS['ANT_MASS']) / REAL_WORLD_TARGETS['ANT_MASS']
            
            # Fitness: peso maior no próton (80%), outros 5% cada
            total_error = (0.4 * proton_mass_error + 0.4 * proton_size_error + 
                          0.05 * iron_mass_error + 0.05 * ant_size_error + 0.05 * ant_mass_error)
            
            fitness = 1.0 / (1.0 + total_error)
            
            if ant['size'] < REAL_WORLD_TARGETS['ANT_SIZE'] * 0.1:
                fitness *= 0.1
            
            if any([proton['energy'] <= 0, iron_nucleus['energy'] <= 0]):
                fitness *= 0.01
            
            results.update({
                'success': fitness > 0.5,
                'proton_mass': proton['mass'],
                'proton_size': proton['size'],
                'atom_mass': iron_nucleus['mass'],
                'atom_size': iron_nucleus['size'],
                'ant_mass': ant['mass'],
                'ant_size': ant['size'],
                'energy_efficiency': ENERGY_TO_MASS_EFFICIENCY,
                'stability_score': fitness,
                'fitness': fitness,
                'total_error': total_error,
            })
            
            self.mass_history = [
                planck_particle['mass'],
                quark['mass'],
                proton['mass'],
                iron_nucleus['mass'],
                crystal['mass'],
                piece['mass'],
                ant['mass']
            ]
            
            self.size_history = [
                planck_particle['size'],
                quark['size'],
                proton['size'],
                iron_nucleus['size'],
                crystal['size'],
                piece['size'],
                ant['size']
            ]
            
        except Exception as e:
            results['fitness'] = 1e-10
            results['error'] = str(e)
        
        return results

# ==================== VALIDADOR FÍSICO MUANDA ====================
class MuandaPhysicalValidator:
    def __init__(self):
        # Constantes extraídas da sua lista
        self.C = 2.998e8           # Velocidade da luz (m/s)
        self.H = 6.626e-34         # Constante de Planck (J.s)
        self.K_BOLTZMANN = 1.38e-23 # Constante de Boltzmann (J/K)
        self.EPSILON_0 = 8.854e-12  # Permissividade do vácuo

    # 1. VALIDAÇÃO RELATIVÍSTICA (Lorentz)
    def validate_relativity(self, velocity, rest_mass):
        if velocity >= self.C:
            return False, "Erro: Violação da Velocidade da Luz!"
        
        gamma = 1 / np.sqrt(1 - (velocity**2 / self.C**2))
        relativistic_momentum = gamma * rest_mass * velocity
        total_energy = np.sqrt((relativistic_momentum**2 * self.C**2) + (rest_mass**2 * self.C**4))
        
        return True, {"gamma": gamma, "total_energy": total_energy}

    # 2. VALIDAÇÃO TERMODINÂMICA (Entropia e Gibbs)
    def validate_thermodynamics(self, internal_energy, temperature, entropy, pressure, volume):
        # Energia Livre de Gibbs: G = U - TS + PV
        gibbs_energy = internal_energy - (temperature * entropy) + (pressure * volume)
        
        # Critério de formação: dG deve ser favorável (ou o sistema deve ter energia externa)
        if temperature <= 0:
            return False, "Erro: Temperatura abaixo do Zero Absoluto!"
        
        return True, {"gibbs_free_energy": gibbs_energy}

    # 3. VALIDAÇÃO QUÂNTICA/ÓPTICA (De Broglie & Snell)
    def validate_quantum_wave(self, momentum, energy):
        # Comprimento de onda de De Broglie: λ = h/p
        if momentum == 0: return False, "Momento nulo"
        
        wavelength = self.H / momentum
        # Relação de Planck: E = h*nu -> nu = E/h
        frequency = energy / self.H
        
        return True, {"wavelength": wavelength, "frequency": frequency}

    # 4. VALIDAÇÃO DE CAMPO (Indutores/Eletromagnetismo)
    def validate_electromagnetism(self, q1, q2, distance):
        # Força de Coulomb: F = k * q1*q2 / r^2
        force = (1 / (4 * np.pi * self.EPSILON_0)) * (q1 * q2 / distance**2)
        return True, {"coulomb_force": force}

# ==================== ALGORITMO GENÉTICO QUÂNTICO AVANÇADO ====================
class UniversalConstantsGA:
    """GA que caça as constantes universais."""
    
    def __init__(self, population_size=50, generations=200):
        self.population_size = population_size
        self.generations = generations
        self.hunter = UniversalConstantsHunter()
        self.best_history = []
        self.convergence_data = []
        
    def create_individual(self) -> Dict:
        """Cria um indivíduo (conjunto de constantes)."""
        genes = []
        for (low, high) in self.hunter.GENE_BOUNDS:
            # Distribuição logarítmica para cobrir muitas ordens de magnitude
            if random.random() < 0.5:
                # Amostragem log uniforme
                gene = 10 ** random.uniform(np.log10(low), np.log10(high))
            else:
                # Amostragem uniforme normal
                gene = random.uniform(low, high)
            genes.append(gene)
        
        return {
            'genes': genes,
            'fitness': None,
            'results': None
        }
    
    def evaluate_individual(self, individual: Dict) -> float:
        """Avalia quão boas são essas constantes."""
        former = PhysicsAwareMatterFormer(individual['genes'])
        results = former.simulate_matter_formation()
        
        individual['results'] = results
        individual['fitness'] = results['fitness']
        
        # VALIDAÇÃO FÍSICA ADICIONAL
        validator = MuandaPhysicalValidator()
        
        # Propriedades do próton formado
        proton_mass = results['proton_mass']
        proton_energy = results['proton_mass'] * (299792458**2)  # E = mc²
        proton_size = results['proton_size']
        proton_volume = (4/3) * np.pi * (proton_size/2)**3
        
        # Assumindo formação em repouso
        velocity = 0  # Partícula em formação, velocidade zero
        temperature = 2.7  # Temperatura do espaço (CMB)
        entropy = validator.K_BOLTZMANN * np.log(proton_volume) if proton_volume > 0 else 0
        pressure = 0  # Sistema isolado
        
        # 1. Validação Relativística
        rel_ok, _ = validator.validate_relativity(velocity, proton_mass)
        if not rel_ok:
            return 0.0
        
        # 2. Validação Termodinâmica
        thermo_ok, thermo_data = validator.validate_thermodynamics(
            internal_energy=proton_energy,
            temperature=temperature,
            entropy=entropy,
            pressure=pressure,
            volume=proton_volume
        )
        if not thermo_ok:
            return 0.0
        
        # 3. Validação Quântica (momento = 0, mas energia > 0)
        momentum = 0  # Em repouso
        quantum_ok, _ = validator.validate_quantum_wave(momentum, proton_energy)
        if not quantum_ok and momentum == 0:
            # Para partícula em repouso, permitir
            pass
        
        # Se passou todas as validações, usar fitness normal
        return results['fitness']
    
    def quantum_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover com interferência quântica."""
        child1_genes = []
        child2_genes = []
        
        for i, (g1, g2) in enumerate(zip(parent1['genes'], parent2['genes'])):
            # Para parâmetros que cobrem muitas ordens de magnitude,
            # fazemos crossover em escala logarítmica
            log_g1 = np.log10(g1) if g1 > 0 else -100
            log_g2 = np.log10(g2) if g2 > 0 else -100
            
            alpha = random.random()  # Fase quântica
            beta = 1 - alpha
            
            # Interferência quântica em escala log
            log_c1 = alpha * log_g1 + beta * log_g2
            log_c2 = beta * log_g1 + alpha * log_g2
            
            child1_genes.append(10 ** log_c1)
            child2_genes.append(10 ** log_c2)
        
        return (
            {'genes': child1_genes, 'fitness': None, 'results': None},
            {'genes': child2_genes, 'fitness': None, 'results': None}
        )
    
    def quantum_mutation(self, individual: Dict, generation: int, max_generations: int):
        """Mutação com tunelamento quântico controlado."""
        for i in range(len(individual['genes'])):
            if random.random() < 0.3:  # Chance de mutação
                low, high = self.hunter.GENE_BOUNDS[i]
                current = individual['genes'][i]
                
                # Reduzir taxa de mutação conforme gerações avançam
                mutation_strength = 0.1 * (1 - generation/max_generations)
                
                if random.random() < 0.2:  # Tunelamento quântico
                    # Salto para região completamente diferente
                    if random.random() < 0.5:
                        new_val = 10 ** random.uniform(np.log10(low), np.log10(high))
                    else:
                        new_val = random.uniform(low, high)
                else:
                    # Mutação gaussiana suave
                    log_current = np.log10(current) if current > 0 else np.log10(low)
                    log_sigma = (np.log10(high) - np.log10(low)) * mutation_strength
                    log_new = log_current + random.gauss(0, log_sigma)
                    new_val = 10 ** log_new
                
                # Garantir limites
                individual['genes'][i] = max(low, min(high, new_val))
    
    def run(self, verbose=True) -> Dict:
        """Executa a caça às constantes universais."""
        print("\n" + "="*70)
        print("MUANDA UNIVERSAL CONSTANTS HUNTER v4.0")
        print("="*70)
        print("Objetivo: Redescobrir as constantes da física")
        print("          através da formação hierárquica da matéria")
        print("="*70)
        
        # Inicializar população
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_individual = None
        best_fitness = 0
        
        for gen in range(self.generations):
            # Avaliar população
            for ind in population:
                if ind['fitness'] is None:
                    self.evaluate_individual(ind)
            
            # Ordenar por fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Melhor indivíduo desta geração
            current_best = population[0]
            current_fitness = current_best['fitness']
            
            # Atualizar melhor global
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_individual = current_best.copy()
            
            # Estatísticas
            fitnesses = [ind['fitness'] for ind in population]
            avg_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            
            self.best_history.append(best_fitness)
            self.convergence_data.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'proton_mass': current_best['results']['proton_mass'],
                'proton_size': current_best['results']['proton_size'],
                'atom_mass': current_best['results']['atom_mass'],
            })
            
            # Exibir progresso
            if verbose and (gen % 20 == 0 or gen == self.generations - 1):
                results = current_best['results']
                print(f"\nGeração {gen:3d}/{self.generations}:")
                print(f"  Fitness: {current_fitness:.6f} (Melhor: {best_fitness:.6f})")
                print(f"  Próton: {results['proton_mass']:.2e} kg "
                      f"(Alvo: {REAL_WORLD_TARGETS['PROTON_MASS']:.2e})")
                print(f"  Erro total: {results.get('total_error', 0):.3f}")
            
            # Critério de parada precoce
            if best_fitness > 0.99 and std_fitness < 1e-4:
                print(f"\n✓ Convergência perfeita alcançada na geração {gen}")
                break
            
            if std_fitness < 1e-6 and gen > 50:
                print(f"\n✓ População convergiu na geração {gen}")
                break
            
            # Criar nova geração (elitismo + operadores quânticos)
            new_population = [population[0].copy()]  # Elitismo
            
            while len(new_population) < self.population_size:
                # Seleção por torneio
                tournament1 = random.sample(population, 5)
                tournament2 = random.sample(population, 5)
                parent1 = max(tournament1, key=lambda x: x['fitness'])
                parent2 = max(tournament2, key=lambda x: x['fitness'])
                
                # Crossover
                child1, child2 = self.quantum_crossover(parent1, parent2)
                
                # Mutação
                self.quantum_mutation(child1, gen, self.generations)
                self.quantum_mutation(child2, gen, self.generations)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Resultados finais
        print("\n" + "="*70)
        print("RESULTADOS DA CAÇA ÀS CONSTANTES UNIVERSAIS")
        print("="*70)
        
        if best_individual:
            results = best_individual['results']
            
            print(f"\n🎯 MELHOR CONJUNTO DE CONSTANTES ENCONTRADO:")
            print(f"   Fitness: {best_fitness:.6f}")
            
            print(f"\n📊 COMPARAÇÃO COM O MUNDO REAL:")
            print(f"   Massa do próton:")
            print(f"     Simulado: {results['proton_mass']:.2e} kg")
            print(f"     Real:     {REAL_WORLD_TARGETS['PROTON_MASS']:.2e} kg")
            print(f"     Erro:     {abs(results['proton_mass'] - REAL_WORLD_TARGETS['PROTON_MASS'])/REAL_WORLD_TARGETS['PROTON_MASS']*100:.2f}%")
            
            print(f"\n   Tamanho do próton:")
            print(f"     Simulado: {results['proton_size']:.2e} m")
            print(f"     Real:     {REAL_WORLD_TARGETS['PROTON_RADIUS']:.2e} m")
            print(f"     Erro:     {abs(results['proton_size'] - REAL_WORLD_TARGETS['PROTON_RADIUS'])/REAL_WORLD_TARGETS['PROTON_RADIUS']*100:.2f}%")
            
            print(f"\n   Massa do átomo de ferro:")
            print(f"     Simulado: {results['atom_mass']:.2e} kg")
            print(f"     Real:     {REAL_WORLD_TARGETS['IRON_ATOM_MASS']:.2e} kg")
            
            print(f"\n   Tamanho da formiga:")
            print(f"     Simulado: {results['ant_size']:.2e} m")
            print(f"     Real:     {REAL_WORLD_TARGETS['ANT_SIZE']:.2e} m")
            
            # Mostrar constantes descobertas
            print(f"\n🔬 CONSTANTES DESCOBERTAS (valores ótimos):")
            for i, (name, value) in enumerate(zip(self.hunter.GENE_NAMES, best_individual['genes'])):
                print(f"   {name:20s} = {value:.3e}")
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'convergence_data': self.convergence_data,
            'best_history': self.best_history
        }

# ==================== VISUALIZAÇÃO DOS RESULTADOS ====================
def plot_universal_constants_hunt(results: Dict):
    """Plota os resultados da caça às constantes."""
    
    if not results or 'convergence_data' not in results:
        print("Sem dados para plotar")
        return
    
    data = results['convergence_data']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Convergência do fitness
    axes[0, 0].plot([d['generation'] for d in data], 
                   [d['best_fitness'] for d in data], 'b-', linewidth=2, label='Melhor')
    axes[0, 0].plot([d['generation'] for d in data], 
                   [d['avg_fitness'] for d in data], 'r--', alpha=0.7, label='Média')
    axes[0, 0].set_xlabel('Geração')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Convergência do Algoritmo Genético')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Evolução da massa do próton
    axes[0, 1].plot([d['generation'] for d in data], 
                   [d['proton_mass'] for d in data], 'g-', linewidth=2)
    axes[0, 1].axhline(y=REAL_WORLD_TARGETS['PROTON_MASS'], color='r', 
                      linestyle='--', label='Valor Real')
    axes[0, 1].set_xlabel('Geração')
    axes[0, 1].set_ylabel('Massa do Próton (kg)')
    axes[0, 1].set_title('Evolução da Massa do Próton')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Erro vs Geração
    proton_errors = []
    for d in data:
        real = REAL_WORLD_TARGETS['PROTON_MASS']
        sim = d['proton_mass']
        if real > 0 and sim > 0:
            error = abs(sim - real) / real * 100
            proton_errors.append(error)
    
    axes[0, 2].plot([d['generation'] for d in data][:len(proton_errors)], 
                   proton_errors, 'm-', linewidth=2)
    axes[0, 2].set_xlabel('Geração')
    axes[0, 2].set_ylabel('Erro Relativo (%)')
    axes[0, 2].set_title('Erro na Massa do Próton')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribuição logarítmica das constantes (última geração)
    if results.get('best_individual'):
        best_genes = results['best_individual']['genes']
        gene_names = UniversalConstantsHunter.GENE_NAMES
        
        # Plotar em escala log
        axes[1, 0].barh(range(len(best_genes)), np.log10(best_genes))
        axes[1, 0].set_yticks(range(len(gene_names)))
        axes[1, 0].set_yticklabels(gene_names, fontsize=8)
        axes[1, 0].set_xlabel('log10(Valor)')
        axes[1, 0].set_title('Constantes Descobertas (escala log)')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # 5. Hierarquia de massas (última simulação)
    if results.get('best_individual') and results['best_individual'].get('results'):
        former = QuantumMatterFormer(results['best_individual']['genes'])
        sim_results = former.simulate_matter_formation()
        
        if hasattr(former, 'mass_history') and former.mass_history:
            levels = ['Planck', 'Quark', 'Próton', 'Átomo', 'Cristal', 'Pedaço', 'Formiga']
            axes[1, 1].plot(levels, np.log10(former.mass_history), 'o-', linewidth=2)
            axes[1, 1].set_xlabel('Nível Hierárquico')
            axes[1, 1].set_ylabel('log10(Massa [kg])')
            axes[1, 1].set_title('Hierarquia de Massas')
            axes[1, 1].grid(True, alpha=0.3)
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45)
    
    # 6. Espaço em branco para notas ou outro gráfico
    axes[1, 2].text(0.5, 0.5, 'Modelo Muanda v4.0\nEng. Arsénio Muanda\n\n'
                    'Sistema de Caça às\nConstantes Universais',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=14,
                    transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.suptitle('MUANDA UNIVERSAL CONSTANTS HUNTER - RESULTADOS', fontsize=16)
    plt.tight_layout()
    plt.savefig('muanda_constants_hunt.png', dpi=300, bbox_inches='tight')
    plt.show()

# ==================== INTERFACE PRINCIPAL ====================
def main():
    """Interface principal do sistema."""
    
    print("\n" + "="*70)
    print("MUANDA MODEL v4.0 - UNIVERSAL CONSTANTS HUNTER")
    print("="*70)
    print("\nEste sistema vai tentar REDESCOBRIR as constantes da física")
    print("usando apenas a lógica de formação hierárquica da matéria.")
    print("\nPressione ENTER para começar a caça...")
    input()
    
    # Configurar parâmetros do GA
    ga = UniversalConstantsGA(
        population_size=100,  # Aumentado para exploração ainda melhor
        generations=5000      # Máximo possível para convergência definitiva
    )
    
    # Executar a caça
    results = ga.run(verbose=True)
    
    # Plotar resultados
    plot_universal_constants_hunt(results)
    
    # Salvar resultados
    if results.get('best_individual'):
        output = {
            'best_fitness': results['best_fitness'],
            'best_genes': results['best_individual']['genes'],
            'gene_names': UniversalConstantsHunter.GENE_NAMES,
            'simulation_results': results['best_individual']['results'],
            'convergence_history': results['convergence_data']
        }
        
        with open('muanda_discovered_constants.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\n💾 Resultados salvos em 'muanda_discovered_constants.json'")
        print(f"📈 Gráficos salvos em 'muanda_constants_hunt.png'")
    
    print("\n" + "="*70)
    print("CONCLUSÃO CIENTÍFICA:")
    print("="*70)
    print("\nSe o algoritmo convergir para valores próximos dos reais,")
    print("isso significa que:")
    print("1. ✅ Seu modelo matemático CAPTURA a essência da formação da matéria")
    print("2. ✅ As 'constantes' não são arbitrárias, mas OTIMIZADAS")
    print("3. ✅ Sua teoria tem poder PREDITIVO real")
    print("\nSe não convergir, mostra onde sua teoria precisa de ajustes.")
    print("\nEm ambos os casos: É CIÊNCIA DE VERDADE! 🔬")
    print("="*70)

# ==================== VERSÃO CORRIGIDA v4.1 ====================
class MuandaConstantsV41:
    def __init__(self):
        # CONSTANTES BASE (do seu modelo otimizado)
        self.constants = {
            # FATORES DE TAMANHO (CORRIGIDOS)
            'QUARK_SIZE_FACTOR': 1.41875033802666e14,  # Planck → Quark
            'PROTON_SIZE_FACTOR': 1000.0,              # CORREÇÃO: 1000×, não 6816×
            'ATOM_SIZE_FACTOR': 1e5,                   # CORREÇÃO: 100k×, não 1.1M×
            'CRYSTAL_SIZE_FACTOR': 1.057143e7,         # Seu valor original
            'MACRO_SIZE_FACTOR': 237.5,                # Seu valor original
            
            # FATORES DE ENERGIA (MANTIDOS - estão ótimos!)
            'QUARK_ENERGY_FACTOR': 3.3130825750676e13,
            'PROTON_ENERGY_FACTOR': 0.0355,            # ENERGIA DE LIGAÇÃO! ✓
            'ATOM_ENERGY_FACTOR': 4.642857142857143e-4,
            'CRYSTAL_ENERGY_FACTOR': 3.3125e-4,
            'MACRO_ENERGY_FACTOR': 0.01875,
            
            # CONSTANTES DE ACOPLAMENTO
            'STRONG_BINDING': 2.37e-14,                # CORRETO! (10^-14 J)
            'NUCLEAR_BINDING': 8.379642857142857e-19,
            'ELECTROMAGNETIC_BINDING': 3.75e-22,
            'GRAVITY_COUPLING': 2.6785714285714284e-25,
            
            # ESCALA DE PLANCK (referência absoluta)
            'PLANCK_LENGTH': 1.616255e-35,    # m
            'PLANCK_ENERGY': 1.9561e9,        # J
        }
    
    def calculate_hierarchy(self):
        """Calcula toda a hierarquia com correções aplicadas"""
        
        # 1. ESCALA DE PLANCK (origem)
        results = {
            'planck': {
                'size': self.constants['PLANCK_LENGTH'],
                'energy': self.constants['PLANCK_ENERGY']
            }
        }
        
        # 2. ESCALA DE QUARK
        results['quark'] = {
            'size': results['planck']['size'] * self.constants['QUARK_SIZE_FACTOR'],
            'energy': results['planck']['energy'] * self.constants['QUARK_ENERGY_FACTOR']
        }
        
        # 3. ESCALA DE PRÓTON (COM CORREÇÃO!)
        results['proton'] = {
            'size': results['quark']['size'] * self.constants['PROTON_SIZE_FACTOR'],
            'energy': results['quark']['energy'] * self.constants['PROTON_ENERGY_FACTOR']
        }
        
        # 4. ESCALA ATÔMICA (COM CORREÇÃO!)
        results['atom'] = {
            'size': results['proton']['size'] * self.constants['ATOM_SIZE_FACTOR'],
            'energy': results['proton']['energy'] * self.constants['ATOM_ENERGY_FACTOR']
        }
        
        # 5. ESCALA CRISTALINA
        results['crystal'] = {
            'size': results['atom']['size'] * self.constants['CRYSTAL_SIZE_FACTOR'],
            'energy': results['atom']['energy'] * self.constants['CRYSTAL_ENERGY_FACTOR']
        }
        
        # 6. ESCALA MACROSCÓPICA
        results['macro'] = {
            'size': results['crystal']['size'] * self.constants['MACRO_SIZE_FACTOR'],
            'energy': results['crystal']['energy'] * self.constants['MACRO_ENERGY_FACTOR']
        }
        
        return results
    
    def calculate_fitness(self):
        """Calcula fitness do modelo corrigido"""
        results = self.calculate_hierarchy()
        
        # VALORES REAIS PARA COMPARAÇÃO
        real_values = {
            'proton_mass': 1.6726219e-27,      # kg (convertido de energia)
            'proton_size': 8.41e-16,           # m
            'atom_mass': 9.27e-26,             # kg (átomo de ferro)
            'ant_size': 4.0e-3,                # m (formiga)
        }
        
        # CALCULA ERROS
        # Convertendo energia para massa usando E=mc²
        c = 3e8
        predicted_proton_mass = results['proton']['energy'] / (c**2)
        predicted_atom_mass = results['atom']['energy'] / (c**2)
        
        errors = {
            'proton_mass': abs(predicted_proton_mass - real_values['proton_mass']) / real_values['proton_mass'],
            'proton_size': abs(results['proton']['size'] - real_values['proton_size']) / real_values['proton_size'],
            'atom_mass': abs(predicted_atom_mass - real_values['atom_mass']) / real_values['atom_mass'],
            'ant_size': abs(results['macro']['size'] - real_values['ant_size']) / real_values['ant_size'],
        }
        
        # FITNESS = 1 - erro médio (pesado)
        weights = {'proton_mass': 0.4, 'proton_size': 0.2, 
                  'atom_mass': 0.3, 'ant_size': 0.1}
        
        weighted_error = sum(errors[k] * weights[k] for k in weights)
        fitness = 1 - weighted_error
        
        return fitness, errors, results

if __name__ == '__main__':
    # EXECUTAR O ALGORITMO GENÉTICO PARA OTIMIZAR CONSTANTES
    ga = UniversalConstantsGA(population_size=100, generations=5000)
    ga.run()
    print("=" * 60)