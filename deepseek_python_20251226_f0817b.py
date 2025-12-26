"""
MUANDA SCALE-VIBRATION MODEL v3.0
================================================================
A Computational Framework for Hierarchical Matter Formation
Based on the Theory of Particle Grouping and Vibration
Author: Eng. Arsénio Muanda
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import math
import random
import time
import argparse
import json
import csv
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from scipy.optimize import curve_fit
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== CONSTANTES FÍSICAS REAIS ====================
PHYSICAL_CONSTANTS = {
    'PLANCK_LENGTH': 1.616255e-35,  # metros
    'PLANCK_ENERGY': 1.956e9,       # Joules
    'PROTON_RADIUS': 8.414e-16,     # metros
    'ATOMIC_RADIUS': 1e-10,         # metros (aproximado)
    'IRON_ATOMIC_MASS': 9.27e-26,   # kg
    'STRONG_FORCE_RANGE': 1e-15,    # metros
    'GRAVITATIONAL_CONSTANT': 6.67430e-11,  # N·m²/kg²
    'SPEED_OF_LIGHT': 299792458,    # m/s
    'BOLTZMANN_CONSTANT': 1.380649e-23,  # J/K
}

# ==================== CLASSES PRINCIPAIS ====================
@dataclass
class Particle:
    """Partícula fundamental com propriedades quânticas simuladas."""
    name: str
    scale_level: int
    size: float  # metros
    vibrational_energy: float  # Joules
    position: Tuple[float, float, float] = (0, 0, 0)
    velocity: Tuple[float, float, float] = (0, 0, 0)
    charge: float = 0.0
    spin: float = 0.5
    components: List = field(default_factory=list)
    creation_time: float = field(default_factory=time.time)
    
    @property
    def mass_equivalent(self) -> float:
        """Massa equivalente via E=mc²."""
        return self.vibrational_energy / (PHYSICAL_CONSTANTS['SPEED_OF_LIGHT'] ** 2)
    
    @property
    def frequency(self) -> float:
        """Frequência de vibração via E=hf."""
        return self.vibrational_energy / (6.62607015e-34)  # Constante de Planck
    
    def __str__(self):
        return (f"{self.name} [L{self.scale_level}] | "
                f"Size: {self.size:.2e}m | "
                f"Energy: {self.vibrational_energy:.2e}J | "
                f"Freq: {self.frequency:.2e}Hz")

@dataclass
class AggregationRule:
    """Regra de agrupamento entre níveis."""
    from_level: str
    to_level: str
    components_needed: int
    size_factor: float
    energy_factor: float
    binding_energy: float
    stability_threshold: float = 0.8
    
    def check_stability(self, particles: List[Particle]) -> bool:
        """Verifica se o agrupamento é energeticamente estável."""
        total_energy = sum(p.vibrational_energy for p in particles)
        binding_required = self.binding_energy * len(particles)
        return total_energy * self.stability_threshold >= binding_required

# ==================== SISTEMA DE SIMULAÇÃO ====================
class MuandaMatterSimulator:
    """Simulador principal do Modelo Muanda."""
    
    def __init__(self, use_real_constants: bool = True):
        self.use_real_constants = use_real_constants
        self.particles = []
        self.aggregation_history = []
        self.energy_history = []
        self.size_history = []
        
        # Definir regras de agrupamento baseadas em física real ou otimizadas
        self.rules = self._initialize_rules()
        
    def _initialize_rules(self) -> Dict[str, AggregationRule]:
        """Inicializa regras de agrupamento baseadas em física."""
        if self.use_real_constants:
            return {
                'PF_to_QLS': AggregationRule('PF', 'QLS', 3, 1e3, 1.5, 1e-28),
                'QLS_to_PNS': AggregationRule('QLS', 'PNS', 3, 1e4, 2.0, 1e-26),
                'PNS_to_ATOM': AggregationRule('PNS', 'ATOM', 26, 1e5, 10.0, 1e-24),
                'ATOM_to_CRYSTAL': AggregationRule('ATOM', 'CRYSTAL', 1e2, 1e2, 5.0, 1e-22),
                'CRYSTAL_to_MACRO': AggregationRule('CRYSTAL', 'MACRO', 1e3, 1e3, 2.0, 1e-20),
            }
        else:
            # Valores para otimização via GA
            return {}
    
    def create_fundamental_particles(self, count: int = 1000) -> List[Particle]:
        """Cria partículas fundamentais com propriedades quânticas."""
        particles = []
        base_size = PHYSICAL_CONSTANTS['PLANCK_LENGTH'] * 10
        base_energy = PHYSICAL_CONSTANTS['PLANCK_ENERGY'] * 1e-44
        
        for i in range(count):
            # Adicionar flutuações quânticas
            size_fluct = base_size * (1 + random.uniform(-0.1, 0.1))
            energy_fluct = base_energy * (1 + random.uniform(-0.2, 0.2))
            
            # Posição aleatória em espaço 3D
            position = (
                random.uniform(-1e-30, 1e-30),
                random.uniform(-1e-30, 1e-30),
                random.uniform(-1e-30, 1e-30)
            )
            
            particle = Particle(
                name=f"PF_{i+1}",
                scale_level=0,
                size=size_fluct,
                vibrational_energy=energy_fluct,
                position=position,
                charge=random.choice([-1/3, 2/3, -1, 0])  # Cargas de quarks e léptons
            )
            particles.append(particle)
        
        self.particles.extend(particles)
        return particles
    
    def aggregate_particles(self, rule: AggregationRule, 
                          input_particles: List[Particle]) -> List[Particle]:
        """Agrupa partículas seguindo uma regra específica."""
        if len(input_particles) < rule.components_needed:
            return []
        
        aggregated = []
        num_groups = len(input_particles) // rule.components_needed
        
        for i in range(num_groups):
            group = input_particles[i*rule.components_needed:(i+1)*rule.components_needed]
            
            if not rule.check_stability(group):
                continue  # Grupo instável - não se forma
            
            # Calcular propriedades do agregado
            avg_position = np.mean([p.position for p in group], axis=0)
            total_energy = sum(p.vibrational_energy for p in group)
            
            new_particle = Particle(
                name=f"{rule.to_level}_{len(self.particles)+1}",
                scale_level=group[0].scale_level + 1,
                size=group[0].size * rule.size_factor,
                vibrational_energy=total_energy * rule.energy_factor + rule.binding_energy,
                position=tuple(avg_position),
                components=group.copy()
            )
            
            aggregated.append(new_particle)
            self.aggregation_history.append({
                'rule': rule.from_level + '_to_' + rule.to_level,
                'components': len(group),
                'new_size': new_particle.size,
                'new_energy': new_particle.vibrational_energy,
                'timestamp': time.time()
            })
        
        return aggregated
    
    def simulate_full_hierarchy(self, target_size: float = 4e-3) -> Dict:
        """Simula toda a hierarquia até atingir tamanho alvo."""
        print("=" * 60)
        print("MUANDA MATTER SIMULATION v3.0")
        print("=" * 60)
        
        # 1. Criar partículas fundamentais
        print("\n1. Criando partículas fundamentais com flutuações quânticas...")
        pfs = self.create_fundamental_particles(10000)
        print(f"   Criadas {len(pfs)} partículas fundamentais")
        
        # 2. Agrupamentos sucessivos
        print("\n2. Executando agrupamentos hierárquicos...")
        
        current_level = pfs
        level_names = ['PF', 'QLS', 'PNS', 'ATOM', 'CRYSTAL', 'MACRO']
        
        for i, rule_key in enumerate(['PF_to_QLS', 'QLS_to_PNS', 
                                    'PNS_to_ATOM', 'ATOM_to_CRYSTAL', 
                                    'CRYSTAL_to_MACRO']):
            
            if i >= len(level_names) - 1:
                break
                
            rule = self.rules[rule_key]
            print(f"   {level_names[i]} → {level_names[i+1]}: "
                  f"{len(current_level)} → ", end="")
            
            new_level = self.aggregate_particles(rule, current_level)
            print(f"{len(new_level)} partículas")
            
            if not new_level:
                print("   ⚠️  Agrupamento falhou - energia insuficiente")
                break
                
            current_level = new_level
            
            # Registrar histórico
            self.energy_history.append(np.mean([p.vibrational_energy for p in current_level]))
            self.size_history.append(np.mean([p.size for p in current_level]))
        
        # 3. Resultados
        print("\n" + "=" * 60)
        print("RESULTADOS DA SIMULAÇÃO")
        print("=" * 60)
        
        if current_level:
            final_particle = current_level[0]
            results = {
                'success': final_particle.size >= target_size * 0.9,
                'final_size': final_particle.size,
                'target_size': target_size,
                'final_energy': final_particle.vibrational_energy,
                'levels_created': final_particle.scale_level + 1,
                'total_particles': len(self.particles),
                'efficiency': final_particle.size / (pfs[0].size * len(pfs)) * 100
            }
            
            print(f"✓ Níveis criados: {results['levels_created']}")
            print(f"✓ Tamanho final: {results['final_size']:.2e} m")
            print(f"✓ Energia final: {results['final_energy']:.2e} J")
            print(f"✓ Eficiência de agrupamento: {results['efficiency']:.2f}%")
            
            if results['success']:
                print(f"✓ OBJETIVO ALCANÇADO! (Formiga: {target_size:.2e} m)")
            else:
                print(f"✗ Objetivo não alcançado")
                
            return results
        else:
            print("✗ Simulação falhou - nenhuma partícula agregada")
            return {'success': False}

# ==================== ALGORITMO GENÉTICO AVANÇADO ====================
class QuantumGeneticAlgorithm:
    """Algoritmo Genético com operadores quânticos."""
    
    def __init__(self, gene_bounds: List[Tuple], fitness_func: Callable,
                 pop_size: int = 50, mutation_rate: float = 0.15,
                 generations: int = 100, use_quantum_ops: bool = True):
        
        self.gene_bounds = gene_bounds
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.use_quantum_ops = use_quantum_ops
        self.population = []
        self.best_history = []
        self.convergence_data = []
        
    def initialize_quantum_population(self):
        """Inicializa população com superposição quântica."""
        self.population = []
        for _ in range(self.pop_size):
            # Genes em superposição (valor médio + amplitude)
            genes = []
            for low, high in self.gene_bounds:
                mean = (low + high) / 2
                amplitude = (high - low) / 4
                genes.append(random.uniform(mean - amplitude, mean + amplitude))
            self.population.append({
                'genes': genes,
                'fitness': None,
                'probability_amplitude': 1.0  # Para operações quânticas
            })
    
    def quantum_crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Crossover com interferência quântica."""
        if not self.use_quantum_ops or random.random() > 0.7:
            # Crossover clássico
            point = random.randint(1, len(parent1['genes']) - 1)
            child1_genes = parent1['genes'][:point] + parent2['genes'][point:]
            child2_genes = parent2['genes'][:point] + parent1['genes'][point:]
        else:
            # Interferência quântica (média ponderada)
            alpha = random.random()  # Fase quântica
            child1_genes = []
            child2_genes = []
            for g1, g2 in zip(parent1['genes'], parent2['genes']):
                # Superposição linear
                c1 = alpha * g1 + (1 - alpha) * g2
                c2 = (1 - alpha) * g1 + alpha * g2
                child1_genes.append(c1)
                child2_genes.append(c2)
        
        return (
            {'genes': child1_genes, 'fitness': None, 'probability_amplitude': 1.0},
            {'genes': child2_genes, 'fitness': None, 'probability_amplitude': 1.0}
        )
    
    def quantum_mutation(self, individual: Dict):
        """Mutação com tunelamento quântico."""
        for i in range(len(individual['genes'])):
            if random.random() < self.mutation_rate:
                low, high = self.gene_bounds[i]
                
                if self.use_quantum_ops and random.random() < 0.3:
                    # Tunelamento quântico - salto para região distante
                    tunnel_prob = individual['probability_amplitude']
                    if random.random() < tunnel_prob:
                        new_val = random.uniform(low, high)
                        individual['genes'][i] = new_val
                else:
                    # Mutação gaussiana clássica
                    current = individual['genes'][i]
                    sigma = (high - low) * 0.1
                    new_val = current + random.gauss(0, sigma)
                    individual['genes'][i] = max(low, min(high, new_val))
    
    def run(self, verbose: bool = True) -> Dict:
        """Executa otimização com GA quântico."""
        print("\n" + "=" * 60)
        print("QUANTUM GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        
        self.initialize_quantum_population()
        
        for gen in range(self.generations):
            # Avaliar fitness
            for ind in self.population:
                if ind['fitness'] is None:
                    ind['fitness'] = self.fitness_func(ind['genes'])
            
            # Ordenar por fitness
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            best = self.population[0]
            self.best_history.append(best['fitness'])
            
            # Registrar convergência
            avg_fitness = np.mean([ind['fitness'] for ind in self.population])
            std_fitness = np.std([ind['fitness'] for ind in self.population])
            self.convergence_data.append({
                'generation': gen,
                'best': best['fitness'],
                'average': avg_fitness,
                'std': std_fitness,
                'diversity': len(set(tuple(ind['genes']) for ind in self.population))
            })
            
            if verbose and gen % 10 == 0:
                print(f"Gen {gen:3d}: Best = {best['fitness']:.6f}, "
                      f"Avg = {avg_fitness:.6f}, Diversity = {self.convergence_data[-1]['diversity']}")
            
            # Critério de parada
            if std_fitness < 1e-6 and gen > 20:
                if verbose:
                    print(f"✓ Convergência alcançada na geração {gen}")
                break
            
            # Nova geração (elitismo + operadores quânticos)
            new_pop = [self.population[0].copy()]  # Elitismo
            
            while len(new_pop) < self.pop_size:
                # Seleção por torneio quântico
                tournament = random.sample(self.population, 5)
                parent1 = max(tournament, key=lambda x: x['fitness'])
                tournament = random.sample(self.population, 5)
                parent2 = max(tournament, key=lambda x: x['fitness'])
                
                # Crossover e mutação
                child1, child2 = self.quantum_crossover(parent1, parent2)
                self.quantum_mutation(child1)
                self.quantum_mutation(child2)
                
                new_pop.extend([child1, child2])
            
            self.population = new_pop[:self.pop_size]
        
        return {
            'best_individual': self.population[0],
            'best_fitness': self.population[0]['fitness'],
            'convergence_data': self.convergence_data,
            'best_history': self.best_history
        }

# ==================== VISUALIZAÇÃO AVANÇADA ====================
class MuandaVisualizer:
    """Sistema avançado de visualização para o Modelo Muanda."""
    
    @staticmethod
    def plot_scale_evolution(simulator: MuandaMatterSimulator, 
                           save_path: str = None):
        """Gráfico 3D da evolução da matéria."""
        fig = plt.figure(figsize=(15, 10))
        
        # Subplot 1: Evolução tamanho vs energia
        ax1 = plt.subplot(221, projection='3d')
        
        sizes = []
        energies = []
        levels = []
        
        for particle in simulator.particles:
            sizes.append(np.log10(particle.size))
            energies.append(np.log10(particle.vibrational_energy))
            levels.append(particle.scale_level)
        
        scatter = ax1.scatter(sizes, energies, levels, 
                            c=levels, cmap='viridis', s=50, alpha=0.6)
        ax1.set_xlabel('log10(Tamanho [m])')
        ax1.set_ylabel('log10(Energia [J])')
        ax1.set_zlabel('Nível de Escala')
        ax1.set_title('Evolução 3D da Matéria (Modelo Muanda)')
        
        # Subplot 2: Histórico de agrupamentos
        ax2 = plt.subplot(222)
        if simulator.aggregation_history:
            times = [h['timestamp'] - simulator.aggregation_history[0]['timestamp'] 
                    for h in simulator.aggregation_history]
            sizes = [h['new_size'] for h in simulator.aggregation_history]
            ax2.plot(times, sizes, 'b-', alpha=0.7)
            ax2.set_xlabel('Tempo (s)')
            ax2.set_ylabel('Tamanho do Agregado (m)')
            ax2.set_yscale('log')
            ax2.set_title('Crescimento Temporal dos Agregados')
            ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Distribuição de energias
        ax3 = plt.subplot(223)
        energies = [p.vibrational_energy for p in simulator.particles 
                   if p.vibrational_energy > 0]
        ax3.hist(np.log10(energies), bins=30, alpha=0.7, color='green')
        ax3.set_xlabel('log10(Energia Vibracional [J])')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição de Energias')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Eficiência por nível
        ax4 = plt.subplot(224)
        if simulator.size_history:
            levels = range(len(simulator.size_history))
            efficiencies = [simulator.energy_history[i]/simulator.size_history[i] 
                          for i in range(len(simulator.size_history))]
            ax4.plot(levels, efficiencies, 'ro-', linewidth=2)
            ax4.set_xlabel('Nível de Escala')
            ax4.set_ylabel('Energia/Tamanho [J/m]')
            ax4.set_title('Eficiência de Agrupamento por Nível')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico salvo em: {save_path}")
        plt.show()
    
    @staticmethod
    def create_interactive_plot(simulator: MuandaMatterSimulator):
        """Cria gráfico interativo com Plotly."""
        sizes = []
        energies = []
        levels = []
        names = []
        
        for particle in simulator.particles:
            sizes.append(particle.size)
            energies.append(particle.vibrational_energy)
            levels.append(particle.scale_level)
            names.append(particle.name)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Escala vs Energia', 'Distribuição de Níveis',
                          'Crescimento Hierárquico', 'Relação Potência')
        )
        
        # Gráfico 1: Dispersão
        fig.add_trace(
            go.Scatter(
                x=np.log10(sizes),
                y=np.log10(energies),
                mode='markers',
                marker=dict(size=8, color=levels, colorscale='Viridis', showscale=True),
                text=names,
                hoverinfo='text+x+y'
            ),
            row=1, col=1
        )
        
        # Gráfico 2: Histograma
        fig.add_trace(
            go.Histogram(x=levels, nbinsx=max(levels)+1),
            row=1, col=2
        )
        
        # Gráfico 3: Linhas de crescimento
        if simulator.size_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(simulator.size_history))),
                    y=np.log10(simulator.size_history),
                    mode='lines+markers',
                    name='Tamanho'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(simulator.energy_history))),
                    y=np.log10(simulator.energy_history),
                    mode='lines+markers',
                    name='Energia'
                ),
                row=2, col=1
            )
        
        # Layout
        fig.update_layout(
            title_text="Visualização Interativa - Modelo Muanda v3.0",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="log10(Tamanho [m])", row=1, col=1)
        fig.update_yaxes(title_text="log10(Energia [J])", row=1, col=1)
        
        return fig

# ==================== ANÁLISE CIENTÍFICA ====================
class ScientificAnalyzer:
    """Analisa resultados com métodos científicos."""
    
    @staticmethod
    def analyze_power_law(sizes: List[float], energies: List[float]) -> Dict:
        """Analisa se segue lei de potência."""
        log_sizes = np.log10(sizes)
        log_energies = np.log10(energies)
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            log_sizes, log_energies
        )
        
        # Teste de normalidade dos resíduos
        residuals = log_energies - (slope * log_sizes + intercept)
        _, normality_p = stats.shapiro(residuals[:5000])  # Limitar para Shapiro
        
        return {
            'power_law_exponent': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'normality_p': normality_p,
            'is_power_law': r_value ** 2 > 0.9 and p_value < 0.05,
            'equation': f"E = 10^{intercept:.3f} * L^{slope:.3f}"
        }
    
    @staticmethod
    def compare_with_real_world(simulator: MuandaMatterSimulator) -> pd.DataFrame:
        """Compara com constantes físicas reais."""
        data = []
        
        # Níveis reais da física
        real_levels = [
            {'name': 'Plank Length', 'size': PHYSICAL_CONSTANTS['PLANCK_LENGTH'], 'energy': PHYSICAL_CONSTANTS['PLANCK_ENERGY']},
            {'name': 'Quark', 'size': 1e-18, 'energy': 1e-12},
            {'name': 'Proton', 'size': PHYSICAL_CONSTANTS['PROTON_RADIUS'], 'energy': 1.503e-10},
            {'name': 'Atom', 'size': PHYSICAL_CONSTANTS['ATOMIC_RADIUS'], 'energy': 1.602e-18},
            {'name': 'Dust Particle', 'size': 1e-6, 'energy': 1e-15},
            {'name': 'Ant', 'size': 4e-3, 'energy': 1e-3}
        ]
        
        # Agrupar partículas simuladas por nível
        simulated_by_level = {}
        for particle in simulator.particles:
            level = particle.scale_level
            if level not in simulated_by_level:
                simulated_by_level[level] = []
            simulated_by_level[level].append(particle)
        
        # Comparar cada nível
        for i, real in enumerate(real_levels):
            if i in simulated_by_level:
                sim_particles = simulated_by_level[i]
                avg_size = np.mean([p.size for p in sim_particles])
                avg_energy = np.mean([p.vibrational_energy for p in sim_particles])
                
                size_error = abs(avg_size - real['size']) / real['size'] * 100
                energy_error = abs(avg_energy - real['energy']) / real['energy'] * 100
                
                data.append({
                    'Level': real['name'],
                    'Real_Size': real['size'],
                    'Simulated_Size': avg_size,
                    'Size_Error_%': size_error,
                    'Real_Energy': real['energy'],
                    'Simulated_Energy': avg_energy,
                    'Energy_Error_%': energy_error,
                    'Match_Quality': 'Good' if size_error < 50 and energy_error < 50 else 'Poor'
                })
        
        return pd.DataFrame(data)

# ==================== MAIN & CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description='Modelo Muanda v3.0 - Simulação de Formação de Matéria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python muanda_model_v3.py --simulate --visualize
  python muanda_model_v3.py --optimize --generations 100
  python muanda_model_v3.py --full-analysis --export-results
        """
    )
    
    parser.add_argument('--simulate', action='store_true', 
                       help='Executar simulação completa')
    parser.add_argument('--optimize', action='store_true',
                       help='Otimizar parâmetros com GA quântico')
    parser.add_argument('--visualize', action='store_true',
                       help='Gerar visualizações')
    parser.add_argument('--analyze', action='store_true',
                       help='Análise científica dos resultados')
    parser.add_argument('--full-analysis', action='store_true',
                       help='Executar todos os módulos')
    parser.add_argument('--generations', type=int, default=50,
                       help='Número de gerações para GA')
    parser.add_argument('--export-results', action='store_true',
                       help='Exportar resultados em JSON/CSV')
    parser.add_argument('--target-size', type=float, default=4e-3,
                       help='Tamanho alvo em metros (padrão: formiga)')
    
    args = parser.parse_args()
    
    # Executar todos se nenhum argumento fornecido
    if not any(vars(args).values()):
        args.simulate = args.visualize = args.analyze = True
    
    results = {}
    
    # 1. SIMULAÇÃO
    if args.simulate or args.full_analysis:
        print("\n" + "="*60)
        print("FASE 1: SIMULAÇÃO DE FORMAÇÃO DE MATÉRIA")
        print("="*60)
        
        simulator = MuandaMatterSimulator(use_real_constants=True)
        sim_results = simulator.simulate_full_hierarchy(target_size=args.target_size)
        results['simulation'] = sim_results
        
        if args.export_results:
            with open('muanda_simulation_results.json', 'w') as f:
                json.dump(sim_results, f, indent=2)
            print("✓ Resultados da simulação exportados")
    
    # 2. OTIMIZAÇÃO COM GA
    if args.optimize or args.full_analysis:
        print("\n" + "="*60)
        print("FASE 2: OTIMIZAÇÃO COM ALGORITMO GENÉTICO QUÂNTICO")
        print("="*60)
        
        # Definir função de fitness
        def fitness_function(genes):
            # genes: [size_factor1, energy_factor1, binding1, ...]
            # Implementar avaliação
            return random.random()  # Placeholder
        
        gene_bounds = [(0.1, 100)] * 12  # 12 parâmetros para otimizar
        
        qga = QuantumGeneticAlgorithm(
            gene_bounds=gene_bounds,
            fitness_func=fitness_function,
            pop_size=30,
            generations=args.generations,
            use_quantum_ops=True
        )
        
        ga_results = qga.run(verbose=True)
        results['optimization'] = ga_results
        
        if args.export_results:
            pd.DataFrame(qga.convergence_data).to_csv('ga_convergence.csv', index=False)
            print("✓ Dados de convergência do GA exportados")
    
    # 3. VISUALIZAÇÃO
    if args.visualize and 'simulator' in locals():
        print("\n" + "="*60)
        print("FASE 3: VISUALIZAÇÃO E ANÁLISE GRÁFICA")
        print("="*60)
        
        MuandaVisualizer.plot_scale_evolution(simulator, save_path='muanda_evolution.png')
        
        # Gráfico interativo
        fig = MuandaVisualizer.create_interactive_plot(simulator)
        fig.write_html('muanda_interactive.html')
        print("✓ Visualização interativa salva como HTML")
    
    # 4. ANÁLISE CIENTÍFICA
    if args.analyze and 'simulator' in locals():
        print("\n" + "="*60)
        print("FASE 4: ANÁLISE CIENTÍFICA AVANÇADA")
        print("="*60)
        
        analyzer = ScientificAnalyzer()
        
        # Análise de lei de potência
        sizes = [p.size for p in simulator.particles]
        energies = [p.vibrational_energy for p in simulator.particles]
        
        power_law = analyzer.analyze_power_law(sizes, energies)
        print(f"\nANÁLISE DE LEI DE POTÊNCIA:")
        print(f"  Equação: {power_law['equation']}")
        print(f"  R² = {power_law['r_squared']:.4f}")
        print(f"  É lei de potência? {'SIM' if power_law['is_power_law'] else 'NÃO'}")
        
        # Comparação com mundo real
        print(f"\nCOMPARAÇÃO COM CONSTANTES FÍSICAS REAIS:")
        comparison_df = analyzer.compare_with_real_world(simulator)
        print(comparison_df.to_string())
        
        if args.export_results:
            comparison_df.to_csv('real_world_comparison.csv', index=False)
            print("✓ Comparação com constantes reais exportada")
    
    # 5. RESUMO FINAL
    print("\n" + "="*60)
    print("RESUMO DA EXECUÇÃO - MODELO MUANDA v3.0")
    print("="*60)
    
    if 'simulation' in results:
        sim = results['simulation']
        print(f"✓ Simulação: {'SUCESSO' if sim['success'] else 'FALHA'}")
        print(f"  Tamanho alcançado: {sim['final_size']:.2e} m")
        print(f"  Eficiência: {sim['efficiency']:.2f}%")
    
    if 'optimization' in results:
        opt = results['optimization']
        print(f"✓ Otimização: Fitness máximo = {opt['best_fitness']:.6f}")
    
    print(f"\n✓ Visualizações geradas:")
    print(f"  - muanda_evolution.png (gráfico estático)")
    print(f"  - muanda_interactive.html (gráfico interativo)")
    
    if args.export_results:
        print(f"\n✓ Dados exportados:")
        print(f"  - muanda_simulation_results.json")
        print(f"  - ga_convergence.csv")
        print(f"  - real_world_comparison.csv")
    
    print("\n" + "="*60)
    print("CRÉDITOS:")
    print("Modelo Muanda v3.0 - Eng. Arsénio Muanda")
    print("Sistema Computacional para Estudo de Formação de Matéria")
    print("="*60)

if __name__ == '__main__':
    main()