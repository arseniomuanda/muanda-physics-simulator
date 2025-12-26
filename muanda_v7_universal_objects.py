# ==================== MUANDA MODEL v7.0 ====================
# EXTENSÃO: Objetos 3D Universais com Visualização Hierárquica

import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Carregar constantes do v4.0
try:
    with open('muanda_discovered_constants.json', 'r') as f:
        v4_constants = json.load(f)
    print("✓ Constantes do v4.0 carregadas!")
except:
    v4_constants = {}

# Banco de dados de materiais
MATERIALS_DB = {
    'iron': {
        'name': 'Ferro',
        'density': 7874,  # kg/m³
        'atomic_mass': 9.27e-26,  # kg
        'atomic_radius': 1.26e-10,  # m
        'lattice_constant': 2.866e-10,  # m (BCC)
        'crystal_structure': 'BCC',
        'atomic_number': 26,
        'mass_number': 56
    },
    'gold': {
        'name': 'Ouro',
        'density': 19300,  # kg/m³
        'atomic_mass': 3.27e-25,  # kg
        'atomic_radius': 1.35e-10,  # m
        'lattice_constant': 4.078e-10,  # m (FCC)
        'crystal_structure': 'FCC',
        'atomic_number': 79,
        'mass_number': 197
    },
    'carbon': {
        'name': 'Carbono (Diamante)',
        'density': 3510,  # kg/m³
        'atomic_mass': 1.99e-26,  # kg
        'atomic_radius': 7.0e-11,  # m
        'lattice_constant': 3.567e-10,  # m (Diamante)
        'crystal_structure': 'Diamante',
        'atomic_number': 6,
        'mass_number': 12
    },
    'copper': {
        'name': 'Cobre',
        'density': 8960,  # kg/m³
        'atomic_mass': 1.06e-25,  # kg
        'atomic_radius': 1.28e-10,  # m
        'lattice_constant': 3.615e-10,  # m (FCC)
        'crystal_structure': 'FCC',
        'atomic_number': 29,
        'mass_number': 64
    }
}

@dataclass
class Object3D:
    """Representa um objeto 3D com suas propriedades físicas."""
    mass: Optional[float] = None  # kg
    height: Optional[float] = None  # m
    width: Optional[float] = None  # m
    depth: Optional[float] = None  # m
    shape: str = 'cube'  # cube, sphere, cylinder
    diameter: Optional[float] = None  # m (para esfera/cilindro)
    material: str = 'iron'

    def __post_init__(self):
        if self.material not in MATERIALS_DB:
            raise ValueError(f"Material '{self.material}' não suportado. Use: {list(MATERIALS_DB.keys())}")

        self.material_props = MATERIALS_DB[self.material]

        # Calcular propriedades derivadas
        self.calculate_derived_properties()

    def calculate_derived_properties(self):
        """Calcula volume, densidade, número de átomos, etc."""

        # Calcular volume baseado na forma
        if self.shape == 'sphere':
            if self.diameter:
                self.volume = (4/3) * np.pi * (self.diameter/2)**3
                self.height = self.width = self.depth = self.diameter
            elif self.mass:
                self.volume = self.mass / self.material_props['density']
                self.diameter = 2 * (3*self.volume/(4*np.pi))**(1/3)
                self.height = self.width = self.depth = self.diameter

        elif self.shape == 'cube':
            if self.height and self.width and self.depth:
                self.volume = self.height * self.width * self.depth
            elif self.mass:
                self.volume = self.mass / self.material_props['density']
                side = self.volume**(1/3)
                self.height = self.width = self.depth = side

        elif self.shape == 'cylinder':
            # Assumir altura = diâmetro para simplificar
            if self.diameter and self.height:
                radius = self.diameter / 2
                self.volume = np.pi * radius**2 * self.height
                self.width = self.depth = self.diameter
            elif self.mass:
                # Assumir proporção altura = diâmetro
                self.volume = self.mass / self.material_props['density']
                # Volume = π r² h, com h = 2r, então volume = π r² (2r) = 2π r³
                radius = (self.volume / (2 * np.pi))**(1/3)
                self.diameter = 2 * radius
                self.height = self.diameter  # proporção 1:1
                self.width = self.depth = self.diameter

        # Calcular massa se não fornecida
        if not self.mass:
            self.mass = self.volume * self.material_props['density']

        # Número de átomos
        self.num_atoms = self.mass / self.material_props['atomic_mass']

        # Raio equivalente (para esfera de mesmo volume)
        self.equivalent_radius = (3 * self.volume / (4 * np.pi))**(1/3)

        # Densidade calculada vs teórica
        self.calculated_density = self.mass / self.volume
        self.theoretical_density = self.material_props['density']
        self.density_ratio = self.calculated_density / self.theoretical_density

class MuandaObject3D:
    """
    Construtor universal de objetos 3D baseado no modelo Muanda.
    Permite visualizar a formação hierárquica da matéria.
    """

    def __init__(self, obj: Object3D):
        self.obj = obj

        # Constantes fundamentais (do v4.0 ou valores padrão)
        self.constants = {
            'PLANCK_RADIUS': 1.616255e-35,  # m
            'QUARK_SIZE_FACTOR': v4_constants.get('best_genes', [0, 1.41875033802666e14])[1] if v4_constants else 1.41875033802666e14,
            'PROTON_RADIUS': 8.41e-16,  # m
        }

        # Construir hierarquia
        self.build_hierarchy()

    def build_hierarchy(self):
        """Constrói a hierarquia completa de tamanhos."""

        # Níveis hierárquicos
        self.levels = [
            'Planck',
            'Quark',
            'Próton',
            'Núcleo',
            'Átomo',
            'Célula Unitária',
            'Objeto Macro'
        ]

        # Raios para cada nível (fixos, exceto o último)
        planck_r = self.constants['PLANCK_RADIUS']
        quark_r = planck_r * self.constants['QUARK_SIZE_FACTOR']
        proton_r = self.constants['PROTON_RADIUS']

        # Núcleo: aproximado baseado no número atômico
        Z = self.obj.material_props['atomic_number']
        A = self.obj.material_props['mass_number']
        nucleus_r = 1.2e-15 * (A)**(1/3)  # Fórmula aproximada

        atom_r = self.obj.material_props['atomic_radius']
        cell_r = self.obj.material_props['lattice_constant'] / 2  # Raio característico
        object_r = self.obj.equivalent_radius

        self.radii = [
            planck_r,
            quark_r,
            proton_r,
            nucleus_r,
            atom_r,
            cell_r,
            object_r
        ]

        # Calcular fatores de salto
        self.jump_factors = []
        for i in range(1, len(self.radii)):
            factor = self.radii[i] / self.radii[i-1]
            self.jump_factors.append(factor)

        # Calcular número de entidades em cada nível
        self.calculate_quantities()

    def calculate_quantities(self):
        """Calcula quantidades em cada nível hierárquico."""

        # Começando do objeto macro
        num_objects = 1

        # Células unitárias no objeto
        cell_volume = self.obj.material_props['lattice_constant']**3
        atoms_per_cell = 2 if self.obj.material_props['crystal_structure'] == 'BCC' else 4  # BCC=2, FCC=4
        num_cells = self.obj.volume / cell_volume

        # Átomos no objeto
        num_atoms = num_cells * atoms_per_cell

        # Prótons no objeto (aproximado)
        protons_per_atom = self.obj.material_props['atomic_number']
        num_protons = num_atoms * protons_per_atom

        # Quarks (3 por próton)
        num_quarks = num_protons * 3

        # Planck (muito aproximado, não físico)
        # Cada quark tem ~10^20 unidades de Planck ou algo, mas vamos pular

        self.quantities = {
            'Objeto Macro': num_objects,
            'Célula Unitária': int(num_cells),
            'Átomo': int(num_atoms),
            'Próton': int(num_protons),
            'Quark': int(num_quarks),
            'Planck': 'Incontável'  # Não calculável realisticamente
        }

    def print_summary(self):
        """Imprime resumo do objeto e hierarquia."""

        print(f"\n{'='*60}")
        print(f"OBJETO 3D: {self.obj.material_props['name']}")
        print(f"{'='*60}")

        print(f"Forma: {self.obj.shape}")
        print(f"Dimensões: {self.obj.height:.3f} × {self.obj.width:.3f} × {self.obj.depth:.3f} m")
        print(f"Massa: {self.obj.mass:.3f} kg")
        print(f"Volume: {self.obj.volume:.2e} m³")
        print(f"Densidade: {self.obj.calculated_density:.0f} kg/m³ (Teórica: {self.obj.theoretical_density:.0f} kg/m³)")

        print(f"\nNÚMERO DE ENTIDADES:")
        for level, qty in self.quantities.items():
            print(f"  {level:15s}: {qty}")

        print(f"\nHIERARQUIA DE TAMANHOS:")
        for level, radius in zip(self.levels, self.radii):
            print(f"  {level:15s}: {radius:.2e} m")

        print(f"\nFATORES DE SALTO:")
        for i, factor in enumerate(self.jump_factors):
            from_level = self.levels[i]
            to_level = self.levels[i+1]
            print(f"  {from_level} → {to_level}: ×{factor:.2e}")

    def visualize_hierarchy(self):
        """Gera visualização com círculos concêntricos."""

        visualizer = HierarchyVisualizer(self)
        visualizer.plot_concentric_circles()
        visualizer.plot_3d_diagram()
        visualizer.save_results()

class HierarchyVisualizer:
    """Sistema de visualização para a hierarquia Muanda."""

    def __init__(self, muanda_obj):
        self.muanda_obj = muanda_obj
        self.obj = muanda_obj.obj

    def plot_concentric_circles(self):
        """Plota círculos concêntricos em múltiplas escalas."""

        levels = self.muanda_obj.levels
        radii = self.muanda_obj.radii

        # Três gráficos para diferentes escalas
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. Escala Quântica
        quantum_idx = slice(0, 3)  # Planck, Quark, Próton
        self._plot_scale(axes[0], levels[quantum_idx], radii[quantum_idx],
                        "Escala Quântica", quantum_idx)

        # 2. Escala Atômica
        atomic_idx = slice(2, 6)  # Próton, Núcleo, Átomo, Célula
        self._plot_scale(axes[1], levels[atomic_idx], radii[atomic_idx],
                        "Escala Atômica", atomic_idx)

        # 3. Escala Macro
        macro_idx = slice(5, 7)  # Célula, Objeto
        self._plot_scale(axes[2], levels[macro_idx], radii[macro_idx],
                        "Escala Macroscópica", macro_idx)

        plt.suptitle(f"Hierarquia Muanda - {self.obj.material_props['name']} "
                    f"({self.obj.mass:.3f} kg)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f'muanda_v7_{self.obj.material}_{self.obj.shape}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_scale(self, ax, scale_levels, scale_radii, title, idx_slice):
        """Plota uma escala específica."""

        # Usar escala logarítmica
        ax.set_xscale('log')
        ax.set_yscale('log')

        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']

        # Desenhar círculos
        for i, (level, radius) in enumerate(zip(scale_levels, scale_radii)):
            color_idx = (idx_slice.start or 0) + i
            circle = plt.Circle((0, 0), radius, fill=False,
                              edgecolor=colors[color_idx % len(colors)],
                              linewidth=2, label=level)
            ax.add_patch(circle)

        # Configurar eixos
        if scale_radii:
            r_min, r_max = min(scale_radii), max(scale_radii)
            margin = 10
            ax.set_xlim(r_min / margin, r_max * margin)
            ax.set_ylim(r_min / margin, r_max * margin)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, which='both', alpha=0.3)

        # Anotar raios
        for i, (level, radius) in enumerate(zip(scale_levels, scale_radii)):
            angle = 45 + i * 30  # Varia o ângulo
            x_text = radius * np.cos(np.radians(angle))
            y_text = radius * np.sin(np.radians(angle))
            ax.text(x_text, y_text, f'{radius:.1e}', fontsize=7,
                   ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2',
                   facecolor='white', alpha=0.8))

    def plot_3d_diagram(self):
        """Plota diagrama 3D conceitual da hierarquia."""

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Posições dos níveis (em z)
        z_positions = np.linspace(0, 1, len(self.muanda_obj.levels))

        # Raios normalizados para visualização
        radii_norm = np.array(self.muanda_obj.radii)
        radii_norm = (radii_norm - radii_norm.min()) / (radii_norm.max() - radii_norm.min())
        radii_norm = radii_norm * 0.5 + 0.1  # Escalar para 0.1-0.6

        # Desenhar esferas para cada nível
        for i, (level, radius_norm, z) in enumerate(zip(self.muanda_obj.levels, radii_norm, z_positions)):
            # Coordenadas da esfera
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = radius_norm * np.outer(np.cos(u), np.sin(v))
            y = radius_norm * np.outer(np.sin(u), np.sin(v))
            z_sphere = z + radius_norm * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z_sphere, color=plt.cm.viridis(i/len(self.muanda_obj.levels)),
                          alpha=0.7, label=level)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Nível Hierárquico')
        ax.set_title(f'Diagrama 3D da Hierarquia - {self.obj.material_props["name"]}')
        ax.legend()

        plt.savefig(f'muanda_v7_3d_{self.obj.material}_{self.obj.shape}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self):
        """Salva resultados em JSON."""

        results = {
            'object': {
                'material': self.obj.material,
                'shape': self.obj.shape,
                'mass_kg': self.obj.mass,
                'dimensions_m': {
                    'height': self.obj.height,
                    'width': self.obj.width,
                    'depth': self.obj.depth
                },
                'volume_m3': self.obj.volume,
                'density_kg_m3': self.obj.calculated_density,
                'num_atoms': self.obj.num_atoms
            },
            'hierarchy': {
                'levels': self.muanda_obj.levels,
                'radii_m': self.muanda_obj.radii,
                'jump_factors': self.muanda_obj.jump_factors,
                'quantities': self.muanda_obj.quantities
            }
        }

        with open(f'muanda_v7_{self.obj.material}_{self.obj.shape}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

# ==================== EXEMPLOS DE USO ====================

def example_iron_cube():
    """Exemplo: Cubo de ferro de 5cm."""
    obj = Object3D(
        height=0.05,  # 5 cm
        width=0.05,
        depth=0.05,
        material='iron'
    )

    muanda_obj = MuandaObject3D(obj)
    muanda_obj.print_summary()
    muanda_obj.visualize_hierarchy()

def example_gold_sphere():
    """Exemplo: Esfera de ouro de 10cm de diâmetro."""
    obj = Object3D(
        shape='sphere',
        diameter=0.1,  # 10 cm
        material='gold'
    )

    muanda_obj = MuandaObject3D(obj)
    muanda_obj.print_summary()
    muanda_obj.visualize_hierarchy()

def example_carbon_cylinder():
    """Exemplo: Cilindro de carbono (diamante)."""
    obj = Object3D(
        shape='cylinder',
        diameter=0.02,  # 2 cm
        height=0.05,    # 5 cm
        material='carbon'
    )

    muanda_obj = MuandaObject3D(obj)
    muanda_obj.print_summary()
    muanda_obj.visualize_hierarchy()

def example_small_iron_sphere():
    """Exemplo: Pequena esfera de ferro (como uma bolinha)."""
    obj = Object3D(
        shape='sphere',
        diameter=0.01,  # 1 cm
        material='iron'
    )

    muanda_obj = MuandaObject3D(obj)
    muanda_obj.print_summary()
    muanda_obj.visualize_hierarchy()

# ==================== EXECUÇÃO PRINCIPAL ====================

if __name__ == "__main__":
    print("🎯 MUANDA MODEL v7.0 - OBJETOS 3D UNIVERSAIS")
    print("Visualização hierárquica com círculos concêntricos")

    # Executar exemplos
    print("\n1️⃣ EXEMPLO: Cubo de Ferro (5cm)")
    example_iron_cube()

    print("\n2️⃣ EXEMPLO: Esfera de Ouro (10cm)")
    example_gold_sphere()

    print("\n3️⃣ EXEMPLO: Cilindro de Carbono (2x5cm)")
    example_carbon_cylinder()

    print("\n4️⃣ EXEMPLO: Pequena Esfera de Ferro (1cm)")
    example_small_iron_sphere()

    print("\n✅ VISUALIZAÇÕES SALVAS!")
    print("Imagens: muanda_v7_*.png")
    print("Dados: muanda_v7_*_results.json")