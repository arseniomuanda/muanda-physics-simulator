"""
VISUALIZADOR GRÁFICO DOS SALTOS HIERÁRQUICOS - MUANDA MODEL v5.0
Gera "fotos" visuais dos saltos mais relevantes na construção da matéria
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def load_construction_results():
    """Carrega os resultados da construção do ferro."""
    try:
        with open('muanda_v5_iron_construction.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Arquivo muanda_v5_iron_construction.json não encontrado!")
        return None

def create_hierarchy_visualization(results):
    """Cria visualização gráfica da hierarquia completa."""

    # Extrair dados da hierarquia
    hierarchy = results['hierarchy']
    levels = [item[0] for item in hierarchy]
    masses = [item[1] for item in hierarchy]
    sizes = [item[2] for item in hierarchy]

    # Criar figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('SALTOS HIERÁRQUICOS NA CONSTRUÇÃO DA MATÉRIA\nMuanda Model v5.0', fontsize=16, fontweight='bold')

    # Plot 1: Hierarquia de Massas
    ax1.plot(range(len(levels)), masses, 'ro-', linewidth=2, markersize=8, markerfacecolor='red')
    ax1.set_yscale('log')
    ax1.set_title('Hierarquia de Massas', fontsize=14)
    ax1.set_ylabel('Massa (kg) - Escala Logarítmica')
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels(levels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Adicionar valores nas bolinhas
    for i, (mass, level) in enumerate(zip(masses, levels)):
        ax1.annotate('2e', xy=(i, mass), xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                    fontsize=8)

    # Plot 2: Hierarquia de Tamanhos
    ax2.plot(range(len(levels)), sizes, 'bo-', linewidth=2, markersize=8, markerfacecolor='blue')
    ax2.set_yscale('log')
    ax2.set_title('Hierarquia de Tamanhos', fontsize=14)
    ax2.set_ylabel('Tamanho (m) - Escala Logarítmica')
    ax2.set_xlabel('Níveis Hierárquicos')
    ax2.set_xticks(range(len(levels)))
    ax2.set_xticklabels(levels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Adicionar valores
    for i, (size, level) in enumerate(zip(sizes, levels)):
        ax2.annotate('2e', xy=(i, size), xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                    fontsize=8)

    plt.tight_layout()
    plt.savefig('muanda_hierarchy_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_saltos_detalhados(results):
    """Cria visualizações detalhadas dos saltos mais relevantes."""

    hierarchy = results['hierarchy']

    # Identificar os maiores saltos
    mass_ratios = []
    size_ratios = []
    for i in range(1, len(hierarchy)):
        mass_ratio = hierarchy[i][1] / hierarchy[i-1][1] if hierarchy[i-1][1] > 0 else 0
        size_ratio = hierarchy[i][2] / hierarchy[i-1][2] if hierarchy[i-1][2] > 0 else 0
        mass_ratios.append(mass_ratio)
        size_ratios.append(size_ratio)

    # Saltos mais relevantes (top 3 por tamanho e massa)
    top_mass_jumps = sorted(enumerate(mass_ratios), key=lambda x: x[1], reverse=True)[:3]
    top_size_jumps = sorted(enumerate(size_ratios), key=lambda x: x[1], reverse=True)[:3]

    # Criar figura com subplots para cada salto relevante
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SALTOS MAIS RELEVANTES - "FOTOS" DOS GRANDES PULOS', fontsize=16, fontweight='bold')

    # Saltos de massa
    for i, (idx, ratio) in enumerate(top_mass_jumps):
        ax = axes[0, i]
        level_from = hierarchy[idx][0]
        level_to = hierarchy[idx+1][0]
        mass_from = hierarchy[idx][1]
        mass_to = hierarchy[idx+1][1]

        # Plot do salto
        ax.bar(['Antes', 'Depois'], [mass_from, mass_to], color=['gray', 'red'], alpha=0.7)
        ax.set_yscale('log')
        ax.set_title('.2e', fontsize=12)
        ax.set_ylabel('Massa (kg)')
        ax.grid(True, alpha=0.3)

        # Adicionar ratio
        ax.text(0.5, 0.95, '.1f', transform=ax.transAxes,
               ha='center', va='top', bbox=dict(boxstyle='round', facecolor='yellow'))

    # Saltos de tamanho
    for i, (idx, ratio) in enumerate(top_size_jumps):
        ax = axes[1, i]
        level_from = hierarchy[idx][0]
        level_to = hierarchy[idx+1][0]
        size_from = hierarchy[idx][2]
        size_to = hierarchy[idx+1][2]

        # Plot do salto
        ax.bar(['Antes', 'Depois'], [size_from, size_to], color=['gray', 'blue'], alpha=0.7)
        ax.set_yscale('log')
        ax.set_title('.2e', fontsize=12)
        ax.set_ylabel('Tamanho (m)')
        ax.grid(True, alpha=0.3)

        # Adicionar ratio
        ax.text(0.5, 0.95, '.1f', transform=ax.transAxes,
               ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

    plt.tight_layout()
    plt.savefig('muanda_saltos_relevantes.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_conceptual_diagram(results):
    """Cria diagrama conceitual dos níveis hierárquicos."""

    hierarchy = results['hierarchy']
    levels = [item[0] for item in hierarchy]

    # Criar diagrama de blocos
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, len(levels))
    ax.set_ylim(0, 1)

    # Desenhar blocos para cada nível
    for i, level in enumerate(levels):
        # Bloco principal
        rect = plt.Rectangle((i, 0.3), 0.8, 0.4, facecolor='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(rect)

        # Texto do nível
        ax.text(i+0.4, 0.5, level, ha='center', va='center', fontsize=10, fontweight='bold')

        # Massa e tamanho abaixo
        mass = hierarchy[i][1]
        size = hierarchy[i][2]
        ax.text(i+0.4, 0.1, '.1e', ha='center', va='center', fontsize=8)
        ax.text(i+0.4, 0.2, '.1e', ha='center', va='center', fontsize=8)

        # Setas entre níveis
        if i < len(levels)-1:
            ax.arrow(i+0.8, 0.5, 0.2-0.8, 0, head_width=0.05, head_length=0.05,
                    fc='red', ec='red', linewidth=2)

    ax.set_title('DIAGRAMA CONCEITUAL DA HIERARQUIA MUANDA\nSaltos de Escala na Construção da Matéria',
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(levels)-0.5)
    ax.set_ylim(0, 1)

    # Legendas
    ax.text(0, 0.9, 'Níveis Hierárquicos', fontsize=12, fontweight='bold')
    ax.text(0, 0.85, 'Cada bloco representa um nível de organização da matéria', fontsize=10)

    plt.savefig('muanda_conceptual_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Função principal para gerar todas as visualizações."""

    print("🔍 Carregando resultados da construção...")
    results = load_construction_results()

    if not results:
        return

    print("📊 Gerando visualizações gráficas dos saltos hierárquicos...")

    # 1. Visualização completa da hierarquia
    print("   1. Hierarquia completa (massa e tamanho)...")
    create_hierarchy_visualization(results)

    # 2. Saltos mais relevantes em detalhe
    print("   2. Saltos mais relevantes (top 3)...")
    create_saltos_detalhados(results)

    # 3. Diagrama conceitual
    print("   3. Diagrama conceitual...")
    create_conceptual_diagram(results)

    print("✅ Visualizações salvas:")
    print("   - muanda_hierarchy_visualization.png")
    print("   - muanda_saltos_relevantes.png")
    print("   - muanda_conceptual_diagram.png")
    print("\n🎯 'Fotos' dos saltos hierárquicos geradas com sucesso!")

if __name__ == "__main__":
    main()