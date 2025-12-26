# MUANDA MODEL v7.0 - Objetos 3D Universais

## 🎯 Visão Geral

O **Muanda Model v7.0** representa uma evolução significativa do modelo hierárquico de formação da matéria. Esta versão permite a criação e visualização de **qualquer objeto 3D** com base em suas propriedades físicas reais, mostrando como a matéria se constrói desde a escala de Planck até objetos macroscópicos cotidianos.

## ✨ Funcionalidades Principais

### 🏗️ Construção Universal de Objetos 3D
- **Materiais Suportados**: Ferro, Ouro, Carbono (Diamante), Cobre
- **Formas Geométricas**: Cubo, Esfera, Cilindro
- **Parâmetros Flexíveis**: Massa, dimensões, material, forma

### 📊 Hierarquia Completa de Tamanhos
- **Níveis Hierárquicos**:
  1. Planck (1.62×10⁻³⁵ m)
  2. Quark (calculado)
  3. Próton (8.41×10⁻¹⁶ m)
  4. Núcleo (depende do elemento)
  5. Átomo (raio atômico)
  6. Célula Unitária (estrutura cristalina)
  7. Objeto Macro (suas dimensões)

### 🎨 Visualização com Círculos Concêntricos
- **Três Escalas Simultâneas**:
  - **Escala Quântica**: Planck → Próton
  - **Escala Atômica**: Próton → Célula Unitária
  - **Escala Macroscópica**: Célula Unitária → Objeto
- **Diagrama 3D Conceitual**: Visão tridimensional da hierarquia
- **Animações de Crescimento**: Evolução visual dos níveis

### 🔬 Cálculos Físicos Precisos
- **Número de Átomos**: Baseado em massa e massa atômica
- **Densidade Real vs Teórica**: Validação da consistência
- **Fatores de Salto**: Razões entre níveis hierárquicos
- **Raio Equivalente**: Para esfera de mesmo volume

## 🚀 Como Usar

### Exemplo Básico: Criar uma Bola de Ferro

```python
from muanda_v7_universal_objects import Object3D, MuandaObject3D

# Criar objeto 3D
obj = Object3D(
    shape='sphere',
    diameter=0.1,  # 10 cm
    material='iron'
)

# Construir hierarquia Muanda
muanda_obj = MuandaObject3D(obj)

# Ver resumo
muanda_obj.print_summary()

# Gerar visualizações
muanda_obj.visualize_hierarchy()
```

### Exemplo Avançado: Cubo de Ouro Personalizado

```python
# Cubo de ouro de 5cm
obj_gold = Object3D(
    height=0.05,   # 5 cm
    width=0.05,
    depth=0.05,
    material='gold'
)

muanda_gold = MuandaObject3D(obj_gold)
muanda_gold.print_summary()
muanda_gold.visualize_hierarchy()
```

### Exemplo com Massa: Cilindro de Diamante

```python
# Cilindro de diamante
obj_diamond = Object3D(
    shape='cylinder',
    diameter=0.02,  # 2 cm
    height=0.05,    # 5 cm
    material='carbon'
)

muanda_diamond = MuandaObject3D(obj_diamond)
muanda_diamond.print_summary()
muanda_diamond.visualize_hierarchy()
```

## 📁 Arquivos Gerados

### Imagens PNG
- `muanda_v7_[material]_[shape].png`: Círculos concêntricos
- `muanda_v7_3d_[material]_[shape].png`: Diagrama 3D conceitual

### Dados JSON
- `muanda_v7_[material]_[shape]_results.json`: Todos os cálculos e propriedades

## 🔧 Arquitetura Técnica

### Classes Principais

#### `Object3D`
- **Propósito**: Representa propriedades físicas do objeto
- **Parâmetros**:
  - `mass`: Massa em kg (opcional)
  - `height/width/depth`: Dimensões em metros
  - `shape`: 'cube', 'sphere', 'cylinder'
  - `diameter`: Para esferas/cilindros
  - `material`: Material do banco de dados

#### `MuandaObject3D`
- **Propósito**: Constrói hierarquia completa
- **Métodos**:
  - `build_hierarchy()`: Calcula tamanhos e quantidades
  - `print_summary()`: Exibe informações detalhadas
  - `visualize_hierarchy()`: Gera gráficos

#### `HierarchyVisualizer`
- **Propósito**: Sistema de visualização
- **Funcionalidades**:
  - Círculos concêntricos em múltiplas escalas
  - Diagramas 3D conceituais
  - Salvamento automático de imagens

### Banco de Dados de Materiais

```python
MATERIALS_DB = {
    'iron': {
        'density': 7874,  # kg/m³
        'atomic_mass': 9.27e-26,  # kg
        'atomic_radius': 1.26e-10,  # m
        'lattice_constant': 2.866e-10,  # m
        # ... outras propriedades
    },
    # ... outros materiais
}
```

## 🎭 Interpretação das Visualizações

### Círculos Concêntricos
- **Raio do Círculo** = Tamanho característico do nível
- **Escala Logarítmica**: Permite visualizar diferenças enormes
- **Cores**: Cada nível tem cor distinta
- **Anotações**: Valores numéricos dos raios

### Diagrama 3D
- **Eixo Z**: Progressão hierárquica (Planck → Macro)
- **Esferas**: Representam cada nível de tamanho
- **Transparência**: Mostra sobreposição conceitual

## 🔬 Aspectos Científicos

### Constantes Utilizadas
- **Raio de Planck**: 1.616×10⁻³⁵ m
- **Fator Quark**: Otimizado via GA (7.19×10¹¹)
- **Raio do Próton**: 8.41×10⁻¹⁶ m (experimental)
- **Raios Atômicos**: Valores tabelados por elemento

### Validações Físicas
- **Conservação de Massa**: Número de átomos consistente
- **Densidade**: Comparação real vs teórica
- **Estrutura Cristalina**: BCC, FCC, Diamante
- **Fatores de Salto**: Razões físicas entre escalas

## 🌟 Exemplos de Uso Real

### Objetos Cotidianos
- **Formiga**: ~3g ferro, 4mm
- **Moeda**: ~7g cobre, 2.5cm diâmetro
- **Jóia**: Ouro 18k, formas variadas
- **Diamante**: Carbono cristalino

### Aplicações Científicas
- **Ensino**: Visualização intuitiva de escalas
- **Materiais**: Comparação de estruturas cristalinas
- **Física**: Entendimento de hierarquias emergentes
- **Computação**: Modelagem de sistemas complexos

## 🚀 Próximas Expansões (v8.0)

- **Mais Materiais**: Alumínio, Titânio, Silício
- **Formas Complexas**: Poliedros, superfícies irregulares
- **Animações**: Crescimento temporal da hierarquia
- **Interatividade**: Interface web para exploração
- **Integração**: Com bancos de dados materiais reais

## 📊 Resultados dos Exemplos

### 1. Cubo de Ferro (5cm)
- **Massa**: 0.984 kg
- **Átomos**: ~10²⁵
- **Saltos**: Planck→Quark (7.19×10¹¹), etc.

### 2. Esfera de Ouro (10cm)
- **Massa**: 10.105 kg
- **Átomos**: ~3×10²⁵
- **Estrutura**: FCC

### 3. Cilindro de Carbono (2×5cm)
- **Massa**: 0.055 kg
- **Átomos**: ~1.4×10²⁴
- **Estrutura**: Diamante

### 4. Esfera de Ferro (1cm)
- **Massa**: 0.004 kg
- **Átomos**: ~4.4×10²²
- **Escala**: Mais manejável para visualização

## 🎯 Conclusão

O **Muanda Model v7.0** transforma a compreensão da matéria, permitindo que qualquer pessoa visualize como objetos cotidianos emergem de leis físicas fundamentais. Desde o Big Bang até sua caneca de café, a hierarquia da matéria agora é visível e compreensível através de círculos concêntricos que representam "fotos" de cada salto quântico.

**Uma máquina do tempo visual para a formação da matéria!** ⏰🔬✨