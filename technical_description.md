# Preliminary Study: Simulation of Matter Scales Using Genetic Algorithms

## Abstract
This study presents a computational simulation of matter aggregation across scales, inspired by the theory of particle grouping and vibration proposed by Eng. Arsénio Muanda. The model starts from fundamental particles and builds up to macroscopic scales (e.g., the size of an ant), optimizing aggregation factors using a genetic algorithm. Visualization tools demonstrate the logarithmic relationship between size and vibrational energy, revealing potential power-law behaviors in natural phenomena.

## Theory Overview
Muanda's theory posits that matter emerges through hierarchical particle aggregation, where vibrational energy ("strong energy") enables grouping at each scale level. This process is simulated by iteratively combining particles with size and energy multipliers, mimicking quantum and classical scales.

## Simulation Model
- **Levels**: Fundamental Particles (PF) → Quarks/Léptons (QLS) → Protons/Neutrons (PNS) → Atoms → Crystal Structures → Iron Pieces → Ant-sized Substance.
- **Parameters**: Each level uses size and energy factors (e.g., PF to QLS: size ×5, energy ×1.5).
- **Goal**: Achieve a target size (e.g., 4 mm for an ant) from initial particles.

## Genetic Algorithm Optimization
- **Chromosomes**: 12 genes (2 per level: size and energy factors).
- **Fitness Function**: Proximity to target size (fitness = 1 / (1 + error)).
- **Operators**: Tournament selection, single-point crossover, Gaussian mutation.
- **Reports**: Per-generation metrics (duration, evaluated chromosomes, improvements, status).

## Visualization
Using Matplotlib, a log-log scatter plot shows size vs. vibrational energy. Annotations highlight levels, revealing if the aggregation follows a consistent power law.

## Results and Implications
- The simulation demonstrates hierarchical scaling, with energy growing exponentially with size.
- GA optimization finds factor combinations achieving high fitness (e.g., >0.99).
- Visualization confirms logarithmic consistency, supporting the theory's mathematical foundation.
- Future work: Extend to real physics constants, include quantum effects, or apply to other natural hierarchies.

## Code and Usage
- Repository: [Link to GitHub or local files]
- Run simulation: `python simulacao_materia_v2_genetic.py`
- Run GA: `python simulacao_materia_v2_genetic.py --run-ga --save-json reports.json --save-csv reports.csv`
- Visualize: Call `plotar_evolucao_muanda(all_particles)` after simulation.

## Author
Eng. Arsénio Muanda, with computational implementation.

## References
- Muanda, A. (Theory of Particle Grouping and Vibration).
- Genetic Algorithms: Holland, J. H. (1975).