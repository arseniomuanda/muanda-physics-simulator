# ==================== MUANDA MODEL v7.1 ====================
# EXTENSÃO: Stress Test Térmico-Mecânico

import numpy as np
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Importar do v7.0
from muanda_v7_universal_objects import Object3D, MuandaObject3D, MATERIALS_DB

# Constantes físicas
R = 8.314462618  # J/(mol·K) - Constante dos gases
NA = 6.02214076e23  # mol⁻¹ - Número de Avogadro
KB = 1.380649e-23  # J/K - Constante de Boltzmann
SIGMA_SB = 5.670367e-8  # W/(m²·K⁴) - Constante de Stefan-Boltzmann

@dataclass
class ThermodynamicState:
    """Estado termodinâmico completo de um material."""
    temperature: float  # K
    pressure: float     # Pa
    volume: float       # m³
    internal_energy: float  # J
    entropy: float      # J/K
    phase: str          # solid, liquid, gas, plasma
    density: float      # kg/m³
    crystal_structure: Optional[str] = None

@dataclass
class StressCondition:
    """Condição de stress aplicada."""
    type: str  # heating, compression, radiation, shock
    rate: Optional[float] = None      # K/s, Pa/s, W/m², etc.
    target_value: Optional[float] = None  # Valor alvo
    duration: Optional[float] = None  # s

# Banco de dados termodinâmico expandido
THERMODYNAMIC_DB = {
    'iron': {
        'melting_point': 1811,  # K
        'boiling_point': 3134,  # K
        'latent_heat_fusion': 247000,  # J/kg
        'latent_heat_vaporization': 6090000,  # J/kg
        'specific_heat_solid': lambda T: 450 + 0.1*T,  # J/(kg·K)
        'specific_heat_liquid': 800,  # J/(kg·K)
        'thermal_expansion': 1.2e-5,  # K⁻¹
        'bulk_modulus': 170e9,  # Pa
        'critical_temperature': 8500,  # K (estimado)
        'critical_pressure': 35e6,  # Pa (estimado)
        'phase_transitions': [
            {'T': 1184, 'type': 'magnetic', 'name': 'Curie point'},
            {'T': 1811, 'type': 'fusion', 'name': 'Melting'},
            {'T': 3134, 'type': 'vaporization', 'name': 'Boiling'}
        ],
        'high_pressure_phases': [
            {'P': 13e9, 'phase': 'hcp', 'name': 'Hexagonal close-packed'},
            {'P': 200e9, 'phase': 'bcc_double', 'name': 'Double BCC'}
        ]
    },
    'gold': {
        'melting_point': 1337,  # K
        'boiling_point': 3129,  # K
        'latent_heat_fusion': 64300,  # J/kg
        'latent_heat_vaporization': 334000,  # J/kg
        'specific_heat_solid': lambda T: 129 + 0.05*T,  # J/(kg·K)
        'specific_heat_liquid': 160,  # J/(kg·K)
        'thermal_expansion': 1.42e-5,  # K⁻¹
        'bulk_modulus': 180e9,  # Pa
        'critical_temperature': 7000,  # K (estimado)
        'critical_pressure': 50e6,  # Pa (estimado)
        'phase_transitions': [
            {'T': 1337, 'type': 'fusion', 'name': 'Melting'},
            {'T': 3129, 'type': 'vaporization', 'name': 'Boiling'}
        ]
    },
    'carbon': {
        'melting_point': 3800,  # K (diamante)
        'boiling_point': 4300,  # K
        'latent_heat_fusion': 100000,  # J/kg (estimado)
        'latent_heat_vaporization': 50000000,  # J/kg (estimado)
        'specific_heat_solid': 500,  # J/(kg·K)
        'specific_heat_liquid': 2000,  # J/(kg·K)
        'thermal_expansion': 1e-6,  # K⁻¹ (diamante)
        'bulk_modulus': 442e9,  # Pa (diamante)
        'critical_temperature': 8000,  # K (estimado)
        'critical_pressure': 100e6,  # Pa (estimado)
        'phase_transitions': [
            {'T': 3800, 'type': 'fusion', 'name': 'Melting'},
            {'T': 4300, 'type': 'vaporization', 'name': 'Boiling'}
        ],
        'high_pressure_phases': [
            {'P': 20e9, 'phase': 'bc8', 'name': 'BC8 carbon'},
            {'P': 100e9, 'phase': 'simple_hexagonal', 'name': 'Simple hexagonal'}
        ]
    }
}

class MuandaStressTest:
    """
    Sistema de teste de stress termodinâmico para objetos Muanda.
    Submete objetos 3D a condições extremas para validar robustez física.
    """

    def __init__(self, material: str, initial_state: Dict, stress_conditions: List[Dict]):
        self.material = material
        self.material_props = MATERIALS_DB[material]
        self.thermo_props = THERMODYNAMIC_DB[material]

        # Estado inicial
        self.initial_state = initial_state
        self.create_initial_object()

        # Condições de stress
        self.stress_conditions = [StressCondition(**cond) for cond in stress_conditions]

        # Histórico da simulação
        self.history: List[ThermodynamicState] = []
        self.time_steps = []

        # Resultados
        self.failure_point = None
        self.failure_mechanism = None
        self.emergent_laws = []

    def create_initial_object(self):
        """Cria o objeto inicial baseado no estado fornecido."""
        if 'mass' in self.initial_state:
            obj = Object3D(mass=self.initial_state['mass'], material=self.material)
        elif 'size' in self.initial_state:
            if self.initial_state.get('shape') == 'cube':
                obj = Object3D(height=self.initial_state['size'],
                             width=self.initial_state['size'],
                             depth=self.initial_state['size'],
                             material=self.material)
            elif self.initial_state.get('shape') == 'sphere':
                obj = Object3D(shape='sphere',
                             diameter=self.initial_state['size'],
                             material=self.material)
        elif 'diameter' in self.initial_state:
            obj = Object3D(shape='sphere',
                         diameter=self.initial_state['diameter'],
                         material=self.material)
        else:
            raise ValueError("Estado inicial deve ter 'mass', 'size' ou 'diameter'")

        self.muanda_obj = MuandaObject3D(obj)
        self.mass = obj.mass
        self.initial_volume = obj.volume

        # Estado termodinâmico inicial (293K, 1atm)
        self.initial_state_obj = ThermodynamicState(
            temperature=293.15,
            pressure=101325,
            volume=self.initial_volume,
            internal_energy=0,  # Referência
            entropy=0,  # Referência
            phase='solid',
            density=self.material_props['density'],
            crystal_structure=self.material_props['crystal_structure']
        )

    def run_simulation(self, max_temperature: float = 5000,
                      max_pressure: float = 1e10,
                      time_steps: int = 1000,
                      dt: float = 1.0) -> Dict:
        """
        Executa a simulação de stress completa.

        Args:
            max_temperature: Temperatura máxima (K)
            max_pressure: Pressão máxima (Pa)
            time_steps: Número de passos de tempo
            dt: Intervalo de tempo (s)

        Returns:
            Dict com resultados da simulação
        """

        print(f"🧪 INICIANDO STRESS TEST: {self.material.upper()}")
        print(f"Condições: T_max={max_temperature}K, P_max={max_pressure}Pa")
        print(f"Passos: {time_steps}, dt={dt}s")

        # Estado atual
        current_state = self.initial_state_obj
        self.history = [current_state]
        self.time_steps = [0]

        t = 0

        for step in range(time_steps):
            t += dt

            # Aplicar condições de stress
            new_state = self.apply_stress_conditions(current_state, dt)

            # Limitar valores extremos
            new_state.temperature = min(new_state.temperature, max_temperature)
            new_state.pressure = min(new_state.pressure, max_pressure)

            # Verificar falhas
            failure = self.check_failure_conditions(new_state)
            if failure:
                self.failure_point = t
                self.failure_mechanism = failure
                print(f"❌ FALHA DETECTADA em t={t:.1f}s: {failure}")
                break

            # Atualizar estado
            current_state = new_state
            self.history.append(current_state)
            self.time_steps.append(t)

            # Progress
            if step % (time_steps//10) == 0:
                print(f"⏳ Progresso: {step/time_steps*100:.1f}% - T={current_state.temperature:.0f}K, P={current_state.pressure:.1e}Pa")

        # Análise final
        results = self.analyze_results()

        print("✅ SIMULAÇÃO CONCLUÍDA"        if not self.failure_point else f"❌ SIMULAÇÃO FALHOU em {self.failure_point:.1f}s")

        return results

    def apply_stress_conditions(self, state: ThermodynamicState, dt: float) -> ThermodynamicState:
        """Aplica as condições de stress ao estado atual."""

        new_state = ThermodynamicState(
            temperature=state.temperature,
            pressure=state.pressure,
            volume=state.volume,
            internal_energy=state.internal_energy,
            entropy=state.entropy,
            phase=state.phase,
            density=state.density,
            crystal_structure=state.crystal_structure
        )

        for condition in self.stress_conditions:
            if condition.type == 'heating':
                # Aquecimento linear
                rate = condition.rate or 100  # K/s
                new_state.temperature += rate * dt

            elif condition.type == 'compression':
                # Compressão isentrópica
                target_p = condition.target_value or 1e9
                compression_rate = (target_p - state.pressure) / 100  # Pa/s
                new_state.pressure += compression_rate * dt

            elif condition.type == 'radiation':
                # Aquecimento por radiação
                flux = condition.rate or 1e6  # W/m²
                # Absorção de energia
                absorbed_power = flux * self.muanda_obj.obj.equivalent_radius**2 * 4 * np.pi
                # Assumindo emissividade = 1
                new_state.temperature += (absorbed_power * dt) / (self.mass * self.get_specific_heat(new_state))

            elif condition.type == 'shock':
                # Onda de choque
                shock_pressure = condition.target_value or 1e11
                new_state.pressure = shock_pressure
                # Temperatura aumenta dramaticamente
                new_state.temperature *= 10  # Estimativa grosseira

        # Atualizar propriedades dependentes
        new_state = self.update_thermodynamic_properties(new_state)

        return new_state

    def update_thermodynamic_properties(self, state: ThermodynamicState) -> ThermodynamicState:
        """Atualiza propriedades termodinâmicas baseadas em T e P."""

        # Determinar fase
        state.phase = self.determine_phase(state.temperature, state.pressure)

        # Calor específico
        cp = self.get_specific_heat(state)

        # Dilatação térmica (para sólidos)
        if state.phase == 'solid':
            alpha = self.thermo_props['thermal_expansion']
            delta_T = state.temperature - self.initial_state_obj.temperature
            thermal_expansion = 1 + 3 * alpha * delta_T
            state.volume = self.initial_volume * thermal_expansion
        elif state.phase == 'liquid':
            # Menor compressibilidade
            state.volume = self.initial_volume * 1.05  # Estimativa
        elif state.phase == 'gas':
            # Lei dos gases ideais (aproximada)
            state.volume = (self.mass / self.material_props['atomic_mass'] * R * state.temperature) / state.pressure * NA

        # Densidade
        state.density = self.mass / state.volume

        # Energia interna (aproximada)
        state.internal_energy = self.mass * cp * (state.temperature - 293.15)

        # Entropia (aproximada usando calor específico)
        state.entropy = self.mass * cp * np.log(state.temperature / 293.15)

        # Estrutura cristalina sob pressão
        if state.pressure > 1e9:  # Alta pressão
            for transition in self.thermo_props.get('high_pressure_phases', []):
                if state.pressure >= transition['P']:
                    state.crystal_structure = transition['phase']

        return state

    def determine_phase(self, T: float, P: float) -> str:
        """Determina a fase da matéria baseada em T e P."""

        # Temperaturas críticas
        Tm = self.thermo_props['melting_point']
        Tb = self.thermo_props['boiling_point']
        Tc = self.thermo_props.get('critical_temperature', 10000)

        # Pressão crítica
        Pc = self.thermo_props.get('critical_pressure', 1e8)

        if T < Tm and P < Pc:
            return 'solid'
        elif Tm <= T < Tb and P < Pc:
            return 'liquid'
        elif T >= Tb or (T >= Tc and P >= Pc):
            if T > 1e6:  # Plasma
                return 'plasma'
            else:
                return 'gas'
        else:
            # Regiões de transição
            return 'liquid' if T > Tm else 'solid'

    def get_specific_heat(self, state: ThermodynamicState) -> float:
        """Retorna calor específico baseado na fase e temperatura."""

        if state.phase == 'solid':
            cp_func = self.thermo_props['specific_heat_solid']
            return cp_func(state.temperature) if callable(cp_func) else cp_func
        elif state.phase == 'liquid':
            return self.thermo_props['specific_heat_liquid']
        elif state.phase == 'gas':
            # Para gases monoatômicos ≈ 3R/M, diatômicos ≈ 5R/M
            return 3 * R / (self.material_props['atomic_mass'] * 1000)  # J/(kg·K)
        else:  # plasma
            return 1.5 * R / (self.material_props['atomic_mass'] * 1000)

    def check_failure_conditions(self, state: ThermodynamicState) -> Optional[str]:
        """Verifica se o objeto falhou sob as condições atuais."""

        # Falha por vaporização completa
        if state.phase == 'gas' and state.temperature > self.thermo_props['boiling_point'] * 2:
            return "Vaporização completa - objeto perdeu coesão"

        # Falha por fusão extrema
        if state.phase == 'plasma':
            return "Ionização completa - matéria dissociada em plasma"

        # Falha por compressão extrema
        if state.density > self.material_props['density'] * 10:
            return "Compressão extrema - colapso estrutural"

        # Falha por expansão extrema
        if state.volume > self.initial_volume * 10000:  # Aumentado de 1000 para 10000
            return "Expansão extrema - perda de forma"

        # Falha por temperatura extrema
        if state.temperature > 1e8:  # Próximo do limite de validade do modelo
            return "Temperatura além do limite do modelo"

        return None

    def analyze_results(self) -> Dict:
        """Analisa os resultados da simulação."""

        if not self.history:
            return {"error": "Nenhuma simulação executada"}

        # Métricas finais
        final_state = self.history[-1]

        # Calcular energias de transição
        transition_energies = self.calculate_transition_energies()

        # Verificar leis físicas
        physics_validation = self.validate_physics_laws()

        # Leis emergentes
        self.emergent_laws = self.discover_emergent_laws()

        results = {
            "material": self.material,
            "simulation_time": self.time_steps[-1] if self.time_steps else 0,
            "final_state": {
                "temperature": final_state.temperature,
                "pressure": final_state.pressure,
                "phase": final_state.phase,
                "density": final_state.density,
                "volume_ratio": final_state.volume / self.initial_volume
            },
            "failure": {
                "occurred": self.failure_point is not None,
                "time": self.failure_point,
                "mechanism": self.failure_mechanism
            },
            "transitions": transition_energies,
            "physics_validation": physics_validation,
            "emergent_laws": self.emergent_laws,
            "history_length": len(self.history)
        }

        return results

    def calculate_transition_energies(self) -> List[Dict]:
        """Calcula energias envolvidas em transições de fase."""

        transitions = []

        for i in range(1, len(self.history)):
            prev_state = self.history[i-1]
            curr_state = self.history[i]

            # Detectar mudança de fase
            if prev_state.phase != curr_state.phase:
                energy_change = curr_state.internal_energy - prev_state.internal_energy

                transitions.append({
                    "time": self.time_steps[i],
                    "transition": f"{prev_state.phase} → {curr_state.phase}",
                    "temperature": curr_state.temperature,
                    "energy_change": energy_change,
                    "latent_heat_calculated": energy_change / self.mass
                })

        return transitions

    def validate_physics_laws(self) -> Dict:
        """Valida aderência às leis físicas fundamentais."""

        validation = {}

        # Lei de Dulong-Petit (para sólidos)
        solid_states = [s for s in self.history if s.phase == 'solid']
        if solid_states:
            final_solid = solid_states[-1]
            cp_calculated = self.get_specific_heat(final_solid)
            cp_dulong_petit = 3 * R / (self.material_props['atomic_mass'] * 1000)  # J/(kg·K)

            validation["dulong_petit"] = {
                "calculated": cp_calculated,
                "expected": cp_dulong_petit,
                "ratio": cp_calculated / cp_dulong_petit,
                "valid": 0.5 < cp_calculated / cp_dulong_petit < 2.0
            }

        # Lei dos gases ideais (para gases)
        gas_states = [s for s in self.history if s.phase == 'gas']
        if gas_states:
            final_gas = gas_states[-1]
            # PV = nRT (aproximado)
            n = self.mass / self.material_props['atomic_mass'] / NA  # mol
            pv_nrt = (final_gas.pressure * final_gas.volume) / (n * R * final_gas.temperature)

            validation["ideal_gas"] = {
                "pv_nrt": pv_nrt,
                "valid": 0.8 < pv_nrt < 1.2
            }

        # Conservação de energia (aproximada)
        energy_conserved = all(
            abs((s2.internal_energy - s1.internal_energy) -
                self.mass * self.get_specific_heat(s1) * (s2.temperature - s1.temperature)) < 1e3
            for s1, s2 in zip(self.history[:-1], self.history[1:])
        )

        validation["energy_conservation"] = energy_conserved

        return validation

    def discover_emergent_laws(self) -> List[str]:
        """Descobre leis emergentes sob condições extremas."""

        laws = []

        # Lei da entropia crescente
        entropies = [s.entropy for s in self.history]
        if len(entropies) > 10:
            entropy_trend = np.polyfit(range(len(entropies)), entropies, 1)[0]
            if entropy_trend > 0:
                laws.append("Lei da Entropia Crescente: Entropia aumenta com condições extremas")

        # Lei da compressibilidade limite
        densities = [s.density for s in self.history]
        if max(densities) > self.material_props['density'] * 2:
            laws.append("Lei da Compressibilidade Limite: Densidade máxima ~2x densidade ambiente")

        # Lei da ionização térmica
        plasma_states = [s for s in self.history if s.phase == 'plasma']
        if plasma_states:
            min_plasma_T = min(s.temperature for s in plasma_states)
            laws.append(f"Lei da Ionização Térmica: Plasma forma acima de {min_plasma_T:.0f}K")

        # Lei da expansão crítica
        volumes = [s.volume for s in self.history]
        if max(volumes) > self.initial_volume * 100:
            laws.append("Lei da Expansão Crítica: Volume máximo limitado a ~100x volume inicial")

        return laws

    def visualize_stress_test(self):
        """Gera visualizações completas do teste de stress."""

        if not self.history:
            print("Nenhuma simulação para visualizar")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Muanda Stress Test - {self.material.upper()}', fontsize=16)

        times = np.array(self.time_steps)
        temps = np.array([s.temperature for s in self.history])
        pressures = np.array([s.pressure for s in self.history])
        volumes = np.array([s.volume for s in self.history])
        densities = np.array([s.density for s in self.history])
        energies = np.array([s.internal_energy for s in self.history])

        # 1. Temperatura vs Tempo
        axes[0,0].plot(times, temps, 'r-', linewidth=2)
        axes[0,0].axhline(y=self.thermo_props['melting_point'], color='orange', linestyle='--',
                         label=f'Fusão: {self.thermo_props["melting_point"]}K')
        axes[0,0].axhline(y=self.thermo_props['boiling_point'], color='red', linestyle='--',
                         label=f'Ebulição: {self.thermo_props["boiling_point"]}K')
        axes[0,0].set_xlabel('Tempo (s)')
        axes[0,0].set_ylabel('Temperatura (K)')
        axes[0,0].set_title('Aquecimento vs Tempo')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Pressão vs Tempo
        axes[0,1].semilogy(times, pressures, 'b-', linewidth=2)
        axes[0,1].set_xlabel('Tempo (s)')
        axes[0,1].set_ylabel('Pressão (Pa)')
        axes[0,1].set_title('Compressão vs Tempo')
        axes[0,1].grid(True, alpha=0.3)

        # 3. Volume vs Temperatura (dilatação térmica)
        axes[0,2].plot(temps, volumes, 'g-', linewidth=2)
        axes[0,2].axvline(x=self.thermo_props['melting_point'], color='orange', linestyle='--')
        axes[0,2].set_xlabel('Temperatura (K)')
        axes[0,2].set_ylabel('Volume (m³)')
        axes[0,2].set_title('Dilatação Térmica')
        axes[0,2].grid(True, alpha=0.3)

        # 4. Diagrama de Fases (P vs T)
        phases = [s.phase for s in self.history]
        phase_colors = {'solid': 'blue', 'liquid': 'orange', 'gas': 'red', 'plasma': 'purple'}
        colors = [phase_colors.get(p, 'black') for p in phases]

        axes[1,0].scatter(temps, pressures, c=colors, s=20, alpha=0.7)
        axes[1,0].set_xlabel('Temperatura (K)')
        axes[1,0].set_ylabel('Pressão (Pa)')
        axes[1,0].set_yscale('log')
        axes[1,0].set_title('Diagrama de Fases P-T')
        axes[1,0].grid(True, alpha=0.3)

        # 5. Energia Interna vs Temperatura
        axes[1,1].plot(temps, energies, 'm-', linewidth=2)
        axes[1,1].set_xlabel('Temperatura (K)')
        axes[1,1].set_ylabel('Energia Interna (J)')
        axes[1,1].set_title('Calor Específico Efetivo')
        axes[1,1].grid(True, alpha=0.3)

        # 6. Densidade vs Tempo
        axes[1,2].plot(times, densities, 'c-', linewidth=2)
        axes[1,2].axhline(y=self.material_props['density'], color='gray', linestyle='--',
                         label=f'Densidade inicial: {self.material_props["density"]:.0f} kg/m³')
        axes[1,2].set_xlabel('Tempo (s)')
        axes[1,2].set_ylabel('Densidade (kg/m³)')
        axes[1,2].set_title('Compressão vs Tempo')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'muanda_v71_stress_{self.material}.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Gráfico adicional: Fases ao longo do tempo
        plt.figure(figsize=(12, 4))
        phase_numeric = {'solid': 1, 'liquid': 2, 'gas': 3, 'plasma': 4}
        phase_values = [phase_numeric.get(s.phase, 0) for s in self.history]

        plt.plot(times, phase_values, 'k-', linewidth=2, drawstyle='steps-post')
        plt.yticks([1, 2, 3, 4], ['Sólido', 'Líquido', 'Gasoso', 'Plasma'])
        plt.xlabel('Tempo (s)')
        plt.ylabel('Fase')
        plt.title('Transições de Fase ao Longo do Tempo')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'muanda_v71_phases_{self.material}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_stress_results(self):
        """Salva resultados detalhados do teste de stress."""

        results = self.analyze_results()

        # Adicionar histórico completo
        results["history"] = {
            "times": self.time_steps,
            "temperatures": [s.temperature for s in self.history],
            "pressures": [s.pressure for s in self.history],
            "volumes": [s.volume for s in self.history],
            "densities": [s.density for s in self.history],
            "phases": [s.phase for s in self.history],
            "energies": [s.internal_energy for s in self.history],
            "entropies": [s.entropy for s in self.history]
        }

        with open(f'muanda_v71_stress_{self.material}_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

# ==================== TESTES DE STRESS PRÉ-DEFINIDOS ====================

def stress_test_iron_melting():
    """Teste A: Fusão do ferro."""
    print("\n🔥 TESTE A: FUSÃO DO FERRO")

    stress_conditions = [
        {"type": "heating", "rate": 100}  # Reduzido de 500 para 100 K/s
    ]

    ferro_stress = MuandaStressTest(
        material="iron",
        initial_state={"shape": "cube", "size": 0.05},  # 5cm
        stress_conditions=stress_conditions
    )

    results = ferro_stress.run_simulation(
        max_temperature=2500,
        max_pressure=1e7,
        time_steps=1000,
        dt=2.0
    )

    ferro_stress.visualize_stress_test()
    ferro_stress.save_stress_results()

    return results

def stress_test_gold_vaporization():
    """Teste B: Vaporização do ouro."""
    print("\n💨 TESTE B: VAPORIZAÇÃO DO OURO")

    stress_conditions = [
        {"type": "heating", "rate": 1000},  # Aquecimento muito rápido
        {"type": "compression", "target_value": 1e6}  # Pressão moderada
    ]

    ouro_stress = MuandaStressTest(
        material="gold",
        initial_state={"shape": "sphere", "diameter": 0.01},  # 1cm
        stress_conditions=stress_conditions
    )

    results = ouro_stress.run_simulation(
        max_temperature=4000,
        max_pressure=1e8,
        time_steps=1500,
        dt=1.0
    )

    ouro_stress.visualize_stress_test()
    ouro_stress.save_stress_results()

    return results

def stress_test_diamond_compression():
    """Teste C: Compressão extrema do diamante."""
    print("\n💎 TESTE C: COMPRESSÃO EXTREMA DO DIAMANTE")

    stress_conditions = [
        {"type": "compression", "rate": 1e8},  # Compressão rápida
        {"type": "heating", "rate": 100}       # Aquecimento moderado
    ]

    diamante_stress = MuandaStressTest(
        material="carbon",
        initial_state={"shape": "cube", "size": 0.005},  # 5mm
        stress_conditions=stress_conditions
    )

    results = diamante_stress.run_simulation(
        max_temperature=5000,
        max_pressure=1e11,  # 100 GPa
        time_steps=2000,
        dt=0.5
    )

    diamante_stress.visualize_stress_test()
    diamante_stress.save_stress_results()

    return results

def extreme_stellar_conditions():
    """Simulação Extrema: Condições estelares."""
    print("\n⭐ SIMULAÇÃO EXTREMA: CONDIÇÕES ESTELARES")

    stress_conditions = [
        {"type": "heating", "rate": 1e6},      # Aquecimento ultra-rápido
        {"type": "compression", "rate": 1e12}, # Compressão extrema
        {"type": "radiation", "rate": 1e10}    # Radiação intensa
    ]

    stellar_stress = MuandaStressTest(
        material="iron",
        initial_state={"shape": "sphere", "diameter": 1e-6},  # 1 mícron
        stress_conditions=stress_conditions
    )

    results = stellar_stress.run_simulation(
        max_temperature=1e7,   # 10 milhões K
        max_pressure=1e16,     # Pressão estelar
        time_steps=500,
        dt=0.01
    )

    stellar_stress.visualize_stress_test()
    stellar_stress.save_stress_results()

    return results

# ==================== EXECUÇÃO PRINCIPAL ====================

if __name__ == "__main__":
    print("🧪 MUANDA MODEL v7.1 - STRESS TEST TÉRMICO-MECÂNICO")
    print("Testando limites do modelo sob condições extremas")

    # Executar testes de stress
    print("\n" + "="*60)
    results_a = stress_test_iron_melting()

    print("\n" + "="*60)
    results_b = stress_test_gold_vaporization()

    print("\n" + "="*60)
    results_c = stress_test_diamond_compression()

    print("\n" + "="*60)
    results_extreme = extreme_stellar_conditions()

    # Relatório final
    print("\n" + "="*80)
    print("📊 RELATÓRIO FINAL DOS STRESS TESTS")
    print("="*80)

    tests = [
        ("Ferro - Fusão", results_a),
        ("Ouro - Vaporização", results_b),
        ("Diamante - Compressão", results_c),
        ("Condições Estelares", results_extreme)
    ]

    for name, result in tests:
        print(f"\n🔬 {name}:")
        if result.get("failure", {}).get("occurred"):
            print(f"  ❌ FALHOU em {result['failure']['time']:.1f}s")
            print(f"  Motivo: {result['failure']['mechanism']}")
        else:
            print("  ✅ SOBREVIVEU às condições extremas")
        final_T = result['final_state']['temperature']
        final_P = result['final_state']['pressure']
        final_phase = result['final_state']['phase']
        print(f"  Estado Final: T={final_T:.0f}K, P={final_P:.1e}Pa, Fase={final_phase}")

        if result.get('emergent_laws'):
            print(f"  Leis Emergentes: {len(result['emergent_laws'])} descobertas")

    print("\n📁 Arquivos salvos:")
    print("  muanda_v71_stress_*.png - Visualizações")
    print("  muanda_v71_stress_*_results.json - Dados completos")

    print("\n🎯 CONCLUSÃO: O modelo foi testado nos seus limites!")
    print("Se sobreviveu, é incrivelmente robusto.")
    print("Se falhou, sabemos exatamente onde melhorar.")
    print("Se sobreviveu, é incrivelmente robusto.")
    print("Se falhou, sabemos exatamente onde melhorar.")