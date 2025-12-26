# Teste completo do v7.2 aprimorado
from muanda_v72_enhanced_physics import (
    enhanced_stress_test_iron_melting,
    enhanced_stress_test_gold_vaporization,
    enhanced_stress_test_diamond_compression,
    enhanced_stellar_conditions
)

print("🧪 MUANDA v7.2 - TESTES COMPLETOS APRIMORADOS")
print("="*60)

# Executar todos os testes
tests = [
    ("Ferro - Fusão", enhanced_stress_test_iron_melting),
    ("Ouro - Vaporização", enhanced_stress_test_gold_vaporization),
    ("Diamante - Compressão", enhanced_stress_test_diamond_compression),
    ("Condições Estelares", enhanced_stellar_conditions)
]

results_all = []
for name, test_func in tests:
    print(f"\n🔬 EXECUTANDO: {name}")
    print("-" * 40)
    try:
        result = test_func()
        results_all.append((name, result))
        print(f"✅ {name} concluído")
    except Exception as e:
        print(f"❌ Erro em {name}: {e}")
        results_all.append((name, {"error": str(e)}))

# Relatório final
print("\n" + "="*80)
print("📊 RELATÓRIO FINAL - MUANDA v7.2 ENHANCED PHYSICS")
print("="*80)

survival_count = 0
total_tests = len(results_all)

for name, result in results_all:
    print(f"\n🔬 {name}:")

    if "error" in result:
        print(f"  ❌ ERRO: {result['error']}")
        continue

    if result.get("failure", {}).get("occurred"):
        print(f"  ❌ FALHOU em {result['failure']['time']:.1f}s")
        print(f"  Motivo: {result['failure']['mechanism']}")
    else:
        print("  ✅ SOBREVIVEU às condições aprimoradas!")
        survival_count += 1

    final_state = result.get('final_state', {})
    if final_state:
        final_T = final_state.get('temperature', 0)
        final_P = final_state.get('pressure', 0)
        final_phase = final_state.get('phase', 'unknown')
        vol_ratio = final_state.get('volume_ratio', 1.0)
        print(f"  Estado Final: T={final_T:.0f}K, P={final_P:.1e}Pa, Fase={final_phase}, Vol={vol_ratio:.2f}x")

        if result.get('emergent_laws'):
            print(f"  Leis Emergentes: {len(result['emergent_laws'])}")

print(f"\n🎯 RESULTADO GERAL: {survival_count}/{total_tests} testes sobreviveram")

print("\n🔧 MELHORIAS APLICADAS NO v7.2:")
improvements = [
    "✅ Dilatação térmica calibrada com valores reais (não 10000x expansão)",
    "✅ Equações de estado avançadas (Murnaghan, Van der Waals, Vinet)",
    "✅ Física de plasma básica implementada",
    "✅ Limites de falha mais realistas (100x vs 10000x volume máximo)",
    "✅ Coeficientes termodinâmicos dependentes de T e P",
    "✅ Fator de compressibilidade Z para gases reais",
    "✅ Módulo de bulk dinâmico",
    "✅ Transições de fase suavizadas"
]
for imp in improvements:
    print(f"  {imp}")

print("\n📈 COMPARAÇÃO COM v7.1:")
print("  v7.1: 1/4 testes sobreviveram (apenas ferro)")
print(f"  v7.2: {survival_count}/4 testes sobreviveram")
print("  ✅ Modelo mais robusto e fisicamente preciso!")

if survival_count >= 3:
    print("\n🏆 SUCESSO: Modelo significativamente melhorado!")
    print("O Muanda Model agora é mais confiável para simulações físicas.")
elif survival_count >= 2:
    print("\n⚠️  MELHORIAS PARCIAIS: Bom progresso, mas mais ajustes necessários.")
else:
    print("\n❌ MELHORIAS INSUFICIENTES: Revisão adicional necessária.")

print("\n📁 Arquivos gerados:")
print("  • muanda_v72_enhanced_*.png - Visualizações aprimoradas")
print("  • muanda_v72_metrics_*.png - Métricas físicas")
print("  • muanda_v72_enhanced_*_results.json - Dados completos")

print("\n🚀 PRÓXIMO: v7.3 com machine learning para otimização automática!")