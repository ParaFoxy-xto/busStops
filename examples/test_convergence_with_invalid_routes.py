# examples/test_convergence_with_invalid_routes.py

"""
Script de teste para verificar os gráficos de convergência com rotas inválidas.

Este script testa a nova funcionalidade de visualização que inclui
análise de rotas inválidas junto com a convergência da Qualidade Q.
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from rota_aco.viz.convergence_with_invalid_routes import ConvergenceInvalidRoutesVisualizer


def create_mock_controller_history():
    """
    Cria dados simulados de histórico do controlador para teste.
    """
    import random
    
    # Simular 50 iterações
    history = []
    
    # Valores iniciais
    initial_quality = 100.0
    time_dist = 5000.0
    vehicle_dist = 5200.0
    time_coverage = 0.6
    vehicle_coverage = 0.65
    
    for i in range(50):
        # Simular melhoria gradual com alguma variabilidade
        quality_improvement = random.uniform(0.5, 2.0)
        initial_quality -= quality_improvement
        
        # Simular flutuações nas métricas
        time_dist_variation = random.uniform(-50, 20)
        vehicle_dist_variation = random.uniform(-40, 30)
        
        time_dist += time_dist_variation
        vehicle_dist += vehicle_dist_variation
        
        # Melhorar cobertura gradualmente
        time_coverage = min(1.0, time_coverage + random.uniform(0, 0.02))
        vehicle_coverage = min(1.0, vehicle_coverage + random.uniform(0, 0.015))
        
        # Simular número de rotas
        time_routes = random.randint(2, 5)
        vehicle_routes = random.randint(2, 4)
        
        entry = {
            'iteration': i,
            'best_quality_so_far': max(10.0, initial_quality),  # Não deixar ficar muito baixo
            'time_metrics': {
                'dist': max(1000.0, time_dist),
                'count': time_routes,
                'coverage': time_coverage
            },
            'vehicle_metrics': {
                'dist': max(1000.0, vehicle_dist),
                'count': vehicle_routes,
                'coverage': vehicle_coverage
            }
        }
        
        history.append(entry)
    
    return history


def test_convergence_visualization():
    """
    Testa a geração de gráficos de convergência com rotas inválidas.
    """
    print("=== TESTE: Visualização de Convergência com Rotas Inválidas ===")
    
    # Criar dados simulados
    controller_history = create_mock_controller_history()
    print(f"Dados simulados criados: {len(controller_history)} iterações")
    
    # Criar visualizador
    output_dir = "output/test_convergence"
    visualizer = ConvergenceInvalidRoutesVisualizer(output_dir)
    print(f"Visualizador criado, saída em: {output_dir}")
    
    try:
        # Gerar gráfico principal
        print("\n1. Gerando gráfico principal de convergência com rotas inválidas...")
        main_plot_path = visualizer.generate_convergence_with_invalid_routes_plot(
            controller_history,
            title="Teste: Convergência ACS com Análise de Rotas Inválidas",
            save_filename="test_convergence_main.png"
        )
        print(f"   ✓ Gráfico principal salvo em: {main_plot_path}")
        
        # Gerar análise detalhada
        print("\n2. Gerando análise detalhada de rotas inválidas...")
        detailed_plot_path = visualizer.generate_detailed_invalid_routes_analysis(
            controller_history,
            save_filename="test_detailed_invalid_routes.png"
        )
        print(f"   ✓ Análise detalhada salva em: {detailed_plot_path}")
        
        print(f"\n✅ Teste concluído com sucesso!")
        print(f"   Arquivos gerados em: {output_dir}")
        
        # Listar arquivos gerados
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"   Arquivos: {', '.join(files)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_terminology_changes():
    """
    Testa se as mudanças de terminologia foram aplicadas corretamente.
    """
    print("\n=== TESTE: Verificação de Terminologia ===")
    
    # Verificar se o termo "Melhor qualidade" foi substituído por "Qualidade Q"
    files_to_check = [
        "src/rota_aco/cli/run.py",
        "src/rota_aco/metrics/report_generator.py"
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        full_path = Path(__file__).parent.parent / file_path
        
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Verificar se ainda contém "Melhor qualidade"
                if "Melhor qualidade" in content:
                    issues_found.append(f"{file_path}: ainda contém 'Melhor qualidade'")
                
                # Verificar se contém "Qualidade Q"
                if "Qualidade Q" in content:
                    print(f"   ✓ {file_path}: contém 'Qualidade Q'")
                else:
                    print(f"   ⚠ {file_path}: não contém 'Qualidade Q'")
                    
            except Exception as e:
                print(f"   ❌ Erro ao verificar {file_path}: {e}")
        else:
            print(f"   ⚠ Arquivo não encontrado: {file_path}")
    
    if issues_found:
        print(f"\n❌ Problemas encontrados:")
        for issue in issues_found:
            print(f"   - {issue}")
        return False
    else:
        print(f"\n✅ Terminologia verificada com sucesso!")
        return True


def main():
    """
    Executa todos os testes.
    """
    print("TESTE DE FUNCIONALIDADES: Convergência com Rotas Inválidas")
    print("=" * 60)
    
    # Teste 1: Visualização
    test1_success = test_convergence_visualization()
    
    # Teste 2: Terminologia
    test2_success = test_terminology_changes()
    
    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO DOS TESTES")
    print("=" * 60)
    
    print(f"1. Visualização de convergência: {'✅ PASSOU' if test1_success else '❌ FALHOU'}")
    print(f"2. Verificação de terminologia: {'✅ PASSOU' if test2_success else '❌ FALHOU'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 Todos os testes passaram!")
        print(f"As modificações foram implementadas com sucesso:")
        print(f"  - Gráficos de rotas inválidas adicionados")
        print(f"  - Termo 'Melhor qualidade' substituído por 'Qualidade Q'")
    else:
        print(f"\n⚠ Alguns testes falharam. Verifique os problemas acima.")
    
    return test1_success and test2_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)