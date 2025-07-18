"""
Exemplo de integração do sistema de métricas com algoritmos ACO.

Este módulo demonstra como usar o sistema de métricas com o código ACO
existente sem modificar a implementação original.
"""

import sys
import os
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rota_aco.aco.controller import ACSController
from rota_aco.metrics.aco_integration import (
    ACOMetricsIntegrator, 
    create_metrics_enabled_controller,
    run_aco_with_metrics
)
from rota_aco.metrics.config import MetricsConfig, create_academic_config


def demonstrate_basic_integration():
    """
    Demonstra integração básica do sistema de métricas.
    
    Este exemplo mostra como usar o integrador para coletar dados
    de uma execução ACO sem modificar o código original.
    """
    print("=== Demonstração de Integração Básica ===\n")
    
    # Configuração das métricas para apresentação acadêmica
    metrics_config = create_academic_config()
    print(f"Configuração de métricas:")
    print(f"  - Diretório de saída: {metrics_config.base_output_dir}")
    print(f"  - Formatos de visualização: {metrics_config.output_formats}")
    print(f"  - DPI das figuras: {metrics_config.figure_dpi}")
    
    # Criar integrador
    integrator = ACOMetricsIntegrator(metrics_config)
    
    print(f"\nIntegrador criado com sucesso!")
    print(f"  - Collector inicializado: {integrator.collector is not None}")
    print(f"  - Cache de rotas: {len(integrator._route_cache)} entradas")
    
    # Simular parâmetros de execução
    algorithm_params = {
        'alpha': 1.0,
        'beta': 2.0,
        'rho': 0.1,
        'Q': 1.0
    }
    
    problem_params = {
        'capacity': 70,
        'max_route_length': 100,
        'max_route_attempts': 10
    }
    
    quality_weights = {
        'w_c': 10.0,  # Peso da cobertura
        'w_r': 1.0,   # Peso do número de rotas
        'w_d': 0.5    # Peso da distância
    }
    
    print(f"\nParâmetros configurados:")
    print(f"  - Algoritmo: {algorithm_params}")
    print(f"  - Problema: {problem_params}")
    print(f"  - Qualidade: {quality_weights}")
    
    # Demonstrar determinação do tipo de algoritmo
    algorithm_type = integrator._determine_algorithm_type(quality_weights)
    print(f"\nTipo de algoritmo determinado: {algorithm_type}")
    
    # Simular início de coleta
    execution_id = integrator.collector.start_execution(
        algorithm_type=algorithm_type,
        config={
            'algorithm_params': algorithm_params,
            'problem_params': problem_params,
            'quality_weights': quality_weights
        }
    )
    
    print(f"\nColeta de dados iniciada:")
    print(f"  - ID da execução: {execution_id}")
    print(f"  - Execução atual: {integrator.current_execution_id}")
    
    # Simular dados de iteração
    from rota_aco.metrics.data_models import Route, Solution
    
    # Criar rota de exemplo
    sample_route = Route(
        stops=[1, 2, 3, 4],
        distances=[100.0, 150.0, 200.0],
        passenger_load=[20, 25, 15],
        total_distance=450.0,
        total_passengers=60,
        is_valid=True,
        capacity_violations=0,
        opposite_stops_violations=0
    )
    
    sample_solution = Solution(
        routes=[sample_route],
        total_vehicles=1,
        total_distance=450.0,
        total_passengers_served=60,
        fitness_time=450.0,
        fitness_vehicle=1450.0,
        is_feasible=True,
        generation_time=1.2
    )
    
    # Simular algumas iterações
    for iteration in range(5):
        fitness = 500.0 - (iteration * 10)  # Simular melhoria
        avg_fitness = fitness + 50.0
        variance = 25.0 - (iteration * 2)
        
        integrator.collector.record_iteration(
            iteration=iteration,
            best_fitness=fitness,
            avg_fitness=avg_fitness,
            population_variance=variance,
            best_solution=sample_solution,
            additional_metrics={
                'convergence_rate': iteration * 0.1,
                'diversity_index': variance / 25.0
            }
        )
        
        print(f"  - Iteração {iteration}: fitness={fitness:.1f}, variance={variance:.1f}")
    
    # Finalizar execução
    integrator.collector.record_final_solution(
        solution=sample_solution,
        execution_time=45.5,
        success=True
    )
    
    execution_data = integrator.collector.finish_execution(save_to_disk=True)
    
    print(f"\nExecução finalizada:")
    print(f"  - Sucesso: {execution_data.success}")
    print(f"  - Tempo de execução: {execution_data.execution_time:.1f}s")
    print(f"  - Iterações registradas: {len(execution_data.iterations_data)}")
    print(f"  - Solução final: {execution_data.final_solution is not None}")
    
    # Mostrar resumo
    summary = integrator.collector.get_execution_summary()
    print(f"\nResumo das execuções:")
    print(f"  - Total de execuções: {summary['total_executions']}")
    print(f"  - Taxa de sucesso: {summary['success_rate']:.1%}")
    print(f"  - Tempo médio: {summary['avg_execution_time']:.1f}s")
    
    return integrator


def demonstrate_controller_wrapping():
    """
    Demonstra como criar um wrapper para o ACSController.
    
    Este exemplo mostra como instrumentar um controller existente
    para coleta automática de métricas.
    """
    print("\n=== Demonstração de Wrapper do Controller ===\n")
    
    # Configuração simplificada
    metrics_config = MetricsConfig(
        enable_visualizations=False,  # Desabilitar para exemplo rápido
        enable_reports=False
    )
    
    integrator = ACOMetricsIntegrator(metrics_config)
    
    # Simular parâmetros do controller
    algorithm_params = {'alpha': 1.0, 'beta': 2.0, 'rho': 0.1, 'Q': 1.0}
    problem_params = {'capacity': 70, 'max_route_length': 100}
    quality_weights = {'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5}
    
    # Criar mock controller para demonstração
    class MockController:
        def __init__(self):
            self.history = []
            self.meta_edges = {(1, 2): {'time': 100}, (2, 3): {'time': 150}}
            self.stops_to_visit = [2, 3]
            self.capacity = 70
            self.opposites = {}
        
        def run(self, n_ants, n_iterations, quality_weights, verbose=False):
            # Simular execução
            for i in range(n_iterations):
                self._record_history(
                    iteration_num=i,
                    time_solution=[[1, 2, 3]],
                    vehicle_solution=[[1, 2], [2, 3]],
                    chosen_solution_quality=100.0 - i,
                    best_overall_quality=100.0 - i
                )
            
            return [[1, 2, 3]], 250.0, 1, 0.95
        
        def _record_history(self, iteration_num, time_solution, vehicle_solution, 
                           chosen_solution_quality, best_overall_quality):
            self.history.append({
                'iteration': iteration_num,
                'time_metrics': {'dist': 250, 'count': 1, 'coverage': 0.95},
                'vehicle_metrics': {'dist': 300, 'count': 2, 'coverage': 0.90},
                'chosen_quality': chosen_solution_quality,
                'best_quality_so_far': best_overall_quality
            })
        
        def _calculate_solution_quality(self, routes, w_c, w_r, w_d):
            return 100.0  # Simplified
        
        def _get_solution_metrics(self, routes):
            return 250.0, len(routes), 0.95
    
    mock_controller = MockController()
    
    # Criar wrapper
    wrapped_run = integrator.wrap_controller_execution(
        mock_controller, algorithm_params, problem_params, quality_weights
    )
    
    print("Wrapper criado com sucesso!")
    print("Executando algoritmo com coleta de métricas...")
    
    # Executar com wrapper
    result = wrapped_run(n_ants=10, n_iterations=3, verbose=False)
    
    print(f"\nResultado da execução:")
    print(f"  - Rotas: {result[0]}")
    print(f"  - Distância total: {result[1]}")
    print(f"  - Número de rotas: {result[2]}")
    print(f"  - Cobertura: {result[3]:.1%}")
    
    # Verificar dados coletados
    collected_data = integrator.get_collected_data()
    if collected_data:
        execution = collected_data[-1]
        print(f"\nDados coletados:")
        print(f"  - ID da execução: {execution.execution_id}")
        print(f"  - Tipo de algoritmo: {execution.algorithm_type}")
        print(f"  - Iterações registradas: {len(execution.iterations_data)}")
        print(f"  - Sucesso: {execution.success}")
        
        if execution.iterations_data:
            first_iter = execution.iterations_data[0]
            last_iter = execution.iterations_data[-1]
            print(f"  - Fitness inicial: {first_iter.best_fitness:.1f}")
            print(f"  - Fitness final: {last_iter.best_fitness:.1f}")
            print(f"  - Melhoria: {first_iter.best_fitness - last_iter.best_fitness:.1f}")
    
    return integrator


def demonstrate_file_operations():
    """
    Demonstra operações de arquivo do sistema de métricas.
    """
    print("\n=== Demonstração de Operações de Arquivo ===\n")
    
    # Usar configuração que salva arquivos
    config = MetricsConfig()
    integrator = ACOMetricsIntegrator(config)
    
    print(f"Diretórios de saída:")
    print(f"  - Base: {config.base_output_dir}")
    print(f"  - Dados de execução: {config.get_execution_data_path()}")
    print(f"  - Relatórios: {config.get_reports_path()}")
    
    # Verificar se diretórios foram criados
    import os
    dirs_exist = all(os.path.exists(path) for path in [
        config.get_execution_data_path(),
        config.get_reports_path(),
        config.get_visualizations_path()
    ])
    
    print(f"  - Diretórios criados: {dirs_exist}")
    
    # Listar execuções existentes
    existing_executions = integrator.collector.list_executions()
    print(f"\nExecuções existentes: {len(existing_executions)}")
    
    if existing_executions:
        print("IDs das execuções:")
        for exec_id in existing_executions[:5]:  # Mostrar apenas os primeiros 5
            print(f"  - {exec_id}")
        
        # Tentar carregar uma execução
        try:
            first_execution = integrator.collector.load_execution_data(existing_executions[0])
            print(f"\nExecução carregada com sucesso:")
            print(f"  - ID: {first_execution.execution_id}")
            print(f"  - Tipo: {first_execution.algorithm_type}")
            print(f"  - Sucesso: {first_execution.success}")
            print(f"  - Tempo: {first_execution.execution_time:.1f}s")
        except Exception as e:
            print(f"Erro ao carregar execução: {e}")
    
    return integrator


def main():
    """Função principal que executa todas as demonstrações."""
    print("Sistema de Métricas - Demonstração de Integração ACO")
    print("=" * 60)
    
    try:
        # Demonstração 1: Integração básica
        integrator1 = demonstrate_basic_integration()
        
        # Demonstração 2: Wrapper do controller
        integrator2 = demonstrate_controller_wrapping()
        
        # Demonstração 3: Operações de arquivo
        integrator3 = demonstrate_file_operations()
        
        print("\n" + "=" * 60)
        print("Todas as demonstrações concluídas com sucesso!")
        
    except Exception as e:
        print(f"\nErro durante demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()