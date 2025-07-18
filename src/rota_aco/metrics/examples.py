"""
Exemplos de uso do sistema de métricas expandido.

Este módulo demonstra como utilizar as estruturas de dados e configurações
do sistema de métricas para análise de resultados ACO.
"""

from datetime import datetime
from typing import List
import random

from .data_models import (
    Route, Solution, ExecutionData, IterationData,
    RouteQualityMetrics, ConvergenceMetrics, MetricsReport
)
from .config import MetricsConfig, create_academic_config


def create_sample_route(route_id: int = 1) -> Route:
    """
    Cria uma rota de exemplo para demonstração.
    
    Args:
        route_id: ID da rota para variação nos dados
        
    Returns:
        Route: Rota de exemplo
    """
    num_stops = random.randint(3, 8)
    stops = list(range(route_id * 10, route_id * 10 + num_stops))
    distances = [random.uniform(50, 200) for _ in range(num_stops - 1)]
    passenger_load = [random.randint(5, 25) for _ in range(num_stops - 1)]
    
    total_distance = sum(distances)
    total_passengers = sum(passenger_load)
    
    # Simular algumas violações ocasionais
    capacity_violations = 1 if total_passengers > 70 else 0
    opposite_violations = 1 if random.random() < 0.1 else 0
    is_valid = capacity_violations == 0 and opposite_violations == 0
    
    return Route(
        stops=stops,
        distances=distances,
        passenger_load=passenger_load,
        total_distance=total_distance,
        total_passengers=total_passengers,
        is_valid=is_valid,
        capacity_violations=capacity_violations,
        opposite_stops_violations=opposite_violations
    )


def create_sample_solution(num_routes: int = 3) -> Solution:
    """
    Cria uma solução de exemplo com múltiplas rotas.
    
    Args:
        num_routes: Número de rotas na solução
        
    Returns:
        Solution: Solução de exemplo
    """
    routes = [create_sample_route(i + 1) for i in range(num_routes)]
    
    total_distance = sum(route.total_distance for route in routes)
    total_passengers = sum(route.total_passengers for route in routes)
    
    # Fitness baseado em distância e número de veículos
    fitness_time = total_distance
    fitness_vehicle = len(routes) * 1000 + total_distance  # Penaliza mais veículos
    
    is_feasible = all(route.is_valid for route in routes)
    
    return Solution(
        routes=routes,
        total_vehicles=len(routes),
        total_distance=total_distance,
        total_passengers_served=total_passengers,
        fitness_time=fitness_time,
        fitness_vehicle=fitness_vehicle,
        is_feasible=is_feasible,
        generation_time=random.uniform(0.1, 2.0)
    )


def create_sample_iteration_data(iteration: int, best_so_far: float = None) -> IterationData:
    """
    Cria dados de iteração de exemplo.
    
    Args:
        iteration: Número da iteração
        best_so_far: Melhor fitness até agora (para simular convergência)
        
    Returns:
        IterationData: Dados da iteração
    """
    # Simular convergência gradual
    if best_so_far is None:
        best_fitness = random.uniform(1000, 5000)
    else:
        # Melhoria gradual com chance de estagnação
        improvement = random.uniform(-50, 10) if random.random() < 0.7 else 0
        best_fitness = max(best_so_far + improvement, best_so_far * 0.99)
    
    avg_fitness = best_fitness * random.uniform(1.1, 1.5)
    population_variance = random.uniform(100, 1000)
    
    best_solution = create_sample_solution()
    best_solution.fitness_time = best_fitness
    
    return IterationData(
        iteration=iteration,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
        population_variance=population_variance,
        best_solution=best_solution
    )


def create_sample_execution_data(algorithm_type: str = "ACS-TIME", 
                               num_iterations: int = 100) -> ExecutionData:
    """
    Cria dados de execução completos de exemplo.
    
    Args:
        algorithm_type: Tipo do algoritmo ('ACS-TIME' ou 'ACS-VEHICLE')
        num_iterations: Número de iterações
        
    Returns:
        ExecutionData: Dados de execução completos
    """
    config = {
        "ants": random.randint(10, 50),
        "iterations": num_iterations,
        "evaporation": random.uniform(0.1, 0.3),
        "alpha": random.uniform(0.5, 1.5),
        "beta": random.uniform(1.0, 3.0)
    }
    
    # Gerar dados de iteração com convergência simulada
    iterations_data = []
    best_fitness = None
    
    for i in range(num_iterations):
        iteration_data = create_sample_iteration_data(i, best_fitness)
        iterations_data.append(iteration_data)
        best_fitness = iteration_data.best_fitness
    
    final_solution = iterations_data[-1].best_solution if iterations_data else create_sample_solution()
    
    return ExecutionData(
        algorithm_type=algorithm_type,
        config=config,
        routes=final_solution.routes,
        iterations_data=iterations_data,
        execution_time=random.uniform(30, 300),
        final_solution=final_solution,
        success=True
    )


def demonstrate_basic_usage():
    """Demonstra uso básico do sistema de métricas."""
    print("=== Demonstração do Sistema de Métricas Expandido ===\n")
    
    # 1. Criar configuração
    print("1. Criando configuração acadêmica...")
    config = create_academic_config()
    print(f"   - DPI: {config.figure_dpi}")
    print(f"   - Formatos: {config.output_formats}")
    print(f"   - Diretório base: {config.base_output_dir}")
    
    # 2. Criar dados de exemplo
    print("\n2. Gerando dados de exemplo...")
    execution_data = create_sample_execution_data("ACS-TIME", 50)
    print(f"   - Algoritmo: {execution_data.algorithm_type}")
    print(f"   - Iterações: {len(execution_data.iterations_data)}")
    print(f"   - Tempo de execução: {execution_data.execution_time:.2f}s")
    print(f"   - Rotas geradas: {len(execution_data.routes)}")
    
    # 3. Analisar qualidade das rotas
    print("\n3. Analisando qualidade das rotas...")
    valid_routes = sum(1 for route in execution_data.routes if route.is_valid)
    total_routes = len(execution_data.routes)
    valid_percentage = (valid_routes / total_routes * 100) if total_routes > 0 else 0
    
    total_violations = sum(route.capacity_violations + route.opposite_stops_violations 
                          for route in execution_data.routes)
    
    print(f"   - Rotas válidas: {valid_routes}/{total_routes} ({valid_percentage:.1f}%)")
    print(f"   - Total de violações: {total_violations}")
    
    # 4. Analisar convergência
    print("\n4. Analisando convergência...")
    if execution_data.iterations_data:
        initial_fitness = execution_data.iterations_data[0].best_fitness
        final_fitness = execution_data.iterations_data[-1].best_fitness
        improvement = ((initial_fitness - final_fitness) / initial_fitness * 100)
        
        print(f"   - Fitness inicial: {initial_fitness:.2f}")
        print(f"   - Fitness final: {final_fitness:.2f}")
        print(f"   - Melhoria: {improvement:.2f}%")
    
    # 5. Demonstrar conversão para dicionário
    print("\n5. Convertendo dados para formato de relatório...")
    if execution_data.final_solution:
        solution_dict = {
            'total_vehicles': execution_data.final_solution.total_vehicles,
            'total_distance': execution_data.final_solution.total_distance,
            'total_passengers': execution_data.final_solution.total_passengers_served,
            'is_feasible': execution_data.final_solution.is_feasible
        }
        
        print("   Resumo da solução:")
        for key, value in solution_dict.items():
            print(f"     - {key}: {value}")
    
    print("\n=== Demonstração concluída ===")


def demonstrate_multiple_executions():
    """Demonstra análise de múltiplas execuções."""
    print("\n=== Análise de Múltiplas Execuções ===\n")
    
    # Gerar múltiplas execuções
    executions = []
    for i in range(5):
        algorithm = "ACS-TIME" if i % 2 == 0 else "ACS-VEHICLE"
        execution = create_sample_execution_data(algorithm, random.randint(30, 100))
        executions.append(execution)
    
    print(f"Geradas {len(executions)} execuções para análise comparativa:")
    
    # Análise comparativa básica
    acs_time_executions = [e for e in executions if e.algorithm_type == "ACS-TIME"]
    acs_vehicle_executions = [e for e in executions if e.algorithm_type == "ACS-VEHICLE"]
    
    print(f"\n- ACS-TIME: {len(acs_time_executions)} execuções")
    print(f"- ACS-VEHICLE: {len(acs_vehicle_executions)} execuções")
    
    # Comparar tempos de execução
    if acs_time_executions:
        avg_time_time = sum(e.execution_time for e in acs_time_executions) / len(acs_time_executions)
        print(f"- Tempo médio ACS-TIME: {avg_time_time:.2f}s")
    
    if acs_vehicle_executions:
        avg_time_vehicle = sum(e.execution_time for e in acs_vehicle_executions) / len(acs_vehicle_executions)
        print(f"- Tempo médio ACS-VEHICLE: {avg_time_vehicle:.2f}s")
    
    # Comparar qualidade das soluções
    all_final_fitness = []
    for execution in executions:
        if execution.final_solution:
            all_final_fitness.append(execution.final_solution.fitness_time)
    
    if all_final_fitness:
        avg_fitness = sum(all_final_fitness) / len(all_final_fitness)
        min_fitness = min(all_final_fitness)
        max_fitness = max(all_final_fitness)
        
        print(f"\nQualidade das soluções:")
        print(f"- Fitness médio: {avg_fitness:.2f}")
        print(f"- Melhor fitness: {min_fitness:.2f}")
        print(f"- Pior fitness: {max_fitness:.2f}")


if __name__ == "__main__":
    # Executar demonstrações
    demonstrate_basic_usage()
    demonstrate_multiple_executions()
    
    print("\n" + "="*50)
    print("Para usar o sistema de métricas em seu código:")
    print("1. Importe as classes necessárias")
    print("2. Configure o MetricsConfig conforme necessário")
    print("3. Colete dados durante execuções ACO")
    print("4. Use as classes de métricas para análise")
    print("5. Gere relatórios e visualizações")
    print("="*50)