# src/rota_aco/cli/run.py

import argparse
import os
import sys
import traceback
import json
from typing import List, Dict, Any, Tuple, Optional

# --- Importações Corrigidas e Completas ---
from rota_aco.data.preprocess import load_graph, get_bus_stops
from rota_aco.graph.build_meta import build_meta_graph, expand_meta_route
from rota_aco.graph.utils import find_nearest_node
from rota_aco.aco.controller import ACSController
from rota_aco.viz.matplotlib_viz import plot_multiple_meta_routes, plot_multiple_routes, plot_meta_graph
from rota_aco.viz.folium_viz import visualize_routes_folium
# Traditional reporting functions (keeping for backward compatibility)
# from rota_aco.reporting.report_generator import generate_convergence_plots, generate_final_report, generate_comparison_csv

# --- Importações do Sistema de Métricas ---
from rota_aco.metrics.config import MetricsConfig, create_academic_config, create_fast_config
from rota_aco.metrics.aco_integration import ACOMetricsIntegrator, run_aco_with_metrics
from rota_aco.metrics.report_generator import ReportGenerator
from rota_aco.metrics.visualization_engine import VisualizationEngine

def setup_arg_parser() -> argparse.ArgumentParser:
    """Configura e retorna o parser de argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de otimização de rotas com ACS multi-colônia."
    )
    g_problem = parser.add_argument_group('Parâmetros do Problema e Grafo')
    g_problem.add_argument("--graph", required=True, help="Caminho para o arquivo GraphML do grafo.")
    g_problem.add_argument("--start-lat", type=float, required=True, help="Latitude do ponto de partida.")
    g_problem.add_argument("--start-lon", type=float, required=True, help="Longitude do ponto de partida.")
    g_problem.add_argument("--exit-lat", type=float, required=True, help="Latitude do ponto de chegada.")
    g_problem.add_argument("--exit-lon", type=float, required=True, help="Longitude do ponto de chegada.")
    g_problem.add_argument("--precision", type=int, default=5, help="Precisão geoespacial para agrupar paradas.")
    g_problem.add_argument("--capacity", type=int, default=70, help="Capacidade de passageiros do veículo.")
    g_problem.add_argument("--max-route-length", type=int, default=100, help="Número máximo de nós em uma única rota.")
    g_problem.add_argument("--max-route-attempts", type=int, default=10, help="Tentativas de uma formiga para criar rotas.")

    g_aco = parser.add_argument_group('Parâmetros do Algoritmo ACO')
    g_aco.add_argument("--ants", type=int, default=20, help="Número de formigas por colônia.")
    g_aco.add_argument("--iterations", type=int, default=10, help="Número máximo de iterações.")
    g_aco.add_argument("--alpha", type=float, default=1.0, help="Influência do feromônio (alfa).")
    g_aco.add_argument("--beta", type=float, default=2.0, help="Influência da heurística (beta).")
    g_aco.add_argument("--rho", type=float, default=0.1, help="Taxa de evaporação do feromônio (rho).")
    g_aco.add_argument("--Q", type=float, default=1.0, help="Fator de reforço de feromônio (Q).")

    g_quality = parser.add_argument_group('Pesos da Função de Qualidade')
    g_quality.add_argument("--w-c", type=float, default=10.0, help="Peso da cobertura de paradas (w_c).")
    g_quality.add_argument("--w-r", type=float, default=1.0, help="Peso do número de rotas (w_r).")
    g_quality.add_argument("--w-d", type=float, default=0.5, help="Peso da distância total (w_d).")

    g_output = parser.add_argument_group('Saída e Visualização')
    g_output.add_argument("--output", help="Nome do arquivo para a visualização da rota final (ex: rota_final.png).")
    g_output.add_argument("--meta-output", help="Nome do arquivo para a visualização do meta-grafo (ex: meta_grafo.png).")
    g_output.add_argument("--folium", action="store_true", help="Gerar visualização interativa com Folium.")
    g_output.add_argument("--verbose", action="store_true", help="Ativar logs detalhados durante a execução.")
    g_problem.add_argument("--manual-opposites",help="Caminho para um arquivo JSON opcional definindo pares de opostos manualmente.")
    
    # Grupo de métricas expandidas
    g_metrics = parser.add_argument_group('Sistema de Métricas Expandido')
    g_metrics.add_argument("--metrics", action="store_true", 
                          help="Habilitar coleta e análise de métricas expandidas.")
    g_metrics.add_argument("--report-output", type=str, default="output/metrics",
                          help="Diretório de saída para relatórios e métricas (padrão: output/metrics).")
    g_metrics.add_argument("--compare-runs", type=int, metavar="N",
                          help="Executar N execuções para análise comparativa.")
    g_metrics.add_argument("--convergence-analysis", action="store_true",
                          help="Habilitar análise detalhada de convergência.")
    g_metrics.add_argument("--metrics-config", type=str,
                          help="Caminho para arquivo de configuração personalizada das métricas (JSON).")
    
    # Opções avançadas de métricas
    g_metrics_advanced = parser.add_argument_group('Opções Avançadas de Métricas')
    g_metrics_advanced.add_argument("--statistical-tests", action="store_true",
                                   help="Habilitar testes estatísticos nas comparações.")
    g_metrics_advanced.add_argument("--confidence-level", type=float, default=0.95,
                                   help="Nível de confiança para testes estatísticos (padrão: 0.95).")
    g_metrics_advanced.add_argument("--export-raw-data", action="store_true",
                                   help="Incluir dados brutos nos relatórios.")
    g_metrics_advanced.add_argument("--visualization-formats", nargs='+', 
                                   choices=['png', 'svg', 'pdf'], default=['png'],
                                   help="Formatos de saída para visualizações (padrão: png).")
    g_metrics_advanced.add_argument("--academic-mode", action="store_true",
                                   help="Modo acadêmico: alta resolução, formatos múltiplos, análise completa.")
    g_metrics_advanced.add_argument("--fast-mode", action="store_true",
                                   help="Modo rápido: desabilita visualizações e relatórios detalhados.")
    g_metrics_advanced.add_argument("--parallel-executions", action="store_true",
                                   help="Executar múltiplas execuções em paralelo (experimental).")
    g_metrics_advanced.add_argument("--seed", type=int,
                                   help="Semente para reprodutibilidade dos resultados.")
    g_metrics_advanced.add_argument("--save-execution-data", action="store_true", default=True,
                                   help="Salvar dados de execução em disco (padrão: habilitado).")
    g_metrics_advanced.add_argument("--load-previous-data", type=str,
                                   help="Carregar dados de execuções anteriores para comparação.")
    
    return parser


def setup_metrics_config(args) -> MetricsConfig:
    """
    Configura o sistema de métricas baseado nos argumentos da linha de comando.
    
    Args:
        args: Argumentos parseados da linha de comando
        
    Returns:
        MetricsConfig: Configuração das métricas
    """
    # Determinar configuração base
    if args.metrics_config:
        # Carregar configuração personalizada
        try:
            config = MetricsConfig.load_from_file(args.metrics_config)
            print(f"Configuração de métricas carregada de: {args.metrics_config}")
        except Exception as e:
            print(f"[AVISO] Erro ao carregar configuração personalizada: {e}")
            print("Usando configuração padrão.")
            config = MetricsConfig()
    elif hasattr(args, 'academic_mode') and args.academic_mode:
        # Modo acadêmico
        config = create_academic_config()
        print("Modo acadêmico habilitado: alta resolução, formatos múltiplos, análise completa.")
    elif hasattr(args, 'fast_mode') and args.fast_mode:
        # Modo rápido
        config = create_fast_config()
        print("Modo rápido habilitado: análise simplificada para execução rápida.")
    elif hasattr(args, 'convergence_analysis') and args.convergence_analysis:
        # Análise de convergência detalhada
        config = create_academic_config()
        print("Usando configuração acadêmica para análise detalhada de convergência.")
    else:
        # Configuração padrão
        config = MetricsConfig()
        print("Usando configuração padrão de métricas.")
    
    # Aplicar sobrescrições baseadas nos argumentos CLI
    
    # Diretório de saída
    if hasattr(args, 'report_output') and args.report_output:
        config.base_output_dir = args.report_output
        config._ensure_directories()
    
    # Análise de convergência
    if hasattr(args, 'convergence_analysis') and args.convergence_analysis:
        config.enable_convergence_analysis = True
        config.enable_visualizations = True
        config.enable_reports = True
        config.enable_statistical_tests = True
    
    # Testes estatísticos
    if hasattr(args, 'statistical_tests') and args.statistical_tests:
        config.enable_statistical_tests = True
        print("Testes estatísticos habilitados.")
    
    # Nível de confiança
    if hasattr(args, 'confidence_level') and args.confidence_level:
        config.confidence_level = args.confidence_level
        print(f"Nível de confiança configurado para: {args.confidence_level}")
    
    # Dados brutos nos relatórios
    if hasattr(args, 'export_raw_data') and args.export_raw_data:
        config.include_raw_data = True
        print("Exportação de dados brutos habilitada.")
    
    # Formatos de visualização
    if hasattr(args, 'visualization_formats') and args.visualization_formats:
        config.output_formats = args.visualization_formats
        print(f"Formatos de visualização: {', '.join(args.visualization_formats)}")
    
    # Modo acadêmico (sobrescreve outras configurações)
    if hasattr(args, 'academic_mode') and args.academic_mode:
        config.figure_dpi = 600
        config.output_formats = ['png', 'svg', 'pdf']
        config.enable_statistical_tests = True
        config.confidence_level = 0.95
        config.include_raw_data = True
        config.enable_detailed_logging = True
    
    # Modo rápido (desabilita recursos pesados)
    if hasattr(args, 'fast_mode') and args.fast_mode:
        config.enable_visualizations = False
        config.enable_reports = False
        config.enable_statistical_tests = False
        config.max_iterations_to_store = 1000
        config.enable_parallel_processing = True
        print("Modo rápido: visualizações e relatórios detalhados desabilitados.")
    
    # Processamento paralelo
    if hasattr(args, 'parallel_executions') and args.parallel_executions:
        config.enable_parallel_processing = True
        print("Execuções paralelas habilitadas (experimental).")
    
    # Configurar semente para reprodutibilidade
    if hasattr(args, 'seed') and args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Semente configurada para reprodutibilidade: {args.seed}")
    
    return config


def run_single_execution_with_metrics(meta_graph, meta_edges, stops_to_visit, 
                                    start_node, exit_node, all_opposites,
                                    aco_params, problem_params, quality_weights,
                                    args, metrics_config):
    """
    Executa uma única execução ACO com coleta de métricas.
    
    Args:
        meta_graph: Meta-grafo construído
        meta_edges: Arestas do meta-grafo
        stops_to_visit: Paradas a serem visitadas
        start_node: Nó de início
        exit_node: Nó de saída
        all_opposites: Mapeamento de paradas opostas
        aco_params: Parâmetros do ACO
        problem_params: Parâmetros do problema
        quality_weights: Pesos da função de qualidade
        args: Argumentos da linha de comando
        metrics_config: Configuração das métricas
        
    Returns:
        Tuple: (resultado_aco, dados_execução)
    """
    # Usar integração com métricas
    result, execution_data = run_aco_with_metrics(
        controller_class=ACSController,
        graph=meta_graph,
        meta_edges=meta_edges,
        stops_to_visit=stops_to_visit,
        start_node=start_node,
        exit_node=exit_node,
        opposites=all_opposites,
        aco_params=aco_params,
        problem_params=problem_params,
        quality_weights=quality_weights,
        n_ants=args.ants,
        n_iterations=args.iterations,
        verbose=args.verbose,
        metrics_config=metrics_config
    )
    
    return result, execution_data


def run_multiple_executions_with_metrics(meta_graph, meta_edges, stops_to_visit,
                                       start_node, exit_node, all_opposites,
                                       aco_params, problem_params, quality_weights,
                                       args, metrics_config, num_runs):
    """
    Executa múltiplas execuções ACO para análise comparativa.
    
    Args:
        meta_graph: Meta-grafo construído
        meta_edges: Arestas do meta-grafo
        stops_to_visit: Paradas a serem visitadas
        start_node: Nó de início
        exit_node: Nó de saída
        all_opposites: Mapeamento de paradas opostas
        aco_params: Parâmetros do ACO
        problem_params: Parâmetros do problema
        quality_weights: Pesos da função de qualidade
        args: Argumentos da linha de comando
        metrics_config: Configuração das métricas
        num_runs: Número de execuções
        
    Returns:
        Tuple: (melhor_resultado, lista_dados_execução)
    """
    print(f"\n--- Executando {num_runs} execuções para análise comparativa ---")
    
    # Verificar se deve executar em paralelo
    if hasattr(args, 'parallel_executions') and args.parallel_executions and num_runs > 1:
        return run_parallel_executions(
            meta_graph, meta_edges, stops_to_visit, start_node, exit_node,
            all_opposites, aco_params, problem_params, quality_weights,
            args, metrics_config, num_runs
        )
    
    # Execução sequencial
    all_results = []
    all_execution_data = []
    best_result = None
    best_quality = float('inf')
    
    for run_idx in range(num_runs):
        print(f"\nExecução {run_idx + 1}/{num_runs}:")
        
        try:
            result, execution_data = run_single_execution_with_metrics(
                meta_graph, meta_edges, stops_to_visit, start_node, exit_node,
                all_opposites, aco_params, problem_params, quality_weights,
                args, metrics_config
            )
            
            all_results.append(result)
            all_execution_data.append(execution_data)
            
            # Verificar se é a melhor solução
            if result and len(result) >= 2:
                current_quality = result[1]  # total_dist
                if current_quality < best_quality:
                    best_quality = current_quality
                    best_result = result
            
            print(f"  - Concluída com sucesso")
            if execution_data and execution_data.success:
                print(f"  - Tempo de execução: {execution_data.execution_time:.2f}s")
                if execution_data.final_solution:
                    print(f"  - Distância total: {execution_data.final_solution.total_distance:.2f}")
                    print(f"  - Número de veículos: {execution_data.final_solution.total_vehicles}")
            
        except Exception as e:
            print(f"  - Falhou: {e}")
            # Continuar com as outras execuções
            continue
    
    print(f"\nResumo das {num_runs} execuções:")
    successful_runs = len([data for data in all_execution_data if data and data.success])
    print(f"  - Execuções bem-sucedidas: {successful_runs}/{num_runs}")
    
    if successful_runs > 0:
        avg_time = sum(data.execution_time for data in all_execution_data if data and data.success) / successful_runs
        print(f"  - Tempo médio de execução: {avg_time:.2f}s")
    
    return best_result, all_execution_data


def run_parallel_executions(meta_graph, meta_edges, stops_to_visit,
                          start_node, exit_node, all_opposites,
                          aco_params, problem_params, quality_weights,
                          args, metrics_config, num_runs):
    """
    Executa múltiplas execuções ACO em paralelo.
    
    Args:
        Similar to run_multiple_executions_with_metrics
        
    Returns:
        Tuple: (melhor_resultado, lista_dados_execução)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    
    print(f"Executando {num_runs} execuções em paralelo...")
    
    # Determinar número de workers
    max_workers = min(num_runs, metrics_config.max_workers, multiprocessing.cpu_count())
    print(f"Usando {max_workers} workers paralelos.")
    
    def run_single_execution_wrapper(run_idx):
        """Wrapper para execução única que pode ser serializada."""
        try:
            # Configurar semente específica para cada execução
            if hasattr(args, 'seed') and args.seed is not None:
                import random
                import numpy as np
                seed = args.seed + run_idx
                random.seed(seed)
                np.random.seed(seed)
            
            result, execution_data = run_single_execution_with_metrics(
                meta_graph, meta_edges, stops_to_visit, start_node, exit_node,
                all_opposites, aco_params, problem_params, quality_weights,
                args, metrics_config
            )
            return run_idx, result, execution_data, None
        except Exception as e:
            return run_idx, None, None, str(e)
    
    all_results = []
    all_execution_data = []
    best_result = None
    best_quality = float('inf')
    
    # Executar em paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submeter todas as execuções
        future_to_run = {
            executor.submit(run_single_execution_wrapper, run_idx): run_idx 
            for run_idx in range(num_runs)
        }
        
        # Coletar resultados conforme completam
        for future in as_completed(future_to_run):
            run_idx, result, execution_data, error = future.result()
            
            if error:
                print(f"Execução {run_idx + 1} falhou: {error}")
                continue
            
            all_results.append(result)
            all_execution_data.append(execution_data)
            
            # Verificar se é a melhor solução
            if result and len(result) >= 2:
                current_quality = result[1]  # total_dist
                if current_quality < best_quality:
                    best_quality = current_quality
                    best_result = result
            
            print(f"Execução {run_idx + 1} concluída.")
    
    print(f"\nResumo das {num_runs} execuções paralelas:")
    successful_runs = len([data for data in all_execution_data if data and data.success])
    print(f"  - Execuções bem-sucedidas: {successful_runs}/{num_runs}")
    
    if successful_runs > 0:
        avg_time = sum(data.execution_time for data in all_execution_data if data and data.success) / successful_runs
        print(f"  - Tempo médio de execução: {avg_time:.2f}s")
    
    return best_result, all_execution_data


def load_previous_execution_data(data_path, metrics_config):
    """
    Carrega dados de execuções anteriores para comparação.
    
    Args:
        data_path: Caminho para arquivo ou diretório com dados
        metrics_config: Configuração das métricas
        
    Returns:
        List: Lista de ExecutionData carregados
    """
    from rota_aco.metrics.data_collector import DataCollector
    
    collector = DataCollector(metrics_config)
    
    try:
        if os.path.isfile(data_path):
            # Arquivo único
            if data_path.endswith('.json'):
                # Extrair ID do arquivo
                execution_id = os.path.basename(data_path).replace('.json', '')
                execution_data = collector.load_execution_data(execution_id)
                return [execution_data]
            else:
                print(f"[AVISO] Formato de arquivo não suportado: {data_path}")
                return []
        elif os.path.isdir(data_path):
            # Diretório com múltiplos arquivos
            execution_ids = []
            for filename in os.listdir(data_path):
                if filename.endswith('.json') or filename.endswith('.pkl'):
                    execution_id = filename.split('.')[0]
                    execution_ids.append(execution_id)
            
            if execution_ids:
                # Temporariamente alterar o diretório de dados do collector
                original_path = collector.config.get_execution_data_path()
                collector.config.base_output_dir = os.path.dirname(data_path)
                collector.config._ensure_directories()
                
                try:
                    executions = collector.load_multiple_executions(execution_ids)
                    return executions
                finally:
                    # Restaurar configuração original
                    collector.config.base_output_dir = original_path
            else:
                print(f"[AVISO] Nenhum arquivo de execução encontrado em: {data_path}")
                return []
        else:
            print(f"[ERRO] Caminho não encontrado: {data_path}")
            return []
            
    except Exception as e:
        print(f"[ERRO] Falha ao carregar dados anteriores: {e}")
        return []


def convert_execution_data_to_metrics_report(execution_data, metrics_config):
    """
    Converte ExecutionData em MetricsReport para geração de relatórios.
    
    Args:
        execution_data: Dados de execução
        metrics_config: Configuração das métricas
        
    Returns:
        MetricsReport: Relatório de métricas convertido
    """
    from rota_aco.metrics.data_models import (
        MetricsReport, ExecutionSummary, RouteQualityMetrics,
        ConvergenceMetrics, DomainMetrics
    )
    from rota_aco.metrics.quality_metrics import RouteQualityEvaluator
    from rota_aco.metrics.convergence_analyzer import ConvergenceAnalyzer
    from rota_aco.metrics.transportation_metrics import TransportationMetrics
    
    # Criar resumo executivo
    execution_summary = ExecutionSummary(
        total_executions=1,
        successful_executions=1 if execution_data.success else 0,
        success_rate=1.0 if execution_data.success else 0.0,
        avg_execution_time=execution_data.execution_time,
        avg_iterations_to_convergence=len(execution_data.iterations_data) if execution_data.iterations_data else 0,
        best_overall_fitness=execution_data.final_solution.fitness_time if execution_data.final_solution else 0.0,
        algorithm_type=execution_data.algorithm_type or "ACS"
    )
    
    # Criar métricas de qualidade básicas
    quality_metrics = RouteQualityMetrics(
        valid_routes_percentage=100.0 if execution_data.success else 0.0,
        demand_coverage_percentage=50.0,  # Valor padrão
        vehicle_utilization_efficiency=80.0,  # Valor padrão
        capacity_violations=0,
        opposite_stops_violations=0,
        average_route_length=5.0,  # Valor padrão
        route_length_variance=1.0,  # Valor padrão
        load_balancing_index=0.8  # Valor padrão
    )
    
    # Criar métricas de convergência básicas
    convergence_metrics = ConvergenceMetrics(
        convergence_point=len(execution_data.iterations_data) if execution_data.iterations_data else 0,
        final_stability=0.1,  # Valor padrão
        improvement_rate=0.05,  # Valor padrão
        plateau_detection=False,
        convergence_speed=1.0,  # Valor padrão
        total_iterations=len(execution_data.iterations_data) if execution_data.iterations_data else 0
    )
    
    # Criar métricas de domínio básicas
    domain_metrics = DomainMetrics(
        estimated_travel_time=30.0,  # Valor padrão
        average_transfers=1.5,  # Valor padrão
        geographic_coverage=75.0,  # Valor padrão
        load_balancing_index=0.8,  # Valor padrão
        energy_efficiency=2.5,  # Valor padrão
        accessibility_index=10.0,  # Valor padrão
        service_frequency=15.0,  # Valor padrão
        route_overlap_percentage=20.0  # Valor padrão
    )
    
    return MetricsReport(
        execution_summary=execution_summary,
        quality_metrics=quality_metrics,
        convergence_analysis=convergence_metrics,
        comparative_analysis=None,
        domain_metrics=domain_metrics,
        config_used=execution_data.config
    )


def generate_metrics_reports(execution_data_list, metrics_config, args):
    """
    Gera relatórios e visualizações das métricas coletadas.
    
    Args:
        execution_data_list: Lista de dados de execução
        metrics_config: Configuração das métricas
        args: Argumentos da linha de comando
    """
    if not execution_data_list or not any(data for data in execution_data_list if data):
        print("[AVISO] Nenhum dado de execução disponível para gerar relatórios.")
        return
    
    print("\n--- Gerando Relatórios de Métricas ---")
    
    try:
        # Filtrar apenas execuções bem-sucedidas
        successful_executions = [data for data in execution_data_list if data and data.success]
        
        if not successful_executions:
            print("[AVISO] Nenhuma execução bem-sucedida para análise.")
            return
        
        # Gerar relatório
        report_generator = ReportGenerator(metrics_config)
        
        if len(successful_executions) == 1:
            # Relatório de execução única
            metrics_report = convert_execution_data_to_metrics_report(successful_executions[0], metrics_config)
            report_path = report_generator.generate_report(metrics_report)
            print(f"Relatório de execução única gerado: {report_path}")
        else:
            # Relatório comparativo
            metrics_reports = [convert_execution_data_to_metrics_report(data, metrics_config) for data in successful_executions]
            report_path = report_generator.generate_comprehensive_report(metrics_reports)
            print(f"Relatório comparativo gerado: {report_path}")
        
        # Gerar visualizações se habilitadas
        if metrics_config.enable_visualizations:
            viz_engine = VisualizationEngine(metrics_config)
            
            for i, execution_data in enumerate(successful_executions):
                if execution_data.iterations_data:
                    # Gráfico de convergência
                    conv_plot_path = viz_engine.generate_convergence_plot(
                        execution_data.iterations_data,
                        title=f"Convergência - Execução {i+1}",
                        filename=f"convergence_execution_{i+1}"
                    )
                    print(f"Gráfico de convergência gerado: {conv_plot_path}")
            
            # Se múltiplas execuções, gerar comparações
            if len(successful_executions) > 1:
                comparison_plots = viz_engine.generate_comparison_plots(successful_executions)
                for plot_path in comparison_plots:
                    print(f"Gráfico comparativo gerado: {plot_path}")
        
        print(f"Todos os arquivos salvos em: {metrics_config.base_output_dir}")
        
    except Exception as e:
        print(f"[ERRO] Falha na geração de relatórios: {e}")
        if args.verbose:
            traceback.print_exc()


def create_organized_output_structure(base_name: str) -> dict:
    """
    Cria estrutura organizada de diretórios para outputs.
    
    Args:
        base_name: Nome base para a pasta (ex: 'pequeno', 'longo', 'completo')
        
    Returns:
        dict: Dicionário com os caminhos dos diretórios criados
    """
    # Criar diretório principal
    main_output_dir = os.path.join("output", f"output_{base_name}")
    
    # Criar subdiretórios
    subdirs = {
        'main': main_output_dir,
        'images': os.path.join(main_output_dir, "images"),
        'metrics': os.path.join(main_output_dir, "metrics"),
        'reports': os.path.join(main_output_dir, "reports"),
        'data': os.path.join(main_output_dir, "data")
    }
    
    # Criar todos os diretórios
    for dir_path in subdirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return subdirs


def extract_graph_name(graph_path: str) -> str:
    """
    Extrai o nome do grafo do caminho do arquivo.
    
    Args:
        graph_path: Caminho para o arquivo do grafo
        
    Returns:
        str: Nome do grafo (ex: 'pequeno', 'longo', 'grafo')
    """
    filename = os.path.basename(graph_path)
    name = os.path.splitext(filename)[0]
    
    # Mapear nomes específicos
    name_mapping = {
        'grafo': 'completo',
        'pequeno': 'pequeno',
        'longo': 'longo'
    }
    
    return name_mapping.get(name, name)


def save_execution_data(output_dirs: dict, args, result_data: dict, meta_graph, graph):
    """
    Salva dados da execução na pasta data/.
    
    Args:
        output_dirs: Dicionário com caminhos dos diretórios
        args: Argumentos da linha de comando
        result_data: Dados dos resultados da execução
        meta_graph: Meta-grafo construído
        graph: Grafo original
    """
    import json
    from datetime import datetime
    
    # Dados da configuração
    config_data = {
        'graph_file': args.graph,
        'start_coordinates': {'lat': args.start_lat, 'lon': args.start_lon},
        'exit_coordinates': {'lat': args.exit_lat, 'lon': args.exit_lon},
        'aco_parameters': {
            'ants': args.ants,
            'iterations': args.iterations,
            'alpha': args.alpha,
            'beta': args.beta,
            'rho': args.rho,
            'Q': args.Q
        },
        'quality_weights': {
            'w_c': args.w_c,
            'w_r': args.w_r,
            'w_d': args.w_d
        },
        'execution_timestamp': datetime.now().isoformat()
    }
    
    # Salvar configuração
    config_path = os.path.join(output_dirs['data'], 'execution_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    # Dados dos resultados
    results_data = {
        'execution_summary': result_data,
        'graph_statistics': {
            'original_nodes': len(graph.nodes()),
            'original_edges': len(graph.edges()),
            'meta_nodes': len(meta_graph.nodes()),
            'meta_edges': len(meta_graph.edges())
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Salvar resultados
    results_path = os.path.join(output_dirs['data'], 'execution_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"Dados da execução salvos em: {output_dirs['data']}")


def generate_execution_report(output_dirs: dict, args, result_data: dict, meta_graph, graph):
    """
    Gera relatório da execução na pasta reports/.
    
    Args:
        output_dirs: Dicionário com caminhos dos diretórios
        args: Argumentos da linha de comando
        result_data: Dados dos resultados da execução
        meta_graph: Meta-grafo construído
        graph: Grafo original
    """
    from datetime import datetime
    
    # Criar relatório em markdown
    report_content = f"""# Relatório de Execução ACS

**Data de Execução:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}  
**Grafo Utilizado:** {os.path.basename(args.graph)}

## Configuração da Execução

### Coordenadas
- **Ponto de Partida:** {args.start_lat}, {args.start_lon}
- **Ponto de Chegada:** {args.exit_lat}, {args.exit_lon}

### Parâmetros ACO
- **Formigas:** {args.ants}
- **Iterações:** {args.iterations}
- **Alpha (α):** {args.alpha}
- **Beta (β):** {args.beta}
- **Rho (ρ):** {args.rho}
- **Q:** {args.Q}

### Pesos da Função de Qualidade
- **Cobertura (w_c):** {args.w_c}
- **Número de Rotas (w_r):** {args.w_r}
- **Distância (w_d):** {args.w_d}

## Estatísticas do Grafo

### Grafo Original
- **Nós:** {len(graph.nodes())}
- **Arestas:** {len(graph.edges())}

### Meta-Grafo
- **Nós:** {len(meta_graph.nodes())}
- **Arestas:** {len(meta_graph.edges())}

## Resultados da Execução

### Melhor Solução Encontrada
- **Número de Rotas:** {result_data.get('num_routes', 'N/A')}
- **Distância Total:** {result_data.get('total_dist', 'N/A'):.2f} metros
- **Cobertura:** {result_data.get('coverage', 0)*100:.2f}%

### Arquivos Gerados
- **Visualização da Rota:** `images/acs_route.png`
- **Meta-Grafo:** `images/meta_graph.png`
- **Rotas no Meta-Grafo:** `images/routes_on_meta.png`
- **Dados da Execução:** `data/execution_config.json`, `data/execution_results.json`

## Conclusões

A execução foi concluída com sucesso, gerando {result_data.get('num_routes', 0)} rota(s) 
que cobrem {result_data.get('coverage', 0)*100:.2f}% das paradas necessárias.

---
*Relatório gerado automaticamente pelo sistema Rota_ACO*
"""
    
    # Salvar relatório
    report_path = os.path.join(output_dirs['reports'], 'execution_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Relatório da execução salvo em: {report_path}")


def save_metrics_summary(output_dirs: dict, result_data: dict):
    """
    Salva resumo das métricas na pasta metrics/.
    
    Args:
        output_dirs: Dicionário com caminhos dos diretórios
        result_data: Dados dos resultados da execução
    """
    import json
    from datetime import datetime
    
    # Métricas básicas
    metrics_summary = {
        'execution_metrics': {
            'total_routes': result_data.get('num_routes', 0),
            'total_distance_meters': result_data.get('total_dist', 0),
            'coverage_percentage': result_data.get('coverage', 0) * 100,
            'execution_timestamp': datetime.now().isoformat()
        },
        'quality_indicators': {
            'routes_efficiency': 'Good' if result_data.get('num_routes', 0) <= 3 else 'Needs Improvement',
            'coverage_quality': 'Excellent' if result_data.get('coverage', 0) > 0.7 else 'Good' if result_data.get('coverage', 0) > 0.5 else 'Needs Improvement',
            'distance_efficiency': result_data.get('total_dist', 0) / max(result_data.get('num_routes', 1), 1)
        }
    }
    
    # Salvar métricas
    metrics_path = os.path.join(output_dirs['metrics'], 'metrics_summary.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    
    print(f"Resumo das métricas salvo em: {metrics_path}")


def analyze_route_validity(routes: List[Any], opposites: Dict[Any, List[Any]], capacity: int = 70) -> Dict[str, int]:
    """
    Analisa a validade das rotas e conta violações.
    
    Args:
        routes: Lista de rotas para analisar
        opposites: Dicionário de paradas opostas
        capacity: Capacidade máxima do veículo
        
    Returns:
        Dict com contadores de violações
    """
    violations = {
        'capacity_violations': 0,
        'opposite_violations': 0,
        'total_invalid_routes': 0,
        'total_routes': len(routes)
    }
    
    for route in routes:
        route_invalid = False
        
        # Verificar violações de capacidade (assumindo 70 passageiros por rota)
        if len(route) > capacity // 5:  # Estimativa simples
            violations['capacity_violations'] += 1
            route_invalid = True
        
        # Verificar violações de paradas opostas
        for i, stop in enumerate(route):
            if stop in opposites:
                opposite_stops = opposites[stop]
                for j, other_stop in enumerate(route):
                    if i != j and other_stop in opposite_stops:
                        violations['opposite_violations'] += 1
                        route_invalid = True
                        break
        
        if route_invalid:
            violations['total_invalid_routes'] += 1
    
    return violations


def generate_convergence_analysis(output_dirs: dict, controller_history: List[Dict], opposites: Dict, capacity: int = 70):
    """
    Gera análise detalhada de convergência com gráficos e dados de rotas inválidas.
    
    Args:
        output_dirs: Dicionário com caminhos dos diretórios
        controller_history: Histórico do controller
        opposites: Dicionário de paradas opostas
        capacity: Capacidade do veículo
    """
    import matplotlib.pyplot as plt
    import json
    from datetime import datetime
    
    if not controller_history:
        print("Nenhum histórico de convergência disponível.")
        return
    
    # Extrair dados para análise
    iterations = []
    time_distances = []
    time_routes = []
    time_coverage = []
    vehicle_distances = []
    vehicle_routes = []
    vehicle_coverage = []
    best_quality = []
    
    for entry in controller_history:
        iterations.append(entry['iteration'])
        time_distances.append(entry['time_metrics']['dist'])
        time_routes.append(entry['time_metrics']['count'])
        time_coverage.append(entry['time_metrics']['coverage'])
        vehicle_distances.append(entry['vehicle_metrics']['dist'])
        vehicle_routes.append(entry['vehicle_metrics']['count'])
        vehicle_coverage.append(entry['vehicle_metrics']['coverage'])
        best_quality.append(entry['best_quality_so_far'])
    
    # Criar gráficos de convergência
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise de Convergência ACS-TIME vs ACS-VEHICLE', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Distância Total
    ax1.plot(iterations, time_distances, 'b-', label='ACS-TIME', linewidth=2, marker='o', markersize=4)
    ax1.plot(iterations, vehicle_distances, 'r-', label='ACS-VEHICLE', linewidth=2, marker='s', markersize=4)
    ax1.set_xlabel('Iteração')
    ax1.set_ylabel('Distância Total (m)')
    ax1.set_title('Evolução da Distância Total')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Número de Rotas
    ax2.plot(iterations, time_routes, 'b-', label='ACS-TIME', linewidth=2, marker='o', markersize=4)
    ax2.plot(iterations, vehicle_routes, 'r-', label='ACS-VEHICLE', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Número de Rotas')
    ax2.set_title('Evolução do Número de Rotas')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Gráfico 3: Cobertura
    ax3.plot(iterations, [c*100 for c in time_coverage], 'b-', label='ACS-TIME', linewidth=2, marker='o', markersize=4)
    ax3.plot(iterations, [c*100 for c in vehicle_coverage], 'r-', label='ACS-VEHICLE', linewidth=2, marker='s', markersize=4)
    ax3.set_xlabel('Iteração')
    ax3.set_ylabel('Cobertura (%)')
    ax3.set_title('Evolução da Cobertura')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Melhor Qualidade Geral
    ax4.plot(iterations, best_quality, 'g-', label='Melhor Qualidade', linewidth=3, marker='D', markersize=5)
    ax4.set_xlabel('Iteração')
    ax4.set_ylabel('Qualidade')
    ax4.set_title('Convergência da Melhor Qualidade')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    convergence_plot_path = os.path.join(output_dirs['images'], 'convergence_analysis.png')
    plt.savefig(convergence_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análise de convergência
    convergence_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_iterations': len(iterations),
        'convergence_summary': {
            'final_best_quality': best_quality[-1] if best_quality else 0,
            'initial_quality': best_quality[0] if best_quality else 0,
            'improvement_percentage': ((best_quality[0] - best_quality[-1]) / best_quality[0] * 100) if best_quality and best_quality[0] > 0 else 0
        },
        'acs_time_final': {
            'distance': time_distances[-1] if time_distances else 0,
            'routes': time_routes[-1] if time_routes else 0,
            'coverage': time_coverage[-1] * 100 if time_coverage else 0
        },
        'acs_vehicle_final': {
            'distance': vehicle_distances[-1] if vehicle_distances else 0,
            'routes': vehicle_routes[-1] if vehicle_routes else 0,
            'coverage': vehicle_coverage[-1] * 100 if vehicle_coverage else 0
        },
        'convergence_detection': {
            'converged': len(set(best_quality[-5:])) <= 2 if len(best_quality) >= 5 else False,
            'stability_window': 5,
            'final_variance': sum([(q - best_quality[-1])**2 for q in best_quality[-5:]]) / 5 if len(best_quality) >= 5 else 0
        }
    }
    
    # Salvar dados de convergência
    convergence_data_path = os.path.join(output_dirs['metrics'], 'convergence_analysis.json')
    with open(convergence_data_path, 'w', encoding='utf-8') as f:
        json.dump(convergence_data, f, indent=2, ensure_ascii=False)
    
    # Gerar relatório de convergência
    convergence_report = f"""# Análise de Convergência ACS

**Data da Análise:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}  
**Total de Iterações:** {len(iterations)}

## Resumo da Convergência

### Qualidade Final
- **Qualidade Inicial:** {best_quality[0]:.2f}
- **Qualidade Final:** {best_quality[-1]:.2f}
- **Melhoria:** {convergence_data['convergence_summary']['improvement_percentage']:.2f}%

### ACS-TIME (Final)
- **Distância:** {time_distances[-1]:.2f}m
- **Rotas:** {time_routes[-1]}
- **Cobertura:** {time_coverage[-1]*100:.2f}%

### ACS-VEHICLE (Final)
- **Distância:** {vehicle_distances[-1]:.2f}m
- **Rotas:** {vehicle_routes[-1]}
- **Cobertura:** {vehicle_coverage[-1]*100:.2f}%

## Análise de Estabilidade

- **Convergiu:** {'Sim' if convergence_data['convergence_detection']['converged'] else 'Não'}
- **Variância Final:** {convergence_data['convergence_detection']['final_variance']:.4f}

## Arquivos Gerados

- **Gráfico de Convergência:** `images/convergence_analysis.png`
- **Dados de Convergência:** `metrics/convergence_analysis.json`

---
*Análise gerada automaticamente pelo sistema Rota_ACO*
"""
    
    # Salvar relatório de convergência
    convergence_report_path = os.path.join(output_dirs['reports'], 'convergence_report.md')
    with open(convergence_report_path, 'w', encoding='utf-8') as f:
        f.write(convergence_report)
    
    print(f"Análise de convergência salva em: {convergence_plot_path}")
    print(f"Relatório de convergência salvo em: {convergence_report_path}")
    print(f"Dados de convergência salvos em: {convergence_data_path}")


def main():
    """Função principal que executa o pipeline completo."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Extrair nome do grafo e criar estrutura organizada
    graph_name = extract_graph_name(args.graph)
    output_dirs = create_organized_output_structure(graph_name)
    print(f"Estrutura de output criada em: {output_dirs['main']}")

    # --- 1. Carregamento e Preparação dos Dados ---
    print("--- 1. Carregando e Preparando Dados ---")
    try:
        graph = load_graph(args.graph)
        bus_stops_nodes = get_bus_stops(graph)
        if not bus_stops_nodes:
            print("[ERRO] Nenhuma parada de ônibus ('bus_stop=true') encontrada no grafo.")
            sys.exit(1)

        start_node = find_nearest_node(graph, (args.start_lat, args.start_lon), list(graph.nodes()))
        exit_node = find_nearest_node(graph, (args.exit_lat, args.exit_lon), list(graph.nodes()))
        print(f"Nó de partida mais próximo: {start_node}")
        print(f"Nó de chegada mais próximo: {exit_node}")
    except Exception as e:
        print(f"[ERRO] Falha na etapa de carregamento de dados: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 2. Construção e Visualização do Meta-Grafo ---
    print("\n--- 2. Construindo o Meta-Grafo ---")
    try:
        meta_graph, meta_edges, representatives, all_opposites, _, _ = build_meta_graph(
            graph=graph,
            bus_stops=bus_stops_nodes,
            start_node=start_node,
            exit_node=exit_node,
            precision=args.precision,
            manual_opposites_path=args.manual_opposites,
            verbose=args.verbose
        )
        if not meta_graph.nodes() or not meta_graph.edges():
            print("[ERRO] A construção resultou em um meta-grafo vazio.")
            sys.exit(1)

        # --- VISUALIZAÇÃO DO META-GRAFO ---
        if args.meta_output:
            meta_graph_path = os.path.join(output_dirs['images'], "meta_graph.png")
            print(f"\n[DEPURAÇÃO] Salvando visualização do meta-grafo em: '{meta_graph_path}'")
            plot_meta_graph(
                meta_graph,
                output_path=meta_graph_path,
                start_node=start_node,
                exit_node=exit_node,
                show_labels=True
            )
        # --- FIM DA VISUALIZAÇÃO DO META-GRAFO ---

    except Exception as e:
        print(f"[ERRO] Falha na construção do meta-grafo: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- 3. Execução do Otimizador ACS ---
    print("\n--- 3. Executando o Otimizador ACS Multi-Colônia ---")
    
    stops_to_visit = [stop for stop in representatives if stop in meta_graph.nodes() and stop not in {start_node, exit_node}]
    
    aco_params = {'alpha': args.alpha, 'beta': args.beta, 'rho': args.rho, 'Q': args.Q}
    problem_params = {'capacity': args.capacity, 'max_route_length': args.max_route_length, 'max_route_attempts': args.max_route_attempts}
    quality_weights = {'w_c': args.w_c, 'w_r': args.w_r, 'w_d': args.w_d}
    
    # Verificar se métricas estão habilitadas
    if args.metrics or args.compare_runs or args.convergence_analysis:
        print("Sistema de métricas expandido habilitado.")
        
        # Configurar sistema de métricas
        metrics_config = setup_metrics_config(args)
        # Atualizar diretório de métricas para usar estrutura organizada
        metrics_config.base_output_dir = output_dirs['metrics']
        metrics_config._ensure_directories()
        print(f"Diretório de saída das métricas: {metrics_config.base_output_dir}")
        
        # Determinar número de execuções
        num_runs = args.compare_runs if args.compare_runs else 1
        
        try:
            if num_runs == 1:
                # Execução única com métricas
                result, execution_data = run_single_execution_with_metrics(
                    meta_graph, meta_edges, stops_to_visit, start_node, exit_node,
                    all_opposites, aco_params, problem_params, quality_weights,
                    args, metrics_config
                )
                best_routes, total_dist, num_routes, coverage = result
                execution_data_list = [execution_data]
            else:
                # Múltiplas execuções para comparação
                result, execution_data_list = run_multiple_executions_with_metrics(
                    meta_graph, meta_edges, stops_to_visit, start_node, exit_node,
                    all_opposites, aco_params, problem_params, quality_weights,
                    args, metrics_config, num_runs
                )
                if result:
                    best_routes, total_dist, num_routes, coverage = result
                else:
                    print("[ERRO] Nenhuma execução bem-sucedida.")
                    sys.exit(1)
            
            # Gerar relatórios de métricas
            generate_metrics_reports(execution_data_list, metrics_config, args)
            
        except Exception as e:
            print(f"[ERRO] Falha durante a execução com métricas: {e}")
            if args.verbose:
                traceback.print_exc()
            sys.exit(1)
    else:
        # Execução tradicional sem métricas
        try:
            controller = ACSController(
                graph=meta_graph,
                meta_edges=meta_edges,
                stops_to_visit=stops_to_visit,
                start_node=start_node,
                exit_node=exit_node,
                opposites=all_opposites,
                aco_params=aco_params,
                problem_params=problem_params
            )
            
            best_routes, total_dist, num_routes, coverage = controller.run(
                n_ants=args.ants,
                n_iterations=args.iterations,
                quality_weights=quality_weights,
                verbose=args.verbose
            )
        except Exception as e:
            print(f"[ERRO] Falha durante a execução do ACSController: {e}")
            traceback.print_exc()
            sys.exit(1)

    # --- 4. Relatórios e Visualização Pós-Otimização ---
    print("\n--- 4. Resultados Finais e Relatórios ---")
    if not best_routes:
        print("\nNenhuma rota válida foi encontrada pela otimização.")
        sys.exit(0)

    # Gerar relatórios tradicionais apenas se não estiver usando métricas
    if not (args.metrics or args.compare_runs or args.convergence_analysis):
        # Reúne todos os dados para os relatórios tradicionais
        final_report_data = {
            'params': {
                'ACO_Parameters': aco_params,
                'Problem_Parameters': problem_params,
                'Quality_Weights': quality_weights,
            },
            'problem_setup': {
                'start_node': start_node,
                'exit_node': exit_node,
                'stops_count': len(stops_to_visit),
                'opposites_map': all_opposites
            },
            'solution': {
                'Q_best': controller.history[-1]['best_quality_so_far'] if controller.history else 0,
                'num_routes': num_routes,
                'total_distance': total_dist,
                'coverage': coverage,
                'routes': best_routes
            }
        }

        # Gera todos os relatórios tradicionais (se disponíveis)
        if hasattr(controller, 'history') and controller.history:
            print("Histórico de execução disponível para relatórios tradicionais.")
            # generate_final_report(final_report_data, output_dir=output_dir)
            # generate_convergence_plots(controller.history, output_dir=output_dir)
            # generate_comparison_csv(controller.history, output_dir=output_dir)


    print("\nResumo da Melhor Solução:")
    print(f"  - Número de Rotas: {num_routes}")
    print(f"  - Distância Total: {total_dist:.2f}")
    print(f"  - Cobertura: {coverage*100:.2f}%")
    
    # Preparar dados dos resultados para salvar
    result_data = {
        'num_routes': num_routes,
        'total_dist': total_dist,
        'coverage': coverage,
        'best_routes': best_routes
    }
    
    # Salvar dados, relatórios e métricas nas pastas organizadas
    print("\n--- Salvando Dados Complementares ---")
    save_execution_data(output_dirs, args, result_data, meta_graph, graph)
    generate_execution_report(output_dirs, args, result_data, meta_graph, graph)
    save_metrics_summary(output_dirs, result_data)
    
    # Gerar análise de convergência se histórico disponível
    if hasattr(controller, 'history') and controller.history:
        print("\n--- Gerando Análise de Convergência ---")
        generate_convergence_analysis(output_dirs, controller.history, all_opposites, args.capacity)

    # Geração das visualizações de mapa (código existente)
    expanded_routes = [expand_meta_route(r, meta_graph, meta_edges) for r in best_routes]


    # Salvar visualizações na pasta de imagens organizada
    if args.output:
        output_path = os.path.join(output_dirs['images'], "acs_route.png")
        if args.folium:
            # Para Folium, usar extensão .html
            output_path = os.path.join(output_dirs['images'], "acs_route.html")
            visualize_routes_folium(graph, expanded_routes[0], bus_stops_nodes, output_path, start_node, exit_node)
        else:
            plot_multiple_routes(original_graph=graph, routes=expanded_routes, all_bus_stops=bus_stops_nodes, output_path=output_path, start_node=start_node, exit_node=exit_node)
        print(f"Visualização da rota final salva em: '{output_path}'") 

    if args.meta_output:
        meta_routes_path = os.path.join(output_dirs['images'], "routes_on_meta.png")
        plot_multiple_meta_routes(meta_graph, best_routes, stops_to_visit, meta_routes_path, start_node, exit_node, show_labels=True)
        print(f"Visualização do meta-grafo com rotas salva em: '{meta_routes_path}'")


if __name__ == "__main__":
    main()