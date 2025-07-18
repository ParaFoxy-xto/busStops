# src/rota_aco/viz/convergence_with_invalid_routes.py

"""
Módulo para gerar gráficos de convergência incluindo análise de rotas inválidas.

Este módulo cria visualizações que mostram tanto a convergência da qualidade Q
quanto a evolução das rotas inválidas ao longo das iterações.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import os
from datetime import datetime


class ConvergenceInvalidRoutesVisualizer:
    """
    Visualizador para gráficos de convergência com análise de rotas inválidas.
    """
    
    def __init__(self, output_dir: str = "output/visualizations"):
        """
        Inicializa o visualizador.
        
        Args:
            output_dir: Diretório para salvar as visualizações
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_convergence_with_invalid_routes_plot(
        self,
        controller_history: List[Dict[str, Any]],
        title: str = "Convergência com Análise de Rotas Inválidas",
        save_filename: Optional[str] = None
    ) -> str:
        """
        Gera gráfico combinado de convergência e rotas inválidas.
        
        Args:
            controller_history: Histórico do controlador ACS
            title: Título do gráfico
            save_filename: Nome do arquivo para salvar (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        if not controller_history:
            raise ValueError("Histórico do controlador não pode estar vazio")
        
        # Extrair dados
        data = self._extract_convergence_data(controller_history)
        
        # Criar figura com subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Layout: 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Gráfico principal: Convergência da Qualidade Q
        ax_main = fig.add_subplot(gs[0, :2])
        self._plot_quality_convergence(ax_main, data)
        
        # Gráfico de rotas inválidas
        ax_invalid = fig.add_subplot(gs[0, 2])
        self._plot_invalid_routes(ax_invalid, data)
        
        # Gráfico de taxa de sucesso
        ax_success = fig.add_subplot(gs[1, 0])
        self._plot_success_rate(ax_success, data)
        
        # Gráfico de cobertura
        ax_coverage = fig.add_subplot(gs[1, 1])
        self._plot_coverage_evolution(ax_coverage, data)
        
        # Gráfico de eficiência
        ax_efficiency = fig.add_subplot(gs[1, 2])
        self._plot_efficiency_metrics(ax_efficiency, data)
        
        # Título geral
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Salvar arquivo
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"convergence_invalid_routes_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _extract_convergence_data(self, controller_history: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Extrai dados necessários do histórico do controlador.
        """
        data = {
            'iterations': [],
            'quality_q': [],
            'time_invalid_routes': [],
            'vehicle_invalid_routes': [],
            'time_coverage': [],
            'vehicle_coverage': [],
            'time_routes_count': [],
            'vehicle_routes_count': [],
            'time_distances': [],
            'vehicle_distances': []
        }
        
        for entry in controller_history:
            data['iterations'].append(entry['iteration'])
            data['quality_q'].append(entry['best_quality_so_far'])
            
            # Métricas ACS-TIME
            time_metrics = entry['time_metrics']
            data['time_coverage'].append(time_metrics['coverage'])
            data['time_routes_count'].append(time_metrics['count'])
            data['time_distances'].append(time_metrics['dist'])
            
            # Métricas ACS-VEHICLE
            vehicle_metrics = entry['vehicle_metrics']
            data['vehicle_coverage'].append(vehicle_metrics['coverage'])
            data['vehicle_routes_count'].append(vehicle_metrics['count'])
            data['vehicle_distances'].append(vehicle_metrics['dist'])
            
            # Estimar rotas inválidas baseado na cobertura
            time_invalid = max(0, time_metrics['count'] - int(time_metrics['coverage'] * time_metrics['count']))
            vehicle_invalid = max(0, vehicle_metrics['count'] - int(vehicle_metrics['coverage'] * vehicle_metrics['count']))
            
            data['time_invalid_routes'].append(time_invalid)
            data['vehicle_invalid_routes'].append(vehicle_invalid)
        
        return data
    
    def _plot_quality_convergence(self, ax, data: Dict[str, List]):
        """
        Plota a convergência da Qualidade Q.
        """
        iterations = data['iterations']
        quality_q = data['quality_q']
        
        # Linha principal da qualidade
        ax.plot(iterations, quality_q, 'g-', linewidth=3, marker='D', 
                markersize=6, label='Qualidade Q', alpha=0.8)
        
        # Adicionar linha de tendência
        if len(iterations) > 5:
            z = np.polyfit(iterations, quality_q, 2)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), 'g--', alpha=0.5, linewidth=2, label='Tendência')
        
        # Destacar ponto de convergência (onde a melhoria se estabiliza)
        if len(quality_q) > 10:
            # Encontrar ponto onde a derivada se aproxima de zero
            diffs = np.diff(quality_q)
            convergence_threshold = np.std(diffs) * 0.1
            
            for i, diff in enumerate(diffs):
                if abs(diff) < convergence_threshold and i > len(diffs) * 0.3:
                    convergence_point = i + 1
                    ax.axvline(x=iterations[convergence_point], color='red', linestyle=':', 
                              alpha=0.7, label=f'Convergência (iter {iterations[convergence_point]})')
                    break
        
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Qualidade Q')
        ax.set_title('Convergência da Qualidade Q')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar anotações
        if quality_q:
            final_quality = quality_q[-1]
            initial_quality = quality_q[0]
            improvement = ((final_quality - initial_quality) / abs(initial_quality)) * 100 if initial_quality != 0 else 0
            
            ax.text(0.02, 0.98, f'Melhoria: {improvement:+.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    def _plot_invalid_routes(self, ax, data: Dict[str, List]):
        """
        Plota a evolução das rotas inválidas.
        """
        iterations = data['iterations']
        time_invalid = data['time_invalid_routes']
        vehicle_invalid = data['vehicle_invalid_routes']
        
        # Barras empilhadas para rotas inválidas
        width = 0.8
        ax.bar(iterations, time_invalid, width, label='ACS-TIME', 
               color='lightcoral', alpha=0.7)
        ax.bar(iterations, vehicle_invalid, width, bottom=time_invalid,
               label='ACS-VEHICLE', color='lightblue', alpha=0.7)
        
        # Linha de tendência total
        total_invalid = [t + v for t, v in zip(time_invalid, vehicle_invalid)]
        if len(iterations) > 3:
            z = np.polyfit(iterations, total_invalid, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), 'r--', linewidth=2, 
                   alpha=0.8, label='Tendência Total')
        
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Rotas Inválidas')
        ax.set_title('Evolução das Rotas Inválidas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        if total_invalid:
            avg_invalid = np.mean(total_invalid)
            max_invalid = max(total_invalid)
            
            ax.text(0.02, 0.98, f'Média: {avg_invalid:.1f}\nMáx: {max_invalid}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    def _plot_success_rate(self, ax, data: Dict[str, List]):
        """
        Plota a taxa de sucesso (rotas válidas).
        """
        iterations = data['iterations']
        time_routes = data['time_routes_count']
        vehicle_routes = data['vehicle_routes_count']
        time_invalid = data['time_invalid_routes']
        vehicle_invalid = data['vehicle_invalid_routes']
        
        # Calcular taxa de sucesso
        time_success_rate = [(t - inv) / max(1, t) * 100 for t, inv in zip(time_routes, time_invalid)]
        vehicle_success_rate = [(v - inv) / max(1, v) * 100 for v, inv in zip(vehicle_routes, vehicle_invalid)]
        
        ax.plot(iterations, time_success_rate, 'b-', linewidth=2, 
                marker='o', markersize=4, label='ACS-TIME')
        ax.plot(iterations, vehicle_success_rate, 'r-', linewidth=2, 
                marker='s', markersize=4, label='ACS-VEHICLE')
        
        # Linha de referência (100%)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Válidas')
        
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Taxa de Rotas Válidas (%)')
        ax.set_title('Taxa de Sucesso das Rotas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    def _plot_coverage_evolution(self, ax, data: Dict[str, List]):
        """
        Plota a evolução da cobertura.
        """
        iterations = data['iterations']
        time_coverage = [c * 100 for c in data['time_coverage']]
        vehicle_coverage = [c * 100 for c in data['vehicle_coverage']]
        
        ax.plot(iterations, time_coverage, 'b-', linewidth=2, 
                marker='o', markersize=4, label='ACS-TIME', alpha=0.8)
        ax.plot(iterations, vehicle_coverage, 'r-', linewidth=2, 
                marker='s', markersize=4, label='ACS-VEHICLE', alpha=0.8)
        
        # Área entre as curvas
        ax.fill_between(iterations, time_coverage, vehicle_coverage, 
                       alpha=0.2, color='gray', label='Diferença')
        
        # Linha de referência (100%)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Cobertura Total')
        
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Cobertura (%)')
        ax.set_title('Evolução da Cobertura')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
    
    def _plot_efficiency_metrics(self, ax, data: Dict[str, List]):
        """
        Plota métricas de eficiência.
        """
        iterations = data['iterations']
        time_distances = data['time_distances']
        vehicle_distances = data['vehicle_distances']
        time_routes = data['time_routes_count']
        vehicle_routes = data['vehicle_routes_count']
        
        # Calcular eficiência (distância por rota)
        time_efficiency = [d / max(1, r) for d, r in zip(time_distances, time_routes)]
        vehicle_efficiency = [d / max(1, r) for d, r in zip(vehicle_distances, vehicle_routes)]
        
        ax.plot(iterations, time_efficiency, 'b-', linewidth=2, 
                marker='o', markersize=4, label='ACS-TIME')
        ax.plot(iterations, vehicle_efficiency, 'r-', linewidth=2, 
                marker='s', markersize=4, label='ACS-VEHICLE')
        
        ax.set_xlabel('Iteração')
        ax.set_ylabel('Distância Média por Rota')
        ax.set_title('Eficiência das Rotas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adicionar linha de tendência
        if len(iterations) > 5:
            combined_efficiency = [(t + v) / 2 for t, v in zip(time_efficiency, vehicle_efficiency)]
            z = np.polyfit(iterations, combined_efficiency, 1)
            p = np.poly1d(z)
            ax.plot(iterations, p(iterations), 'g--', alpha=0.6, 
                   linewidth=2, label='Tendência Média')
            ax.legend()
    
    def generate_detailed_invalid_routes_analysis(
        self,
        controller_history: List[Dict[str, Any]],
        save_filename: Optional[str] = None
    ) -> str:
        """
        Gera análise detalhada focada especificamente em rotas inválidas.
        
        Args:
            controller_history: Histórico do controlador
            save_filename: Nome do arquivo para salvar
            
        Returns:
            Caminho do arquivo salvo
        """
        if not controller_history:
            raise ValueError("Histórico do controlador não pode estar vazio")
        
        data = self._extract_convergence_data(controller_history)
        
        # Criar figura focada em rotas inválidas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise Detalhada de Rotas Inválidas', fontsize=16, fontweight='bold')
        
        # 1. Evolução absoluta de rotas inválidas
        iterations = data['iterations']
        time_invalid = data['time_invalid_routes']
        vehicle_invalid = data['vehicle_invalid_routes']
        
        ax1.plot(iterations, time_invalid, 'b-', linewidth=2, marker='o', 
                markersize=5, label='ACS-TIME')
        ax1.plot(iterations, vehicle_invalid, 'r-', linewidth=2, marker='s', 
                markersize=5, label='ACS-VEHICLE')
        ax1.set_xlabel('Iteração')
        ax1.set_ylabel('Número de Rotas Inválidas')
        ax1.set_title('Evolução Absoluta de Rotas Inválidas')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Percentual de rotas inválidas
        time_routes = data['time_routes_count']
        vehicle_routes = data['vehicle_routes_count']
        
        time_invalid_pct = [inv / max(1, total) * 100 for inv, total in zip(time_invalid, time_routes)]
        vehicle_invalid_pct = [inv / max(1, total) * 100 for inv, total in zip(vehicle_invalid, vehicle_routes)]
        
        ax2.plot(iterations, time_invalid_pct, 'b-', linewidth=2, marker='o', 
                markersize=5, label='ACS-TIME')
        ax2.plot(iterations, vehicle_invalid_pct, 'r-', linewidth=2, marker='s', 
                markersize=5, label='ACS-VEHICLE')
        ax2.set_xlabel('Iteração')
        ax2.set_ylabel('Rotas Inválidas (%)')
        ax2.set_title('Percentual de Rotas Inválidas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlação entre qualidade Q e rotas inválidas
        quality_q = data['quality_q']
        total_invalid = [t + v for t, v in zip(time_invalid, vehicle_invalid)]
        
        ax3.scatter(total_invalid, quality_q, alpha=0.6, s=50, c=iterations, cmap='viridis')
        ax3.set_xlabel('Total de Rotas Inválidas')
        ax3.set_ylabel('Qualidade Q')
        ax3.set_title('Correlação: Qualidade Q vs Rotas Inválidas')
        ax3.grid(True, alpha=0.3)
        
        # Adicionar linha de tendência
        if len(total_invalid) > 3:
            z = np.polyfit(total_invalid, quality_q, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(total_invalid), max(total_invalid), 100)
            ax3.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2)
        
        # 4. Distribuição de rotas inválidas
        all_invalid = time_invalid + vehicle_invalid
        ax4.hist(all_invalid, bins=min(20, len(set(all_invalid))), alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax4.set_xlabel('Número de Rotas Inválidas')
        ax4.set_ylabel('Frequência')
        ax4.set_title('Distribuição de Rotas Inválidas')
        ax4.grid(True, alpha=0.3)
        
        # Adicionar estatísticas
        mean_invalid = np.mean(all_invalid)
        std_invalid = np.std(all_invalid)
        ax4.axvline(mean_invalid, color='red', linestyle='--', 
                   label=f'Média: {mean_invalid:.1f}')
        ax4.axvline(mean_invalid + std_invalid, color='orange', linestyle=':', 
                   label=f'+1σ: {mean_invalid + std_invalid:.1f}')
        ax4.legend()
        
        plt.tight_layout()
        
        # Salvar arquivo
        if save_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"detailed_invalid_routes_analysis_{timestamp}.png"
        
        output_path = os.path.join(self.output_dir, save_filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path