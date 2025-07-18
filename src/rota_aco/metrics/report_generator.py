"""
Gerador de relatórios para o sistema de métricas expandido.

Este módulo implementa a geração de relatórios formatados em markdown
com tabelas, análises estatísticas e referências a visualizações.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .data_models import (
    MetricsReport, ExecutionSummary, RouteQualityMetrics,
    ConvergenceMetrics, ComparativeMetrics, DomainMetrics
)
from .config import MetricsConfig
from .exceptions import ReportGenerationError


@dataclass
class ReportSection:
    """Representa uma seção do relatório."""
    title: str
    content: str
    level: int = 2  # Nível do cabeçalho (1-6)


class ReportGenerator:
    """
    Gerador de relatórios em formato markdown para métricas ACO.
    
    Gera relatórios estruturados com:
    - Resumo executivo
    - Tabelas formatadas de métricas
    - Análise estatística
    - Referências a visualizações
    - Conclusões automatizadas
    """
    
    def __init__(self, config: MetricsConfig):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            config: Configuração do sistema de métricas
        """
        self.config = config
        self._ensure_output_directory()
    
    def _ensure_output_directory(self):
        """Garante que o diretório de relatórios existe."""
        reports_path = self.config.get_reports_path()
        os.makedirs(reports_path, exist_ok=True)
    
    def generate_report(self, metrics_report: MetricsReport, 
                       output_filename: Optional[str] = None) -> str:
        """
        Gera relatório completo em markdown.
        
        Args:
            metrics_report: Dados das métricas para o relatório
            output_filename: Nome do arquivo de saída (opcional)
            
        Returns:
            Caminho para o arquivo de relatório gerado
            
        Raises:
            ReportGenerationError: Se houver erro na geração do relatório
        """
        try:
            # Gerar seções do relatório
            sections = self._generate_report_sections(metrics_report)
            
            # Montar conteúdo completo
            content = self._assemble_report_content(sections, metrics_report)
            
            # Determinar nome do arquivo
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                algorithm = metrics_report.execution_summary.algorithm_type.lower().replace('-', '_')
                output_filename = f"metrics_report_{algorithm}_{timestamp}.md"
            
            # Salvar arquivo
            output_path = os.path.join(self.config.get_reports_path(), output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return output_path
            
        except Exception as e:
            raise ReportGenerationError(f"Erro ao gerar relatório: {str(e)}") from e
    
    def _generate_report_sections(self, metrics_report: MetricsReport) -> List[ReportSection]:
        """Gera todas as seções do relatório."""
        sections = []
        
        # Cabeçalho e resumo executivo
        sections.append(self._generate_header_section(metrics_report))
        sections.append(self._generate_executive_summary_section(metrics_report.execution_summary))
        
        # Métricas de qualidade
        sections.append(self._generate_quality_metrics_section(metrics_report.quality_metrics))
        
        # Análise de convergência
        sections.append(self._generate_convergence_analysis_section(metrics_report.convergence_analysis))
        
        # Análise comparativa (se disponível)
        if metrics_report.comparative_analysis:
            sections.append(self._generate_comparative_analysis_section(metrics_report.comparative_analysis))
        
        # Métricas específicas do domínio
        sections.append(self._generate_domain_metrics_section(metrics_report.domain_metrics))
        
        # Visualizações
        if metrics_report.visualizations:
            sections.append(self._generate_visualizations_section(metrics_report.visualizations))
        
        # Configuração utilizada
        sections.append(self._generate_configuration_section(metrics_report.config_used))
        
        # Conclusões automatizadas
        sections.append(self._generate_conclusions_section(metrics_report))
        
        return sections
    
    def _generate_header_section(self, metrics_report: MetricsReport) -> ReportSection:
        """Gera seção de cabeçalho do relatório."""
        algorithm = metrics_report.execution_summary.algorithm_type
        timestamp = metrics_report.timestamp.strftime("%d/%m/%Y às %H:%M:%S")
        
        content = f"""# Relatório de Métricas - {algorithm}

**Data de Geração:** {timestamp}  
**Algoritmo:** {algorithm}  
**Execuções Analisadas:** {metrics_report.execution_summary.total_executions}  
**Taxa de Sucesso:** {metrics_report.execution_summary.success_rate:.1%}

---
"""
        return ReportSection("Header", content, 1)
    
    def _generate_executive_summary_section(self, summary: ExecutionSummary) -> ReportSection:
        """Gera seção de resumo executivo."""
        content = f"""## Resumo Executivo

| Métrica | Valor |
|---------|-------|
| **Total de Execuções** | {summary.total_executions} |
| **Execuções Bem-sucedidas** | {summary.successful_executions} |
| **Taxa de Sucesso** | {summary.success_rate:.1%} |
| **Tempo Médio de Execução** | {summary.avg_execution_time:.2f}s |
| **Iterações Médias para Convergência** | {summary.avg_iterations_to_convergence:.1f} |
| **Melhor Fitness Geral** | {summary.best_overall_fitness:.4f} |

"""
        return ReportSection("Resumo Executivo", content)
    
    def _generate_quality_metrics_section(self, quality: RouteQualityMetrics) -> ReportSection:
        """Gera seção de métricas de qualidade das rotas."""
        content = f"""## Métricas de Qualidade das Rotas

### Validação e Restrições

| Métrica | Valor |
|---------|-------|
| **Rotas Válidas** | {quality.valid_routes_percentage:.1%} |
| **Cobertura de Demanda** | {quality.demand_coverage_percentage:.1%} |
| **Eficiência de Utilização** | {quality.vehicle_utilization_efficiency:.1%} |
| **Violações de Capacidade** | {quality.capacity_violations} |
| **Violações de Paradas Opostas** | {quality.opposite_stops_violations} |

### Características das Rotas

| Métrica | Valor |
|---------|-------|
| **Comprimento Médio das Rotas** | {quality.average_route_length:.2f} km |
| **Variância do Comprimento** | {quality.route_length_variance:.2f} |
| **Índice de Balanceamento de Carga** | {quality.load_balancing_index:.3f} |

"""
        return ReportSection("Métricas de Qualidade", content)
    
    def _generate_convergence_analysis_section(self, convergence: ConvergenceMetrics) -> ReportSection:
        """Gera seção de análise de convergência."""
        plateau_status = "Detectado" if convergence.plateau_detection else "Não detectado"
        
        content = f"""## Análise de Convergência

### Características de Convergência

| Métrica | Valor |
|---------|-------|
| **Ponto de Convergência** | Iteração {convergence.convergence_point} |
| **Estabilidade Final** | {convergence.final_stability:.4f} |
| **Taxa de Melhoria** | {convergence.improvement_rate:.4f} |
| **Velocidade de Convergência** | {convergence.convergence_speed:.4f} |
| **Total de Iterações** | {convergence.total_iterations} |
| **Plateau Detectado** | {plateau_status} |

### Interpretação

- **Convergência**: O algoritmo convergiu na iteração {convergence.convergence_point} de {convergence.total_iterations} ({(convergence.convergence_point/convergence.total_iterations if convergence.total_iterations > 0 else 0):.1%} do total)
- **Estabilidade**: Valor de {convergence.final_stability:.4f} indica {'alta' if convergence.final_stability < 0.01 else 'moderada' if convergence.final_stability < 0.1 else 'baixa'} estabilidade
- **Eficiência**: {'Convergência rápida' if convergence.convergence_speed > 0.5 else 'Convergência moderada' if convergence.convergence_speed > 0.2 else 'Convergência lenta'}

"""
        return ReportSection("Análise de Convergência", content)
    
    def _generate_comparative_analysis_section(self, comparative: ComparativeMetrics) -> ReportSection:
        """Gera seção de análise comparativa."""
        content = f"""## Análise Comparativa

### Estatísticas Agregadas

| Métrica | Valor |
|---------|-------|
| **Execuções Totais** | {comparative.total_executions} |
| **Execuções Bem-sucedidas** | {comparative.successful_executions} |
| **Taxa de Sucesso** | {comparative.success_rate:.1%} |
| **Tempo Médio ± Desvio** | {comparative.avg_execution_time:.2f} ± {comparative.std_execution_time:.2f}s |
| **Fitness Médio ± Desvio** | {comparative.avg_best_fitness:.4f} ± {comparative.std_best_fitness:.4f} |
| **Fitness Mediano** | {comparative.median_best_fitness:.4f} |
| **Iterações Médias para Convergência** | {comparative.avg_convergence_iterations:.1f} |
| **Eficiência Relativa** | {comparative.relative_efficiency:.2f} |

### Análise Estatística

- **Consistência**: Desvio padrão do fitness de {comparative.std_best_fitness:.4f} indica {'alta' if comparative.std_best_fitness < 0.1 else 'moderada' if comparative.std_best_fitness < 0.5 else 'baixa'} consistência
- **Confiabilidade**: Taxa de sucesso de {comparative.success_rate:.1%} demonstra {'excelente' if comparative.success_rate > 0.9 else 'boa' if comparative.success_rate > 0.7 else 'moderada'} confiabilidade
- **Performance**: Tempo médio de {comparative.avg_execution_time:.2f}s com variação de ±{comparative.std_execution_time:.2f}s

"""
        return ReportSection("Análise Comparativa", content)
    
    def _generate_domain_metrics_section(self, domain: DomainMetrics) -> ReportSection:
        """Gera seção de métricas específicas do domínio."""
        content = f"""## Métricas de Transporte Público

### Eficiência Operacional

| Métrica | Valor |
|---------|-------|
| **Tempo Total de Viagem Estimado** | {domain.estimated_travel_time:.1f} min |
| **Transferências Médias** | {domain.average_transfers:.2f} |
| **Cobertura Geográfica** | {domain.geographic_coverage:.2f} km² |
| **Frequência de Serviço** | {domain.service_frequency:.2f} viagens/hora |

### Eficiência e Sustentabilidade

| Métrica | Valor |
|---------|-------|
| **Eficiência Energética** | {domain.energy_efficiency:.2f} km/passageiro |
| **Índice de Acessibilidade** | {domain.accessibility_index:.2f} paradas/km² |
| **Balanceamento de Carga** | {domain.load_balancing_index:.3f} |
| **Sobreposição de Rotas** | {domain.route_overlap_percentage:.1%} |

### Avaliação Qualitativa

- **Acessibilidade**: {'Excelente' if domain.accessibility_index > 5 else 'Boa' if domain.accessibility_index > 2 else 'Moderada'} densidade de paradas
- **Eficiência**: {'Alta' if domain.energy_efficiency < 2 else 'Moderada' if domain.energy_efficiency < 4 else 'Baixa'} eficiência energética
- **Balanceamento**: {'Bem balanceado' if domain.load_balancing_index > 0.8 else 'Moderadamente balanceado' if domain.load_balancing_index > 0.6 else 'Desbalanceado'}

"""
        return ReportSection("Métricas de Transporte", content)
    
    def _generate_visualizations_section(self, visualizations: List[str]) -> ReportSection:
        """Gera seção de referências às visualizações."""
        content = "## Visualizações Geradas\n\n"
        
        for i, viz_path in enumerate(visualizations, 1):
            filename = os.path.basename(viz_path)
            content += f"{i}. **{filename}**\n"
            content += f"   - Caminho: `{viz_path}`\n\n"
        
        content += """
### Como Interpretar as Visualizações

- **Gráficos de Convergência**: Mostram a evolução do fitness ao longo das iterações
- **Gráficos Comparativos**: Permitem comparação side-by-side entre diferentes configurações
- **Heatmaps**: Visualizam a utilização de paradas e distribuição espacial
- **Histogramas**: Mostram a distribuição de valores de fitness e outras métricas

"""
        return ReportSection("Visualizações", content)
    
    def _generate_configuration_section(self, config: Dict[str, Any]) -> ReportSection:
        """Gera seção de configuração utilizada."""
        content = "## Configuração Utilizada\n\n"
        
        if config:
            content += "```json\n"
            import json
            content += json.dumps(config, indent=2, ensure_ascii=False)
            content += "\n```\n\n"
        else:
            content += "*Configuração padrão utilizada.*\n\n"
        
        return ReportSection("Configuração", content)
    
    def _generate_conclusions_section(self, metrics_report: MetricsReport) -> ReportSection:
        """Gera seção de conclusões automatizadas."""
        summary = metrics_report.execution_summary
        quality = metrics_report.quality_metrics
        convergence = metrics_report.convergence_analysis
        domain = metrics_report.domain_metrics
        
        content = "## Conclusões e Recomendações\n\n"
        
        # Análise de performance geral
        content += "### Performance Geral\n\n"
        if summary.success_rate > 0.9:
            content += f"✅ **Excelente confiabilidade** com taxa de sucesso de {summary.success_rate:.1%}\n\n"
        elif summary.success_rate > 0.7:
            content += f"⚠️ **Boa confiabilidade** com taxa de sucesso de {summary.success_rate:.1%}, mas há margem para melhoria\n\n"
        else:
            content += f"❌ **Confiabilidade baixa** com taxa de sucesso de {summary.success_rate:.1%} - requer investigação\n\n"
        
        # Análise de qualidade das rotas
        content += "### Qualidade das Soluções\n\n"
        if quality.valid_routes_percentage > 0.95:
            content += f"✅ **Excelente qualidade** com {quality.valid_routes_percentage:.1%} de rotas válidas\n\n"
        elif quality.valid_routes_percentage > 0.8:
            content += f"⚠️ **Boa qualidade** com {quality.valid_routes_percentage:.1%} de rotas válidas\n\n"
        else:
            content += f"❌ **Qualidade baixa** com apenas {quality.valid_routes_percentage:.1%} de rotas válidas\n\n"
        
        # Análise de convergência
        content += "### Comportamento de Convergência\n\n"
        convergence_ratio = convergence.convergence_point / convergence.total_iterations if convergence.total_iterations > 0 else 0
        if convergence_ratio < 0.5:
            content += f"✅ **Convergência rápida** na iteração {convergence.convergence_point} ({convergence_ratio:.1%} do total)\n\n"
        elif convergence_ratio < 0.8:
            content += f"⚠️ **Convergência moderada** na iteração {convergence.convergence_point} ({convergence_ratio:.1%} do total)\n\n"
        else:
            content += f"❌ **Convergência lenta** na iteração {convergence.convergence_point} ({convergence_ratio:.1%} do total)\n\n"
        
        # Recomendações
        content += "### Recomendações\n\n"
        
        recommendations = []
        
        if summary.success_rate < 0.8:
            recommendations.append("Investigar causas de falha nas execuções e ajustar parâmetros")
        
        if quality.valid_routes_percentage < 0.9:
            recommendations.append("Revisar restrições de capacidade e paradas opostas")
        
        if convergence_ratio > 0.7:
            recommendations.append("Considerar ajustar parâmetros para acelerar convergência")
        
        if domain.load_balancing_index < 0.7:
            recommendations.append("Melhorar balanceamento de carga entre veículos")
        
        if quality.vehicle_utilization_efficiency < 0.8:
            recommendations.append("Otimizar utilização de veículos para reduzir custos")
        
        if not recommendations:
            content += "✅ **Nenhuma recomendação específica** - o algoritmo está performando bem em todos os aspectos avaliados.\n\n"
        else:
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
            content += "\n"
        
        # Adequação para uso acadêmico
        content += "### Adequação para Apresentação Acadêmica\n\n"
        content += f"Este relatório contém {len(metrics_report.visualizations)} visualizações "
        content += "e métricas abrangentes adequadas para:\n\n"
        content += "- Validação científica da abordagem proposta\n"
        content += "- Comparação com outros algoritmos da literatura\n"
        content += "- Demonstração de eficácia em contexto de TCC\n"
        content += "- Apresentação de resultados quantitativos robustos\n\n"
        
        return ReportSection("Conclusões", content)
    
    def _assemble_report_content(self, sections: List[ReportSection], 
                                metrics_report: MetricsReport) -> str:
        """Monta o conteúdo completo do relatório."""
        content_parts = []
        
        # Adicionar cada seção
        for section in sections:
            content_parts.append(section.content)
        
        # Adicionar rodapé
        footer = f"""
---

*Relatório gerado automaticamente pelo Sistema de Métricas Rota_ACO*  
*Timestamp: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}*
"""
        content_parts.append(footer)
        
        return "\n".join(content_parts)
    
    def generate_summary_table(self, metrics_reports: List[MetricsReport]) -> str:
        """
        Gera tabela resumo comparativa para múltiplos relatórios.
        
        Args:
            metrics_reports: Lista de relatórios de métricas
            
        Returns:
            String com tabela markdown formatada
        """
        if not metrics_reports:
            return "Nenhum relatório fornecido para comparação."
        
        content = "## Tabela Resumo Comparativa\n\n"
        content += "| Algoritmo | Execuções | Taxa Sucesso | Fitness Médio | Tempo Médio | Rotas Válidas |\n"
        content += "|-----------|-----------|--------------|---------------|-------------|---------------|\n"
        
        for report in metrics_reports:
            summary = report.execution_summary
            quality = report.quality_metrics
            
            content += f"| {summary.algorithm_type} | "
            content += f"{summary.total_executions} | "
            content += f"{summary.success_rate:.1%} | "
            content += f"{summary.best_overall_fitness:.4f} | "
            content += f"{summary.avg_execution_time:.2f}s | "
            content += f"{quality.valid_routes_percentage:.1%} |\n"
        
        return content
    
    def generate_comprehensive_report(self, metrics_reports: List[MetricsReport], 
                                    output_filename: Optional[str] = None) -> str:
        """
        Gera relatório abrangente comparando múltiplas execuções.
        
        Args:
            metrics_reports: Lista de relatórios de métricas para comparação
            output_filename: Nome do arquivo de saída (opcional)
            
        Returns:
            Caminho para o arquivo de relatório gerado
            
        Raises:
            ReportGenerationError: Se houver erro na geração do relatório
        """
        try:
            if not metrics_reports:
                raise ValueError("Lista de relatórios não pode estar vazia")
            
            # Cabeçalho do relatório comparativo
            content = self._generate_comparative_header(metrics_reports)
            
            # Tabela resumo
            content += self.generate_summary_table(metrics_reports)
            content += "\n"
            
            # Análise detalhada de cada algoritmo
            content += "## Análise Detalhada por Algoritmo\n\n"
            
            for i, report in enumerate(metrics_reports, 1):
                content += f"### {i}. {report.execution_summary.algorithm_type}\n\n"
                
                # Métricas principais
                content += self._generate_algorithm_summary(report)
                
                # Visualizações específicas
                if report.visualizations:
                    content += f"**Visualizações:** {len(report.visualizations)} gráficos gerados\n\n"
                
                content += "---\n\n"
            
            # Conclusões comparativas
            content += self._generate_comparative_conclusions(metrics_reports)
            
            # Recomendações baseadas na comparação
            content += self._generate_comparative_recommendations(metrics_reports)
            
            # Metadados do relatório
            content += self._generate_report_metadata(metrics_reports)
            
            # Determinar nome do arquivo
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"comprehensive_report_{timestamp}.md"
            
            # Salvar arquivo
            output_path = os.path.join(self.config.get_reports_path(), output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return output_path
            
        except Exception as e:
            raise ReportGenerationError(f"Erro ao gerar relatório abrangente: {str(e)}") from e
    
    def _generate_comparative_header(self, metrics_reports: List[MetricsReport]) -> str:
        """Gera cabeçalho para relatório comparativo."""
        algorithms = [report.execution_summary.algorithm_type for report in metrics_reports]
        timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        
        content = f"""# Relatório Comparativo de Algoritmos ACO

**Data de Geração:** {timestamp}  
**Algoritmos Comparados:** {', '.join(algorithms)}  
**Total de Execuções Analisadas:** {sum(r.execution_summary.total_executions for r in metrics_reports)}

Este relatório apresenta uma análise comparativa detalhada entre diferentes configurações 
e algoritmos ACO aplicados ao problema de roteamento de veículos (VRP).

---

"""
        return content
    
    def _generate_algorithm_summary(self, report: MetricsReport) -> str:
        """Gera resumo de um algoritmo específico."""
        summary = report.execution_summary
        quality = report.quality_metrics
        convergence = report.convergence_analysis
        
        content = f"""**Performance Geral:**
- Taxa de Sucesso: {summary.success_rate:.1%}
- Tempo Médio: {summary.avg_execution_time:.2f}s
- Melhor Fitness: {summary.best_overall_fitness:.4f}

**Qualidade das Soluções:**
- Rotas Válidas: {quality.valid_routes_percentage:.1%}
- Cobertura de Demanda: {quality.demand_coverage_percentage:.1%}
- Eficiência de Utilização: {quality.vehicle_utilization_efficiency:.1%}

**Convergência:**
- Ponto de Convergência: Iteração {convergence.convergence_point}
- Estabilidade Final: {convergence.final_stability:.4f}
- Velocidade: {convergence.convergence_speed:.2f}

"""
        return content
    
    def _generate_comparative_conclusions(self, metrics_reports: List[MetricsReport]) -> str:
        """Gera conclusões comparativas entre algoritmos."""
        content = "## Conclusões Comparativas\n\n"
        
        # Encontrar melhor algoritmo por critério
        best_success_rate = max(metrics_reports, key=lambda r: r.execution_summary.success_rate)
        best_fitness = max(metrics_reports, key=lambda r: r.execution_summary.best_overall_fitness)
        fastest = min(metrics_reports, key=lambda r: r.execution_summary.avg_execution_time)
        best_quality = max(metrics_reports, key=lambda r: r.quality_metrics.valid_routes_percentage)
        
        content += "### Melhores Performances por Critério\n\n"
        content += f"- **Maior Taxa de Sucesso:** {best_success_rate.execution_summary.algorithm_type} "
        content += f"({best_success_rate.execution_summary.success_rate:.1%})\n"
        content += f"- **Melhor Fitness:** {best_fitness.execution_summary.algorithm_type} "
        content += f"({best_fitness.execution_summary.best_overall_fitness:.4f})\n"
        content += f"- **Mais Rápido:** {fastest.execution_summary.algorithm_type} "
        content += f"({fastest.execution_summary.avg_execution_time:.2f}s)\n"
        content += f"- **Qualidade Q:** {best_quality.execution_summary.algorithm_type} "
        content += f"({best_quality.quality_metrics.valid_routes_percentage:.1%})\n\n"
        
        # Análise de trade-offs
        content += "### Análise de Trade-offs\n\n"
        
        if best_fitness.execution_summary.algorithm_type != fastest.execution_summary.algorithm_type:
            content += f"- **Trade-off Qualidade vs Velocidade:** {best_fitness.execution_summary.algorithm_type} "
            content += f"oferece melhor fitness mas {fastest.execution_summary.algorithm_type} é mais rápido\n"
        
        if best_success_rate.execution_summary.algorithm_type != best_fitness.execution_summary.algorithm_type:
            content += f"- **Trade-off Confiabilidade vs Performance:** {best_success_rate.execution_summary.algorithm_type} "
            content += f"é mais confiável mas {best_fitness.execution_summary.algorithm_type} tem melhor fitness\n"
        
        content += "\n"
        return content
    
    def _generate_comparative_recommendations(self, metrics_reports: List[MetricsReport]) -> str:
        """Gera recomendações baseadas na comparação."""
        content = "## Recomendações Baseadas na Comparação\n\n"
        
        # Calcular médias para comparação
        avg_success_rate = sum(r.execution_summary.success_rate for r in metrics_reports) / len(metrics_reports)
        avg_execution_time = sum(r.execution_summary.avg_execution_time for r in metrics_reports) / len(metrics_reports)
        avg_quality = sum(r.quality_metrics.valid_routes_percentage for r in metrics_reports) / len(metrics_reports)
        
        recommendations = []
        
        # Recomendações baseadas em performance
        best_overall = max(metrics_reports, key=lambda r: (
            r.execution_summary.success_rate * 0.3 +
            r.execution_summary.best_overall_fitness * 0.3 +
            r.quality_metrics.valid_routes_percentage * 0.4
        ))
        
        recommendations.append(f"**Algoritmo Recomendado:** {best_overall.execution_summary.algorithm_type} "
                             f"apresenta o melhor equilíbrio entre confiabilidade, fitness e qualidade")
        
        # Recomendações específicas por cenário
        if any(r.execution_summary.success_rate < 0.8 for r in metrics_reports):
            low_success = [r for r in metrics_reports if r.execution_summary.success_rate < 0.8]
            for report in low_success:
                recommendations.append(f"**{report.execution_summary.algorithm_type}:** "
                                     f"Investigar baixa taxa de sucesso ({report.execution_summary.success_rate:.1%})")
        
        if any(r.execution_summary.avg_execution_time > avg_execution_time * 1.5 for r in metrics_reports):
            slow_algorithms = [r for r in metrics_reports if r.execution_summary.avg_execution_time > avg_execution_time * 1.5]
            for report in slow_algorithms:
                recommendations.append(f"**{report.execution_summary.algorithm_type}:** "
                                     f"Otimizar performance (tempo atual: {report.execution_summary.avg_execution_time:.2f}s)")
        
        # Recomendações para uso acadêmico
        recommendations.append("**Para Apresentação Acadêmica:** Utilizar visualizações comparativas "
                             "para demonstrar trade-offs entre algoritmos")
        
        recommendations.append("**Para Validação Científica:** Executar testes estatísticos "
                             "para confirmar significância das diferenças observadas")
        
        for i, rec in enumerate(recommendations, 1):
            content += f"{i}. {rec}\n"
        
        content += "\n"
        return content
    
    def _generate_report_metadata(self, metrics_reports: List[MetricsReport]) -> str:
        """Gera metadados do relatório."""
        content = "## Metadados do Relatório\n\n"
        
        content += "### Informações de Geração\n\n"
        content += f"- **Timestamp:** {datetime.now().isoformat()}\n"
        content += f"- **Algoritmos Analisados:** {len(metrics_reports)}\n"
        content += f"- **Total de Execuções:** {sum(r.execution_summary.total_executions for r in metrics_reports)}\n"
        content += f"- **Total de Visualizações:** {sum(len(r.visualizations) for r in metrics_reports)}\n"
        
        content += "\n### Configurações Utilizadas\n\n"
        
        # Mostrar configurações únicas
        all_configs = [report.config_used for report in metrics_reports if report.config_used]
        if all_configs:
            content += "```json\n"
            import json
            content += json.dumps(all_configs[0], indent=2, ensure_ascii=False)
            content += "\n```\n\n"
        
        content += "### Arquivos de Visualização\n\n"
        
        for i, report in enumerate(metrics_reports, 1):
            if report.visualizations:
                content += f"**{report.execution_summary.algorithm_type}:**\n"
                for viz in report.visualizations:
                    content += f"- `{viz}`\n"
                content += "\n"
        
        content += "---\n\n"
        content += "*Relatório gerado automaticamente pelo Sistema de Métricas Rota_ACO*\n"
        
        return content
    
    def export_to_json(self, metrics_report: MetricsReport, 
                      output_filename: Optional[str] = None) -> str:
        """
        Exporta relatório de métricas para formato JSON.
        
        Args:
            metrics_report: Dados das métricas para exportar
            output_filename: Nome do arquivo de saída (opcional)
            
        Returns:
            Caminho para o arquivo JSON gerado
        """
        try:
            import json
            
            # Converter para dicionário
            data = metrics_report.to_dict()
            
            # Determinar nome do arquivo
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                algorithm = metrics_report.execution_summary.algorithm_type.lower().replace('-', '_')
                output_filename = f"metrics_data_{algorithm}_{timestamp}.json"
            
            # Salvar arquivo
            output_path = os.path.join(self.config.get_reports_path(), output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            return output_path
            
        except Exception as e:
            raise ReportGenerationError(f"Erro ao exportar para JSON: {str(e)}") from e