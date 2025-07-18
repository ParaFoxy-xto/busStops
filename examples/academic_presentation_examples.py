#!/usr/bin/env python3
"""
Exemplos de uso do sistema de métricas para apresentações acadêmicas.

Este script demonstra como usar o sistema de métricas para diferentes
cenários de apresentação acadêmica, incluindo TCC, dissertações,
artigos científicos e apresentações em conferências.
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rota_aco.metrics.config import MetricsConfig, create_academic_config
from rota_aco.metrics.report_generator import ReportGenerator
from rota_aco.metrics.visualization_engine import VisualizationEngine


class AcademicPresentationExamples:
    """Exemplos para diferentes tipos de apresentações acadêmicas."""
    
    def __init__(self, base_output_dir="academic_examples"):
        """Inicializa com diretório base para exemplos."""
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Comando base para execuções (assumindo grafo disponível)
        self.base_cmd = [
            sys.executable, "-m", "rota_aco.cli.run",
            "--graph", "graphml/grafo.graphml",  # Ajustar conforme necessário
            "--start-lat", "-15.7801",
            "--start-lon", "-47.9292",
            "--exit-lat", "-15.7901",
            "--exit-lon", "-47.9392"
        ]
    
    def tcc_undergraduate_example(self):
        """Exemplo para TCC de graduação."""
        print("\n" + "="*60)
        print("EXEMPLO: TCC DE GRADUAÇÃO")
        print("="*60)
        
        output_dir = self.base_output_dir / "tcc_graduacao"
        output_dir.mkdir(exist_ok=True)
        
        print("""
CENÁRIO: Trabalho de Conclusão de Curso
OBJETIVO: Demonstrar eficácia do algoritmo ACO para VRP
REQUISITOS: 
- Análise estatística básica
- Gráficos claros e profissionais
- Relatório em português
- Comparação com baseline
        """)
        
        # Configuração específica para TCC
        config_file = output_dir / "tcc_config.json"
        tcc_config = {
            "convergence_analysis": {
                "convergence_threshold": 0.01,
                "stability_window": 30,
                "enable_plateau_detection": True
            },
            "visualizations": {
                "figure_dpi": 300,
                "output_formats": ["png", "pdf"],
                "color_scheme": "academic",
                "include_confidence_intervals": True,
                "font_size": 12
            },
            "reports": {
                "language": "pt-BR",
                "include_raw_data": False,
                "academic_formatting": True,
                "include_methodology": True
            },
            "statistical_analysis": {
                "confidence_level": 0.95,
                "enable_normality_tests": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(tcc_config, f, indent=2, ensure_ascii=False)
        
        # Comando para TCC
        tcc_cmd = self.base_cmd + [
            "--metrics-config", str(config_file),
            "--compare-runs", "10",
            "--statistical-tests",
            "--seed", "42",  # Para reprodutibilidade
            "--report-output", str(output_dir / "results"),
            "--iterations", "50",
            "--ants", "20"
        ]
        
        print("Comando sugerido:")
        print(" ".join(tcc_cmd))
        
        print("""
SAÍDAS ESPERADAS:
- Relatório comparativo em português
- Gráficos de convergência em alta resolução
- Análise estatística com intervalos de confiança
- Tabelas formatadas para inclusão no texto
- Dados reprodutíveis (semente fixa)

DICAS PARA TCC:
1. Use --seed para garantir reprodutibilidade
2. Execute pelo menos 10 runs para análise estatística
3. Documente todos os parâmetros utilizados
4. Inclua análise de convergência para mostrar estabilidade
        """)
    
    def masters_dissertation_example(self):
        """Exemplo para dissertação de mestrado."""
        print("\n" + "="*60)
        print("EXEMPLO: DISSERTAÇÃO DE MESTRADO")
        print("="*60)
        
        output_dir = self.base_output_dir / "dissertacao_mestrado"
        output_dir.mkdir(exist_ok=True)
        
        print("""
CENÁRIO: Dissertação de Mestrado
OBJETIVO: Proposta de melhoria no algoritmo ACO
REQUISITOS:
- Análise estatística rigorosa
- Comparação com múltiplos algoritmos
- Visualizações para publicação
- Análise de significância estatística
        """)
        
        # Configuração para mestrado
        config_file = output_dir / "mestrado_config.json"
        mestrado_config = {
            "convergence_analysis": {
                "convergence_threshold": 0.005,
                "stability_window": 50,
                "enable_plateau_detection": True,
                "track_diversity": True
            },
            "quality_metrics": {
                "enable_detailed_validation": True,
                "track_constraint_violations": True
            },
            "visualizations": {
                "figure_dpi": 600,
                "output_formats": ["png", "svg", "pdf"],
                "color_scheme": "publication",
                "include_confidence_intervals": True,
                "include_statistical_annotations": True,
                "font_size": 10
            },
            "reports": {
                "language": "pt-BR",
                "include_raw_data": True,
                "academic_formatting": True,
                "include_methodology": True,
                "include_statistical_tests": True
            },
            "statistical_analysis": {
                "confidence_level": 0.95,
                "enable_normality_tests": True,
                "enable_variance_tests": True,
                "enable_effect_size": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(mestrado_config, f, indent=2, ensure_ascii=False)
        
        # Múltiplas configurações para comparação
        configs = [
            ("aco_tradicional", {"alpha": 1.0, "beta": 2.0, "rho": 0.1}),
            ("aco_otimizado", {"alpha": 1.5, "beta": 2.5, "rho": 0.05}),
            ("aco_proposto", {"alpha": 1.2, "beta": 3.0, "rho": 0.08})
        ]
        
        print("Comandos sugeridos para comparação:")
        for config_name, params in configs:
            cmd = self.base_cmd + [
                "--metrics-config", str(config_file),
                "--compare-runs", "30",
                "--statistical-tests",
                "--confidence-level", "0.95",
                "--seed", "42",
                "--report-output", str(output_dir / f"results_{config_name}"),
                "--alpha", str(params["alpha"]),
                "--beta", str(params["beta"]),
                "--rho", str(params["rho"]),
                "--iterations", "100",
                "--ants", "30"
            ]
            print(f"\n# {config_name.upper()}:")
            print(" ".join(cmd))
        
        print("""
SAÍDAS ESPERADAS:
- Relatórios comparativos detalhados
- Gráficos em múltiplos formatos (PNG, SVG, PDF)
- Análise estatística com testes de significância
- Dados brutos para análises adicionais
- Visualizações prontas para publicação

DICAS PARA DISSERTAÇÃO:
1. Execute pelo menos 30 runs por configuração
2. Use testes estatísticos rigorosos
3. Documente metodologia experimental
4. Compare com algoritmos do estado da arte
5. Analise significância prática além da estatística
        """)
    
    def conference_paper_example(self):
        """Exemplo para artigo de conferência."""
        print("\n" + "="*60)
        print("EXEMPLO: ARTIGO DE CONFERÊNCIA")
        print("="*60)
        
        output_dir = self.base_output_dir / "artigo_conferencia"
        output_dir.mkdir(exist_ok=True)
        
        print("""
CENÁRIO: Artigo para Conferência Internacional
OBJETIVO: Publicação de resultados inovadores
REQUISITOS:
- Visualizações de alta qualidade
- Análise estatística rigorosa
- Formato internacional (inglês)
- Comparação com benchmarks
        """)
        
        # Configuração para conferência
        config_file = output_dir / "conference_config.json"
        conference_config = {
            "convergence_analysis": {
                "convergence_threshold": 0.001,
                "stability_window": 100,
                "enable_plateau_detection": True,
                "track_diversity": True
            },
            "visualizations": {
                "figure_dpi": 600,
                "output_formats": ["pdf", "eps"],  # Formatos para publicação
                "color_scheme": "publication",
                "include_confidence_intervals": True,
                "include_statistical_annotations": True,
                "font_size": 8,  # Menor para publicação
                "use_latex_fonts": True
            },
            "reports": {
                "language": "en-US",
                "include_raw_data": True,
                "academic_formatting": True,
                "include_methodology": True,
                "include_statistical_tests": True,
                "citation_style": "ieee"
            },
            "statistical_analysis": {
                "confidence_level": 0.99,  # Mais rigoroso
                "enable_normality_tests": True,
                "enable_variance_tests": True,
                "enable_effect_size": True,
                "enable_power_analysis": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(conference_config, f, indent=2, ensure_ascii=False)
        
        # Comando para conferência
        conference_cmd = self.base_cmd + [
            "--metrics-config", str(config_file),
            "--compare-runs", "50",  # Mais execuções para rigor
            "--statistical-tests",
            "--confidence-level", "0.99",
            "--export-raw-data",
            "--seed", "12345",
            "--report-output", str(output_dir / "conference_results"),
            "--iterations", "200",
            "--ants", "50",
            "--parallel-executions"  # Para acelerar
        ]
        
        print("Comando sugerido:")
        print(" ".join(conference_cmd))
        
        print("""
SAÍDAS ESPERADAS:
- Gráficos em formato EPS/PDF para publicação
- Relatório em inglês com formatação acadêmica
- Análise estatística rigorosa (99% confiança)
- Dados brutos para revisores
- Visualizações com fontes LaTeX

DICAS PARA CONFERÊNCIA:
1. Use formatos vetoriais (PDF, EPS) para figuras
2. Aplique testes estatísticos rigorosos
3. Inclua análise de tamanho do efeito
4. Documente todos os parâmetros experimentais
5. Prepare dados brutos para revisores
        """)
    
    def phd_thesis_example(self):
        """Exemplo para tese de doutorado."""
        print("\n" + "="*60)
        print("EXEMPLO: TESE DE DOUTORADO")
        print("="*60)
        
        output_dir = self.base_output_dir / "tese_doutorado"
        output_dir.mkdir(exist_ok=True)
        
        print("""
CENÁRIO: Tese de Doutorado
OBJETIVO: Contribuição científica significativa
REQUISITOS:
- Análise experimental abrangente
- Múltiplos datasets e cenários
- Análise estatística avançada
- Documentação completa da metodologia
        """)
        
        # Configuração para doutorado
        config_file = output_dir / "phd_config.json"
        phd_config = {
            "convergence_analysis": {
                "convergence_threshold": 0.0001,
                "stability_window": 200,
                "enable_plateau_detection": True,
                "track_diversity": True,
                "enable_advanced_metrics": True
            },
            "quality_metrics": {
                "enable_detailed_validation": True,
                "track_constraint_violations": True,
                "enable_domain_specific_metrics": True
            },
            "visualizations": {
                "figure_dpi": 600,
                "output_formats": ["png", "svg", "pdf", "eps"],
                "color_scheme": "publication",
                "include_confidence_intervals": True,
                "include_statistical_annotations": True,
                "font_size": 10,
                "use_latex_fonts": True,
                "enable_3d_plots": True
            },
            "reports": {
                "language": "pt-BR",
                "include_raw_data": True,
                "academic_formatting": True,
                "include_methodology": True,
                "include_statistical_tests": True,
                "include_appendices": True,
                "detailed_analysis": True
            },
            "statistical_analysis": {
                "confidence_level": 0.99,
                "enable_normality_tests": True,
                "enable_variance_tests": True,
                "enable_effect_size": True,
                "enable_power_analysis": True,
                "enable_bayesian_analysis": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(phd_config, f, indent=2, ensure_ascii=False)
        
        # Múltiplos experimentos para tese
        experiments = [
            ("small_instances", {"graph": "graphml/pequeno.graphml", "runs": 100}),
            ("medium_instances", {"graph": "graphml/grafo.graphml", "runs": 50}),
            ("large_instances", {"graph": "graphml/longo.graphml", "runs": 30})
        ]
        
        print("Comandos sugeridos para experimentos abrangentes:")
        for exp_name, exp_config in experiments:
            cmd = [
                sys.executable, "-m", "rota_aco.cli.run",
                "--graph", exp_config["graph"],
                "--start-lat", "-15.7801",
                "--start-lon", "-47.9292",
                "--exit-lat", "-15.7901",
                "--exit-lon", "-47.9392",
                "--metrics-config", str(config_file),
                "--compare-runs", str(exp_config["runs"]),
                "--statistical-tests",
                "--confidence-level", "0.99",
                "--export-raw-data",
                "--seed", "54321",
                "--report-output", str(output_dir / f"results_{exp_name}"),
                "--iterations", "300",
                "--ants", "100",
                "--parallel-executions"
            ]
            print(f"\n# {exp_name.upper()}:")
            print(" ".join(cmd))
        
        print("""
SAÍDAS ESPERADAS:
- Análise experimental abrangente
- Relatórios detalhados por experimento
- Visualizações em múltiplos formatos
- Análise estatística avançada
- Documentação completa da metodologia
- Dados brutos para reprodutibilidade

DICAS PARA TESE:
1. Execute experimentos em múltiplos datasets
2. Use análise estatística avançada (Bayesiana)
3. Documente completamente a metodologia
4. Inclua análise de sensibilidade de parâmetros
5. Prepare dados para reprodutibilidade completa
        """)
    
    def presentation_slides_example(self):
        """Exemplo para slides de apresentação."""
        print("\n" + "="*60)
        print("EXEMPLO: SLIDES DE APRESENTAÇÃO")
        print("="*60)
        
        output_dir = self.base_output_dir / "slides_apresentacao"
        output_dir.mkdir(exist_ok=True)
        
        print("""
CENÁRIO: Apresentação Oral (Defesa/Conferência)
OBJETIVO: Visualizações claras e impactantes
REQUISITOS:
- Gráficos grandes e legíveis
- Cores contrastantes
- Foco nos resultados principais
- Formato adequado para projeção
        """)
        
        # Configuração para slides
        config_file = output_dir / "slides_config.json"
        slides_config = {
            "visualizations": {
                "figure_dpi": 150,  # Menor DPI para slides
                "output_formats": ["png"],
                "color_scheme": "presentation",
                "font_size": 16,  # Fonte grande para projeção
                "line_width": 3,  # Linhas mais grossas
                "marker_size": 8,
                "use_bold_fonts": True,
                "high_contrast": True
            },
            "reports": {
                "language": "pt-BR",
                "include_raw_data": False,
                "summary_only": True,
                "highlight_key_results": True
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(slides_config, f, indent=2, ensure_ascii=False)
        
        # Comando para slides
        slides_cmd = self.base_cmd + [
            "--metrics-config", str(config_file),
            "--compare-runs", "5",  # Menos execuções para demo
            "--report-output", str(output_dir / "slides_results"),
            "--iterations", "30",
            "--ants", "15"
        ]
        
        print("Comando sugerido:")
        print(" ".join(slides_cmd))
        
        print("""
SAÍDAS ESPERADAS:
- Gráficos otimizados para projeção
- Fontes grandes e legíveis
- Cores de alto contraste
- Foco nos resultados principais

DICAS PARA SLIDES:
1. Use fontes grandes (16pt+)
2. Prefira cores de alto contraste
3. Simplifique gráficos complexos
4. Destaque apenas resultados principais
5. Teste visibilidade em projetor
        """)
    
    def run_all_examples(self):
        """Executa todos os exemplos."""
        print("EXEMPLOS DE USO PARA APRESENTAÇÕES ACADÊMICAS")
        print("=" * 60)
        print("""
Este script demonstra como configurar o sistema de métricas
para diferentes tipos de apresentações acadêmicas.

Cada exemplo inclui:
- Configuração específica para o cenário
- Comandos CLI sugeridos
- Explicação das saídas esperadas
- Dicas específicas para o contexto
        """)
        
        self.tcc_undergraduate_example()
        self.masters_dissertation_example()
        self.conference_paper_example()
        self.phd_thesis_example()
        self.presentation_slides_example()
        
        print("\n" + "="*60)
        print("RESUMO DOS EXEMPLOS")
        print("="*60)
        print(f"""
Todos os arquivos de configuração foram criados em:
{self.base_output_dir.absolute()}

Para usar qualquer exemplo:
1. Ajuste o caminho do grafo GraphML
2. Ajuste as coordenadas de início e fim
3. Execute o comando sugerido
4. Analise os resultados gerados

ESTRUTURA DE ARQUIVOS CRIADA:
{self.base_output_dir}/
├── tcc_graduacao/
│   ├── tcc_config.json
│   └── results/
├── dissertacao_mestrado/
│   ├── mestrado_config.json
│   └── results_*/
├── artigo_conferencia/
│   ├── conference_config.json
│   └── conference_results/
├── tese_doutorado/
│   ├── phd_config.json
│   └── results_*/
└── slides_apresentacao/
    ├── slides_config.json
    └── slides_results/
        """)


def main():
    """Função principal."""
    examples = AcademicPresentationExamples()
    examples.run_all_examples()


if __name__ == "__main__":
    main()