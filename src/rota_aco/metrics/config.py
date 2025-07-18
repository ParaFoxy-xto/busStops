"""
Configuração do sistema de métricas expandido.

Este módulo define as configurações padrão e personalizáveis para o sistema
de métricas, incluindo thresholds, formatos de saída e opções de análise.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import os


@dataclass
class MetricsConfig:
    """Configuração principal do sistema de métricas."""
    
    # Análise de convergência
    enable_convergence_analysis: bool = True
    convergence_threshold: float = 0.001  # Threshold para detectar convergência
    stability_window: int = 50  # Janela para calcular estabilidade
    plateau_threshold: float = 0.0001  # Threshold para detectar plateau
    
    # Métricas de qualidade
    enable_quality_metrics: bool = True
    vehicle_capacity: int = 70  # Capacidade máxima do veículo
    
    # Métricas específicas do domínio
    enable_domain_metrics: bool = True
    average_speed_kmh: float = 30.0  # Velocidade média para cálculo de tempo
    transfer_penalty_minutes: float = 5.0  # Penalidade por transferência
    
    # Visualizações
    enable_visualizations: bool = True
    output_formats: List[str] = field(default_factory=lambda: ['png', 'svg'])
    figure_dpi: int = 300
    figure_size: tuple = (12, 8)
    
    # Relatórios
    enable_reports: bool = True
    report_format: str = 'markdown'  # 'markdown', 'html', 'json'
    include_raw_data: bool = False
    
    # Diretórios de saída
    base_output_dir: str = 'output/metrics'
    execution_data_dir: str = 'execution_data'
    reports_dir: str = 'reports'
    visualizations_dir: str = 'visualizations'
    comparisons_dir: str = 'comparisons'
    
    # Configurações de comparação
    enable_statistical_tests: bool = True
    confidence_level: float = 0.95
    min_executions_for_comparison: int = 3
    
    # Configurações de performance
    max_iterations_to_store: int = 10000  # Limite para evitar uso excessivo de memória
    enable_parallel_processing: bool = True
    max_workers: int = 4
    
    # Logging
    enable_detailed_logging: bool = True
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    
    def __post_init__(self):
        """Validação e configuração pós-inicialização."""
        # Validações básicas
        if self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold deve ser positivo")
        if self.stability_window <= 0:
            raise ValueError("stability_window deve ser positivo")
        if self.vehicle_capacity <= 0:
            raise ValueError("vehicle_capacity deve ser positivo")
        if not (0 < self.confidence_level < 1):
            raise ValueError("confidence_level deve estar entre 0 e 1")
        
        # Criar diretórios se não existirem
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Cria diretórios de saída se não existirem."""
        dirs_to_create = [
            self.base_output_dir,
            os.path.join(self.base_output_dir, self.execution_data_dir),
            os.path.join(self.base_output_dir, self.reports_dir),
            os.path.join(self.base_output_dir, self.visualizations_dir),
            os.path.join(self.base_output_dir, self.comparisons_dir)
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_execution_data_path(self) -> str:
        """Retorna o caminho completo para dados de execução."""
        return os.path.join(self.base_output_dir, self.execution_data_dir)
    
    def get_reports_path(self) -> str:
        """Retorna o caminho completo para relatórios."""
        return os.path.join(self.base_output_dir, self.reports_dir)
    
    def get_visualizations_path(self) -> str:
        """Retorna o caminho completo para visualizações."""
        return os.path.join(self.base_output_dir, self.visualizations_dir)
    
    def get_comparisons_path(self) -> str:
        """Retorna o caminho completo para comparações."""
        return os.path.join(self.base_output_dir, self.comparisons_dir)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário."""
        return {
            'convergence_analysis': {
                'enable_convergence_analysis': self.enable_convergence_analysis,
                'convergence_threshold': self.convergence_threshold,
                'stability_window': self.stability_window,
                'plateau_threshold': self.plateau_threshold
            },
            'quality_metrics': {
                'enable_quality_metrics': self.enable_quality_metrics,
                'vehicle_capacity': self.vehicle_capacity
            },
            'domain_metrics': {
                'enable_domain_metrics': self.enable_domain_metrics,
                'average_speed_kmh': self.average_speed_kmh,
                'transfer_penalty_minutes': self.transfer_penalty_minutes
            },
            'visualizations': {
                'enable_visualizations': self.enable_visualizations,
                'output_formats': self.output_formats,
                'figure_dpi': self.figure_dpi,
                'figure_size': self.figure_size
            },
            'reports': {
                'enable_reports': self.enable_reports,
                'report_format': self.report_format,
                'include_raw_data': self.include_raw_data
            },
            'output_paths': {
                'base_output_dir': self.base_output_dir,
                'execution_data_dir': self.execution_data_dir,
                'reports_dir': self.reports_dir,
                'visualizations_dir': self.visualizations_dir,
                'comparisons_dir': self.comparisons_dir
            },
            'comparison': {
                'enable_statistical_tests': self.enable_statistical_tests,
                'confidence_level': self.confidence_level,
                'min_executions_for_comparison': self.min_executions_for_comparison
            },
            'performance': {
                'max_iterations_to_store': self.max_iterations_to_store,
                'enable_parallel_processing': self.enable_parallel_processing,
                'max_workers': self.max_workers
            },
            'logging': {
                'enable_detailed_logging': self.enable_detailed_logging,
                'log_level': self.log_level
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetricsConfig':
        """Cria configuração a partir de dicionário."""
        # Flatten nested dictionary
        flat_config = {}
        
        for section, values in config_dict.items():
            if isinstance(values, dict):
                flat_config.update(values)
            else:
                flat_config[section] = values
        
        return cls(**flat_config)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'MetricsConfig':
        """Carrega configuração de arquivo JSON."""
        import json
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            print(f"Arquivo de configuração não encontrado: {config_path}")
            print("Usando configuração padrão.")
            return cls()
        except json.JSONDecodeError as e:
            print(f"Erro ao ler arquivo de configuração: {e}")
            print("Usando configuração padrão.")
            return cls()
    
    def save_to_file(self, config_path: str):
        """Salva configuração em arquivo JSON."""
        import json
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# Configuração padrão global
DEFAULT_CONFIG = MetricsConfig()


def get_default_config() -> MetricsConfig:
    """Retorna uma cópia da configuração padrão."""
    return MetricsConfig()


def create_academic_config() -> MetricsConfig:
    """Cria configuração otimizada para apresentações acadêmicas."""
    config = MetricsConfig()
    config.figure_dpi = 600  # Alta resolução para publicação
    config.output_formats = ['png', 'svg', 'pdf']
    config.enable_statistical_tests = True
    config.confidence_level = 0.95
    config.include_raw_data = True
    config.enable_detailed_logging = True
    return config


def create_fast_config() -> MetricsConfig:
    """Cria configuração otimizada para execução rápida."""
    config = MetricsConfig()
    config.enable_visualizations = False
    config.enable_reports = False
    config.enable_statistical_tests = False
    config.max_iterations_to_store = 1000
    config.enable_parallel_processing = True
    return config