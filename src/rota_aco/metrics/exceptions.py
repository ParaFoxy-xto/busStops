"""
Exceções personalizadas para o sistema de métricas expandido.

Este módulo define a hierarquia de exceções específicas para diferentes
tipos de erros que podem ocorrer durante o processamento de métricas.
"""


class MetricsSystemError(Exception):
    """
    Exceção base para todos os erros do sistema de métricas.
    
    Esta é a classe pai de todas as exceções específicas do sistema de métricas,
    permitindo captura genérica de erros relacionados ao módulo.
    """
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message}\nDetalhes: {self.details}"
        return self.message


class DataCollectionError(MetricsSystemError):
    """
    Erro na coleta de dados das execuções ACO.
    
    Levantada quando há problemas na captura, armazenamento ou
    processamento dos dados de execução dos algoritmos ACO.
    """
    
    def __init__(self, message: str, execution_id: str = None, details: str = None):
        self.execution_id = execution_id
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.execution_id:
            return f"{base_msg}\nExecution ID: {self.execution_id}"
        return base_msg


class MetricsCalculationError(MetricsSystemError):
    """
    Erro no cálculo de métricas.
    
    Levantada quando há problemas durante o cálculo de qualquer
    tipo de métrica (qualidade, convergência, comparativa, etc.).
    """
    
    def __init__(self, message: str, metric_type: str = None, details: str = None):
        self.metric_type = metric_type
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.metric_type:
            return f"{base_msg}\nTipo de métrica: {self.metric_type}"
        return base_msg


class VisualizationError(MetricsSystemError):
    """
    Erro na geração de visualizações.
    
    Levantada quando há problemas na criação de gráficos,
    charts ou outras visualizações dos dados de métricas.
    """
    
    def __init__(self, message: str, chart_type: str = None, details: str = None):
        self.chart_type = chart_type
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.chart_type:
            return f"{base_msg}\nTipo de gráfico: {self.chart_type}"
        return base_msg


class ReportGenerationError(MetricsSystemError):
    """
    Erro na geração de relatórios.
    
    Levantada quando há problemas na criação de relatórios
    formatados (markdown, HTML, JSON, etc.).
    """
    
    def __init__(self, message: str, report_format: str = None, details: str = None):
        self.report_format = report_format
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.report_format:
            return f"{base_msg}\nFormato do relatório: {self.report_format}"
        return base_msg


class ConfigurationError(MetricsSystemError):
    """
    Erro na configuração do sistema de métricas.
    
    Levantada quando há problemas na configuração do sistema,
    como valores inválidos ou arquivos de configuração corrompidos.
    """
    
    def __init__(self, message: str, config_parameter: str = None, details: str = None):
        self.config_parameter = config_parameter
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.config_parameter:
            return f"{base_msg}\nParâmetro de configuração: {self.config_parameter}"
        return base_msg


class DataValidationError(MetricsSystemError):
    """
    Erro na validação de dados.
    
    Levantada quando os dados de entrada não atendem aos
    critérios de validação necessários para processamento.
    """
    
    def __init__(self, message: str, data_type: str = None, details: str = None):
        self.data_type = data_type
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.data_type:
            return f"{base_msg}\nTipo de dados: {self.data_type}"
        return base_msg


class FileOperationError(MetricsSystemError):
    """
    Erro em operações de arquivo.
    
    Levantada quando há problemas na leitura, escrita ou
    manipulação de arquivos do sistema de métricas.
    """
    
    def __init__(self, message: str, file_path: str = None, operation: str = None, details: str = None):
        self.file_path = file_path
        self.operation = operation
        super().__init__(message, details)
    
    def __str__(self):
        base_msg = super().__str__()
        if self.file_path:
            base_msg += f"\nArquivo: {self.file_path}"
        if self.operation:
            base_msg += f"\nOperação: {self.operation}"
        return base_msg


# Funções utilitárias para tratamento de exceções

def handle_metrics_error(func):
    """
    Decorator para tratamento padronizado de erros em funções de métricas.
    
    Args:
        func: Função a ser decorada
        
    Returns:
        Função decorada com tratamento de erro
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MetricsSystemError:
            # Re-raise exceções específicas do sistema de métricas
            raise
        except Exception as e:
            # Converte outras exceções em MetricsSystemError
            raise MetricsSystemError(
                f"Erro inesperado na função {func.__name__}",
                details=str(e)
            ) from e
    
    return wrapper


def log_and_raise(exception_class, message: str, logger=None, **kwargs):
    """
    Registra um erro no log e levanta a exceção correspondente.
    
    Args:
        exception_class: Classe da exceção a ser levantada
        message: Mensagem de erro
        logger: Logger para registrar o erro (opcional)
        **kwargs: Argumentos adicionais para a exceção
    """
    if logger:
        logger.error(message)
        if 'details' in kwargs:
            logger.error(f"Detalhes: {kwargs['details']}")
    
    raise exception_class(message, **kwargs)


def safe_execute(func, default_value=None, exception_types=(Exception,)):
    """
    Executa uma função de forma segura, retornando valor padrão em caso de erro.
    
    Args:
        func: Função a ser executada
        default_value: Valor padrão a ser retornado em caso de erro
        exception_types: Tipos de exceção a serem capturadas
        
    Returns:
        Resultado da função ou valor padrão
    """
    try:
        return func()
    except exception_types:
        return default_value