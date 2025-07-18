# Sistema de Métricas Expandido - Rota_ACO

Este módulo fornece análise detalhada da performance dos algoritmos ACO aplicados ao VRP (Vehicle Routing Problem), incluindo métricas de qualidade, análise de convergência e relatórios acadêmicos.

## 🚀 Funcionalidades Principais

### 📊 Métricas de Qualidade
- **% de rotas válidas**: Porcentagem de rotas que atendem todas as restrições
- **Cobertura de demanda**: Porcentagem de paradas atendidas
- **Eficiência de utilização**: Ocupação média dos veículos
- **Violações de restrições**: Contagem de violações de capacidade e paradas opostas

### 📈 Análise de Convergência
- **Evolução do fitness**: Tracking da melhor solução ao longo das iterações
- **Detecção de convergência**: Identificação automática do ponto de convergência
- **Análise de estabilidade**: Medição da variância nas últimas iterações
- **Detecção de plateau**: Identificação de estagnação no algoritmo

### 🔄 Métricas Comparativas
- **Estatísticas agregadas**: Média, mediana, desvio padrão de múltiplas execuções
- **Taxa de sucesso**: Porcentagem de execuções que encontraram soluções válidas
- **Eficiência relativa**: Comparação entre diferentes configurações
- **Análise temporal**: Comparação de tempos de execução

### 🚌 Métricas Específicas de Transporte
- **Tempo de viagem estimado**: Cálculo baseado em velocidade média
- **Número de transferências**: Análise de conectividade entre rotas
- **Cobertura geográfica**: Área atendida pelo sistema
- **Índice de acessibilidade**: Densidade de paradas por área

## 📁 Estrutura do Módulo

```
src/rota_aco/metrics/
├── __init__.py              # Exports principais
├── data_models.py           # Estruturas de dados
├── config.py               # Configurações do sistema
├── exceptions.py           # Exceções personalizadas
├── examples.py             # Exemplos de uso
└── README.md              # Esta documentação
```

## 🛠️ Instalação e Configuração

### Dependências
O sistema de métricas utiliza as mesmas dependências do projeto principal:
- `matplotlib` - Para visualizações
- `numpy` - Para cálculos estatísticos (será adicionado)
- `pandas` - Para manipulação de dados (será adicionado)

### Configuração Básica

```python
from src.rota_aco.metrics import MetricsConfig

# Configuração padrão
config = MetricsConfig()

# Configuração para apresentação acadêmica
from src.rota_aco.metrics.config import create_academic_config
config = create_academic_config()

# Configuração para execução rápida (sem visualizações)
from src.rota_aco.metrics.config import create_fast_config
config = create_fast_config()
```

## 📖 Exemplos de Uso

### Exemplo Básico

```python
from src.rota_aco.metrics import (
    ExecutionData, Route, Solution, MetricsConfig
)

# Criar configuração
config = MetricsConfig()

# Criar dados de exemplo
route = Route(
    stops=[1, 2, 3, 4],
    distances=[100.0, 150.0, 200.0],
    passenger_load=[20, 25, 15],
    total_distance=450.0,
    total_passengers=60,
    is_valid=True
)

solution = Solution(
    routes=[route],
    total_vehicles=1,
    total_distance=450.0,
    total_passengers_served=60,
    fitness_time=450.0,
    fitness_vehicle=1450.0,
    is_feasible=True
)

# Criar dados de execução
execution_data = ExecutionData(
    algorithm_type="ACS-TIME",
    config={"ants": 20, "iterations": 100},
    routes=[route],
    final_solution=solution,
    execution_time=45.5,
    success=True
)
```

### Executar Exemplo Completo

```bash
python -m src.rota_aco.metrics.examples
```

## 🔧 Configurações Disponíveis

### Análise de Convergência
- `convergence_threshold`: Threshold para detectar convergência (padrão: 0.001)
- `stability_window`: Janela para calcular estabilidade (padrão: 50)
- `plateau_threshold`: Threshold para detectar plateau (padrão: 0.0001)

### Métricas de Qualidade
- `vehicle_capacity`: Capacidade máxima do veículo (padrão: 70)

### Visualizações
- `output_formats`: Formatos de saída ['png', 'svg'] (padrão)
- `figure_dpi`: Resolução das imagens (padrão: 300)
- `figure_size`: Tamanho das figuras (padrão: (12, 8))

### Diretórios de Saída
- `base_output_dir`: Diretório base (padrão: 'output/metrics')
- `execution_data_dir`: Dados de execução (padrão: 'execution_data')
- `reports_dir`: Relatórios (padrão: 'reports')
- `visualizations_dir`: Visualizações (padrão: 'visualizations')

## 📊 Estrutura de Dados

### Route (Rota Individual)
```python
@dataclass
class Route:
    stops: List[int]                    # IDs das paradas
    distances: List[float]              # Distâncias entre paradas
    passenger_load: List[int]           # Carga de passageiros
    total_distance: float               # Distância total
    total_passengers: int               # Total de passageiros
    is_valid: bool                      # Se atende restrições
    capacity_violations: int            # Violações de capacidade
    opposite_stops_violations: int      # Violações de paradas opostas
```

### Solution (Conjunto de Rotas)
```python
@dataclass
class Solution:
    routes: List[Route]                 # Lista de rotas
    total_vehicles: int                 # Número de veículos
    total_distance: float               # Distância total
    total_passengers_served: int        # Passageiros atendidos
    fitness_time: float                 # Fitness baseado em tempo
    fitness_vehicle: float              # Fitness baseado em veículos
    is_feasible: bool                   # Se é uma solução viável
```

### ExecutionData (Dados de Execução Completa)
```python
@dataclass
class ExecutionData:
    execution_id: str                   # ID único da execução
    algorithm_type: str                 # 'ACS-TIME' ou 'ACS-VEHICLE'
    config: Dict[str, Any]              # Configuração utilizada
    routes: List[Route]                 # Rotas da melhor solução
    iterations_data: List[IterationData] # Dados de cada iteração
    execution_time: float               # Tempo total de execução
    final_solution: Solution            # Melhor solução encontrada
    success: bool                       # Se a execução foi bem-sucedida
```

## 🎯 Próximos Passos

Este módulo de infraestrutura está completo e testado. Os próximos componentes a serem implementados são:

1. **DataCollector** - Para capturar dados das execuções ACO
2. **MetricsEngine** - Para calcular as métricas
3. **VisualizationEngine** - Para gerar gráficos
4. **ReportGenerator** - Para criar relatórios formatados

## 🧪 Testes

Execute os testes da infraestrutura:

```bash
python -m pytest tests/test_metrics_infrastructure.py -v
```

## 📝 Contribuição

Para adicionar novas métricas ou funcionalidades:

1. Defina novos modelos de dados em `data_models.py`
2. Adicione configurações relevantes em `config.py`
3. Crie exceções específicas em `exceptions.py` se necessário
4. Adicione testes em `tests/test_metrics_*.py`
5. Documente exemplos de uso

## 🔍 Debugging

Para debug detalhado, configure o logging:

```python
config = MetricsConfig(
    enable_detailed_logging=True,
    log_level='DEBUG'
)
```

---

**Desenvolvido para o TCC de Ciência da Computação - UnB**  
*Sistema de Otimização de Rotas com Ant Colony Optimization*