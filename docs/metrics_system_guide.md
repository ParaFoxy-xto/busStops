# Guia Completo do Sistema de Métricas Expandido

## Visão Geral

O Sistema de Métricas Expandido do Rota_ACO fornece análise abrangente da performance dos algoritmos ACO aplicados ao VRP (Vehicle Routing Problem). Este sistema foi projetado especificamente para pesquisa acadêmica e validação científica.

## Características Principais

- **Análise de Convergência**: Tracking detalhado da evolução do fitness
- **Métricas de Qualidade**: Validação de rotas e restrições
- **Análise Comparativa**: Comparação estatística entre execuções
- **Visualizações Avançadas**: Gráficos profissionais para apresentação
- **Relatórios Acadêmicos**: Documentação formatada para publicação
- **Métricas de Transporte**: Análises específicas do domínio

## Instalação e Configuração

### Pré-requisitos

```bash
# Instalar dependências básicas
pip install matplotlib seaborn numpy pandas scipy

# Para monitoramento de memória (opcional)
pip install psutil

# Para processamento paralelo (opcional)
pip install joblib
```

### Estrutura de Arquivos

O sistema cria automaticamente a seguinte estrutura:

```
output/metrics/
├── execution_data/     # Dados brutos das execuções
├── reports/           # Relatórios em Markdown
├── visualizations/    # Gráficos e charts
└── comparisons/       # Análises comparativas
```

## Uso Básico

### 1. Execução Simples com Métricas

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics
```

**Saída:**
- Relatório básico em `output/metrics/reports/`
- Gráfico de convergência em `output/metrics/visualizations/`
- Dados de execução em `output/metrics/execution_data/`

### 2. Análise de Convergência Detalhada

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --convergence-analysis \
    --report-output results/convergence_study
```

**Recursos Adicionais:**
- Detecção automática do ponto de convergência
- Análise de estabilidade da solução
- Métricas de diversidade populacional
- Gráficos com intervalos de confiança

### 3. Comparação de Múltiplas Execuções

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --compare-runs 10 \
    --statistical-tests \
    --confidence-level 0.95
```

**Análises Incluídas:**
- Estatísticas descritivas (média, mediana, desvio padrão)
- Testes de significância estatística
- Intervalos de confiança
- Análise de variabilidade entre execuções

## Modos de Operação

### Modo Acadêmico

Otimizado para publicação científica:

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --academic-mode \
    --export-raw-data \
    --visualization-formats png svg pdf
```

**Características:**
- Alta resolução (600 DPI)
- Múltiplos formatos de saída
- Dados brutos incluídos
- Análise estatística completa
- Formatação para publicação

### Modo Rápido

Para testes e desenvolvimento:

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --fast-mode
```

**Otimizações:**
- Visualizações desabilitadas
- Relatórios simplificados
- Processamento paralelo habilitado
- Armazenamento mínimo de dados

## Configuração Personalizada

### Arquivo de Configuração JSON

Crie um arquivo `metrics_config.json`:

```json
{
    "convergence_analysis": {
        "convergence_threshold": 0.001,
        "stability_window": 50,
        "enable_plateau_detection": true
    },
    "quality_metrics": {
        "capacity_limit": 70,
        "enable_opposite_stops_check": true,
        "coverage_threshold": 0.8
    },
    "visualizations": {
        "figure_dpi": 300,
        "output_formats": ["png", "svg"],
        "color_scheme": "academic",
        "include_confidence_intervals": true
    },
    "reports": {
        "include_raw_data": false,
        "language": "pt-BR",
        "academic_formatting": true
    },
    "statistical_analysis": {
        "confidence_level": 0.95,
        "enable_normality_tests": true,
        "enable_variance_tests": true
    }
}
```

Uso:

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics-config metrics_config.json
```

## Métricas Disponíveis

### 1. Métricas de Qualidade das Rotas

- **Porcentagem de Rotas Válidas**: % de rotas que atendem todas as restrições
- **Cobertura de Demanda**: % de paradas atendidas
- **Eficiência de Utilização**: Ocupação média dos veículos
- **Violações de Capacidade**: Número de rotas que excedem 70 passageiros
- **Violações de Paradas Opostas**: Rotas que visitam paradas conflitantes

### 2. Métricas de Convergência

- **Ponto de Convergência**: Iteração onde melhoria < threshold
- **Estabilidade Final**: Desvio padrão das últimas N iterações
- **Taxa de Melhoria**: Velocidade de convergência
- **Detecção de Platô**: Identificação de estagnação
- **Diversidade Populacional**: Variância das soluções

### 3. Métricas Comparativas

- **Estatísticas Agregadas**: Média, mediana, desvio padrão
- **Taxa de Sucesso**: % de execuções bem-sucedidas
- **Tempo de Execução**: Análise de performance temporal
- **Eficiência Relativa**: Comparação entre algoritmos
- **Testes de Significância**: Validação estatística das diferenças

### 4. Métricas de Transporte

- **Tempo Total de Viagem**: Estimativa baseada nas rotas
- **Número de Transferências**: Média de transferências necessárias
- **Cobertura Geográfica**: Área atendida pelo sistema
- **Balanceamento de Carga**: Distribuição entre veículos
- **Eficiência Energética**: Distância por passageiro
- **Índice de Acessibilidade**: Paradas por km²

## Visualizações

### 1. Gráficos de Convergência

- Evolução do fitness ao longo das iterações
- Comparação entre ACS-TIME e ACS-VEHICLE
- Intervalos de confiança
- Detecção de pontos de convergência

### 2. Gráficos Comparativos

- Box plots para comparação de distribuições
- Gráficos de barras para métricas agregadas
- Scatter plots para correlações
- Heatmaps para análise multidimensional

### 3. Histogramas e Distribuições

- Distribuição de fitness final
- Distribuição de tempos de execução
- Análise de normalidade
- Comparação de variabilidade

### 4. Visualizações Específicas de Transporte

- Mapas de calor de utilização de paradas
- Gráficos de cobertura geográfica
- Análise de balanceamento de carga
- Visualização de eficiência energética

## Relatórios

### Relatório de Execução Única

Inclui:
- Resumo da execução
- Métricas de qualidade das rotas
- Análise de convergência
- Métricas específicas de transporte
- Visualizações incorporadas
- Conclusões automáticas

### Relatório Comparativo

Inclui:
- Resumo de todas as execuções
- Análise estatística comparativa
- Testes de significância
- Recomendações baseadas nos resultados
- Visualizações comparativas
- Tabelas de resultados formatadas

## Reprodutibilidade

### Usando Sementes

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --seed 42 \
    --compare-runs 5
```

### Salvando e Carregando Dados

```bash
# Salvar dados de execução
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --save-execution-data \
    --report-output experiment_1

# Carregar dados anteriores para comparação
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --load-previous-data experiment_1/execution_data \
    --compare-runs 3
```

## Processamento Paralelo

Para múltiplas execuções:

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --compare-runs 20 \
    --parallel-executions
```

**Nota**: O processamento paralelo é experimental e pode não funcionar em todos os ambientes.

## Solução de Problemas

### Problemas Comuns

1. **Erro de Memória**
   - Use `--fast-mode` para datasets grandes
   - Reduza o número de iterações armazenadas
   - Monitore uso de memória com `--verbose`

2. **Visualizações Não Geradas**
   - Verifique se matplotlib está instalado
   - Teste diferentes formatos de saída
   - Use `--verbose` para debug

3. **Relatórios Vazios**
   - Verifique se execuções foram bem-sucedidas
   - Confirme permissões de escrita no diretório
   - Use `--verbose` para logs detalhados

4. **Performance Lenta**
   - Use `--fast-mode` para testes
   - Habilite `--parallel-executions`
   - Reduza número de execuções comparativas

### Logs e Debug

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --verbose
```

## Integração com Código Python

### Uso Programático

```python
from rota_aco.metrics.config import MetricsConfig
from rota_aco.metrics.aco_integration import run_aco_with_metrics
from rota_aco.metrics.report_generator import ReportGenerator
from rota_aco.aco.controller import ACSController

# Configurar métricas
config = MetricsConfig()
config.enable_convergence_analysis = True
config.enable_visualizations = True

# Executar com métricas
result, execution_data = run_aco_with_metrics(
    controller_class=ACSController,
    graph=meta_graph,
    meta_edges=meta_edges,
    stops_to_visit=stops_to_visit,
    start_node=start_node,
    exit_node=exit_node,
    opposites=opposites,
    aco_params={'alpha': 1.0, 'beta': 2.0, 'rho': 0.1, 'Q': 1.0},
    problem_params={'capacity': 70},
    quality_weights={'w_c': 10.0, 'w_r': 1.0, 'w_d': 0.5},
    n_ants=20,
    n_iterations=100,
    metrics_config=config
)

# Gerar relatório
report_gen = ReportGenerator(config)
report_path = report_gen.generate_single_execution_report(execution_data)
```

## Referência Completa de Flags

| Flag | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `--metrics` | bool | False | Habilita coleta de métricas |
| `--report-output` | str | "output/metrics" | Diretório de saída |
| `--compare-runs` | int | 1 | Número de execuções para comparação |
| `--convergence-analysis` | bool | False | Análise detalhada de convergência |
| `--metrics-config` | str | None | Arquivo de configuração JSON |
| `--statistical-tests` | bool | False | Habilita testes estatísticos |
| `--confidence-level` | float | 0.95 | Nível de confiança |
| `--export-raw-data` | bool | False | Inclui dados brutos |
| `--visualization-formats` | list | ["png"] | Formatos de saída |
| `--academic-mode` | bool | False | Modo acadêmico |
| `--fast-mode` | bool | False | Modo rápido |
| `--parallel-executions` | bool | False | Execuções paralelas |
| `--seed` | int | None | Semente para reprodutibilidade |
| `--save-execution-data` | bool | True | Salva dados de execução |
| `--load-previous-data` | str | None | Carrega dados anteriores |

## Exemplos para Cenários Específicos

### Para TCC/Dissertação

```bash
# Análise completa para trabalho acadêmico
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --academic-mode \
    --compare-runs 30 \
    --statistical-tests \
    --confidence-level 0.95 \
    --export-raw-data \
    --seed 42 \
    --report-output tcc_results
```

### Para Desenvolvimento/Debug

```bash
# Teste rápido durante desenvolvimento
python -m rota_aco.cli.run \
    --graph graphml/pequeno.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --fast-mode \
    --iterations 5 \
    --ants 5
```

### Para Comparação de Algoritmos

```bash
# Comparar diferentes configurações
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --compare-runs 15 \
    --statistical-tests \
    --alpha 1.5 --beta 2.5 \
    --report-output config_1

python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --compare-runs 15 \
    --statistical-tests \
    --alpha 1.0 --beta 3.0 \
    --report-output config_2
```

## Conclusão

O Sistema de Métricas Expandido fornece uma plataforma completa para análise científica de algoritmos ACO aplicados ao VRP. Com suas funcionalidades abrangentes de análise, visualização e relatórios, é uma ferramenta essencial para pesquisa acadêmica e validação de resultados.

Para mais informações ou suporte, consulte a documentação técnica ou entre em contato com a equipe de desenvolvimento.