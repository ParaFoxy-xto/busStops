# Guia de Comparação de Algoritmos

Este documento explica como usar as implementações simples de ACO e força bruta para comparação com o sistema principal.

## Algoritmos Implementados

### 1. ACO Simples (`SimpleACO`)
- **Arquivo**: `src/rota_aco/aco/simple_aco.py`
- **Características**:
  - Implementação básica do Ant Colony Optimization
  - Foco na otimização de rotas simples
  - Parâmetros configuráveis (α, β, ρ, Q)
  - Garante visita a todos os pontos obrigatórios

### 2. Força Bruta (`BruteForceOptimizer`)
- **Arquivo**: `src/rota_aco/aco/brute_force.py`
- **Características**:
  - Testa todas as permutações possíveis
  - Encontra a solução ótima garantida
  - Limitações de tempo e número de permutações
  - Adequado apenas para problemas pequenos (< 8 pontos)

### 3. Algoritmo Guloso (`GreedyOptimizer`)
- **Arquivo**: `src/rota_aco/aco/brute_force.py`
- **Características**:
  - Sempre escolhe o próximo ponto mais próximo
  - Muito rápido
  - Boa aproximação inicial
  - Não garante solução ótima

## Scripts de Demonstração

### 1. Demonstração Simples
```bash
python examples/simple_comparison_demo.py
```

**Características**:
- Usa grafo pequeno e simples
- Compara os três algoritmos
- Mostra análise detalhada dos resultados
- Ideal para entender o comportamento básico

### 2. Demonstração CLI com Métricas
```bash
python examples/cli_metrics_demo.py [opções]
```

**Opções principais**:
- `--algorithms aco brute_force greedy`: Escolher algoritmos
- `--graph-type simple|complex`: Tipo de grafo
- `--stops N`: Número de pontos para visitar
- `--show-routes`: Mostrar rotas encontradas
- `--verbose`: Informações detalhadas
- `--output arquivo.json`: Salvar resultados

**Exemplos de uso**:
```bash
# Comparar todos os algoritmos com grafo complexo
python examples/cli_metrics_demo.py --graph-type complex --stops 5

# Apenas ACO e guloso, com rotas detalhadas
python examples/cli_metrics_demo.py -a aco greedy --show-routes --verbose

# Salvar resultados em arquivo
python examples/cli_metrics_demo.py --output results.json
```

### 3. Comparação com Grafos Reais
```bash
python examples/algorithm_comparison.py --graph graphml/pequeno.graphml
```

## Parâmetros dos Algoritmos

### ACO Simples
- **alpha (α)**: Importância do feromônio (padrão: 1.0)
- **beta (β)**: Importância da heurística (padrão: 2.0)
- **rho (ρ)**: Taxa de evaporação (padrão: 0.1)
- **q_param (Q)**: Fator de reforço (padrão: 1.0)
- **n_ants**: Número de formigas por iteração (padrão: 10)
- **n_iterations**: Número de iterações (padrão: 50)

### Força Bruta
- **max_permutations**: Limite de permutações (padrão: 10000)
- **max_time_seconds**: Limite de tempo em segundos

## Métricas de Comparação

### Qualidade da Solução
- **Distância Total**: Soma dos pesos das arestas na rota
- **Cobertura**: Percentual de pontos obrigatórios visitados
- **Gap do Ótimo**: Diferença percentual da solução ótima

### Performance
- **Tempo de Execução**: Tempo total em segundos
- **Eficiência**: Razão distância/tempo
- **Taxa de Sucesso**: Percentual de rotas válidas encontradas

### Específicas do ACO
- **Convergência**: Iteração onde encontrou a melhor solução
- **Melhoria**: Número de iterações com melhoria

## Interpretação dos Resultados

### Quando Usar Cada Algoritmo

#### Força Bruta
- **Use quando**: Problema pequeno (≤ 7 pontos), precisa da solução ótima
- **Não use quando**: Muitos pontos, tempo limitado
- **Vantagens**: Solução ótima garantida
- **Desvantagens**: Crescimento exponencial do tempo

#### Algoritmo Guloso
- **Use quando**: Precisa de solução rápida, primeira aproximação
- **Não use quando**: Qualidade da solução é crítica
- **Vantagens**: Muito rápido, simples
- **Desvantagens**: Pode ficar preso em ótimos locais

#### ACO Simples
- **Use quando**: Equilíbrio entre qualidade e tempo
- **Não use quando**: Problema muito simples ou muito complexo
- **Vantagens**: Boa qualidade, flexível
- **Desvantagens**: Mais lento que guloso, parâmetros para ajustar

### Análise de Gap
- **0-5%**: Excelente (muito próximo do ótimo)
- **5-15%**: Bom (próximo do ótimo)
- **15-30%**: Aceitável
- **>30%**: Pode melhorar (ajustar parâmetros)

## Exemplos de Uso Programático

### ACO Simples
```python
from rota_aco.aco.simple_aco import SimpleACO

aco = SimpleACO(
    graph=graph,
    meta_edges=meta_edges,
    stops_to_visit=stops_to_visit,
    start_node=start_node,
    exit_node=exit_node,
    alpha=1.0,
    beta=2.0,
    rho=0.1
)

route, distance, stats = aco.run(n_ants=10, n_iterations=50)
```

### Força Bruta
```python
from rota_aco.aco.brute_force import BruteForceOptimizer

bf = BruteForceOptimizer(
    graph=graph,
    meta_edges=meta_edges,
    stops_to_visit=stops_to_visit,
    start_node=start_node,
    exit_node=exit_node
)

# Com limite de tempo
route, distance, stats = bf.run_limited(max_time_seconds=30.0)

# Sem limite (apenas para problemas pequenos)
route, distance, stats = bf.run()
```

### Algoritmo Guloso
```python
from rota_aco.aco.brute_force import GreedyOptimizer

greedy = GreedyOptimizer(
    graph=graph,
    meta_edges=meta_edges,
    stops_to_visit=stops_to_visit,
    start_node=start_node,
    exit_node=exit_node
)

route, distance, stats = greedy.run()
```

## Dicas de Otimização

### Para ACO
1. **Ajustar α e β**: Maior α favorece feromônio, maior β favorece heurística
2. **Taxa de evaporação (ρ)**: 0.1-0.3 geralmente funciona bem
3. **Número de formigas**: 10-20 para problemas médios
4. **Iterações**: 50-100 dependendo da complexidade

### Para Força Bruta
1. **Limite de tempo**: Use sempre para problemas > 6 pontos
2. **Pré-processamento**: Remova nós inacessíveis antes
3. **Paralelização**: Considere dividir permutações

### Geral
1. **Pré-análise**: Use guloso para estimar complexidade
2. **Validação**: Compare com força bruta quando possível
3. **Monitoramento**: Acompanhe métricas de convergência

## Limitações Conhecidas

### ACO Simples
- Não implementa otimizações avançadas (elitismo, MAX-MIN)
- Parâmetros fixos durante execução
- Não considera múltiplas rotas simultaneamente

### Força Bruta
- Crescimento fatorial O(n!)
- Uso intensivo de memória para problemas grandes
- Não aproveita estrutura do grafo

### Algoritmo Guloso
- Pode ficar preso em ótimos locais
- Não considera impacto global das decisões
- Sensível à ordem dos pontos

## Próximos Passos

Para melhorar as implementações:

1. **ACO Avançado**: Implementar MAX-MIN AS, elitismo
2. **Heurísticas**: Adicionar heurísticas específicas do domínio
3. **Paralelização**: Implementar versões paralelas
4. **Hibridização**: Combinar algoritmos (ex: ACO + busca local)
5. **Adaptação**: Parâmetros adaptativos durante execução