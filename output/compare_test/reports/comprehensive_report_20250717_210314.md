# Relatório Comparativo de Algoritmos ACO

**Data de Geração:** 17/07/2025 às 21:03:14  
**Algoritmos Comparados:** ACS-VEHICLE, ACS-VEHICLE  
**Total de Execuções Analisadas:** 2

Este relatório apresenta uma análise comparativa detalhada entre diferentes configurações 
e algoritmos ACO aplicados ao problema de roteamento de veículos (VRP).

---

## Tabela Resumo Comparativa

| Algoritmo | Execuções | Taxa Sucesso | Fitness Médio | Tempo Médio | Rotas Válidas |
|-----------|-----------|--------------|---------------|-------------|---------------|
| ACS-VEHICLE | 1 | 100.0% | 2209.1190 | 0.00s | 10000.0% |
| ACS-VEHICLE | 1 | 100.0% | 2209.1190 | 0.00s | 10000.0% |

## Análise Detalhada por Algoritmo

### 1. ACS-VEHICLE

**Performance Geral:**
- Taxa de Sucesso: 100.0%
- Tempo Médio: 0.00s
- Melhor Fitness: 2209.1190

**Qualidade das Soluções:**
- Rotas Válidas: 10000.0%
- Cobertura de Demanda: 5000.0%
- Eficiência de Utilização: 8000.0%

**Convergência:**
- Ponto de Convergência: Iteração 0
- Estabilidade Final: 0.1000
- Velocidade: 1.00

---

### 2. ACS-VEHICLE

**Performance Geral:**
- Taxa de Sucesso: 100.0%
- Tempo Médio: 0.00s
- Melhor Fitness: 2209.1190

**Qualidade das Soluções:**
- Rotas Válidas: 10000.0%
- Cobertura de Demanda: 5000.0%
- Eficiência de Utilização: 8000.0%

**Convergência:**
- Ponto de Convergência: Iteração 0
- Estabilidade Final: 0.1000
- Velocidade: 1.00

---

## Conclusões Comparativas

### Melhores Performances por Critério

- **Maior Taxa de Sucesso:** ACS-VEHICLE (100.0%)
- **Melhor Fitness:** ACS-VEHICLE (2209.1190)
- **Mais Rápido:** ACS-VEHICLE (0.00s)
- **Melhor Qualidade:** ACS-VEHICLE (10000.0%)

### Análise de Trade-offs


## Recomendações Baseadas na Comparação

1. **Algoritmo Recomendado:** ACS-VEHICLE apresenta o melhor equilíbrio entre confiabilidade, fitness e qualidade
2. **Para Apresentação Acadêmica:** Utilizar visualizações comparativas para demonstrar trade-offs entre algoritmos
3. **Para Validação Científica:** Executar testes estatísticos para confirmar significância das diferenças observadas

## Metadados do Relatório

### Informações de Geração

- **Timestamp:** 2025-07-17T21:03:14.365797
- **Algoritmos Analisados:** 2
- **Total de Execuções:** 2
- **Total de Visualizações:** 0

### Configurações Utilizadas

```json
{
  "algorithm_params": {
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.1,
    "Q": 1.0
  },
  "problem_params": {
    "capacity": 70,
    "max_route_length": 100,
    "max_route_attempts": 10
  },
  "quality_weights": {
    "w_c": 10.0,
    "w_r": 1.0,
    "w_d": 0.5
  },
  "n_ants": 5,
  "n_iterations": 5
}
```

### Arquivos de Visualização

---

*Relatório gerado automaticamente pelo Sistema de Métricas Rota_ACO*
