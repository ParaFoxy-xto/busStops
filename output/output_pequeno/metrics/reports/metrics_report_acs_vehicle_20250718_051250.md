# Relatório de Métricas - ACS-VEHICLE

**Data de Geração:** 18/07/2025 às 05:12:50  
**Algoritmo:** ACS-VEHICLE  
**Execuções Analisadas:** 1  
**Taxa de Sucesso:** 100.0%

---

## Resumo Executivo

| Métrica | Valor |
|---------|-------|
| **Total de Execuções** | 1 |
| **Execuções Bem-sucedidas** | 1 |
| **Taxa de Sucesso** | 100.0% |
| **Tempo Médio de Execução** | 0.02s |
| **Iterações Médias para Convergência** | 0.0 |
| **Melhor Fitness Geral** | 3590.9438 |


## Métricas de Qualidade das Rotas

### Validação e Restrições

| Métrica | Valor |
|---------|-------|
| **Rotas Válidas** | 10000.0% |
| **Cobertura de Demanda** | 5000.0% |
| **Eficiência de Utilização** | 8000.0% |
| **Violações de Capacidade** | 0 |
| **Violações de Paradas Opostas** | 0 |

### Características das Rotas

| Métrica | Valor |
|---------|-------|
| **Comprimento Médio das Rotas** | 5.00 km |
| **Variância do Comprimento** | 1.00 |
| **Índice de Balanceamento de Carga** | 0.800 |


## Análise de Convergência

### Características de Convergência

| Métrica | Valor |
|---------|-------|
| **Ponto de Convergência** | Iteração 0 |
| **Estabilidade Final** | 0.1000 |
| **Taxa de Melhoria** | 0.0500 |
| **Velocidade de Convergência** | 1.0000 |
| **Total de Iterações** | 0 |
| **Plateau Detectado** | Não detectado |

### Interpretação

- **Convergência**: O algoritmo convergiu na iteração 0 de 0 (0.0% do total)
- **Estabilidade**: Valor de 0.1000 indica baixa estabilidade
- **Eficiência**: Convergência rápida


## Métricas de Transporte Público

### Eficiência Operacional

| Métrica | Valor |
|---------|-------|
| **Tempo Total de Viagem Estimado** | 30.0 min |
| **Transferências Médias** | 1.50 |
| **Cobertura Geográfica** | 75.00 km² |
| **Frequência de Serviço** | 15.00 viagens/hora |

### Eficiência e Sustentabilidade

| Métrica | Valor |
|---------|-------|
| **Eficiência Energética** | 2.50 km/passageiro |
| **Índice de Acessibilidade** | 10.00 paradas/km² |
| **Balanceamento de Carga** | 0.800 |
| **Sobreposição de Rotas** | 2000.0% |

### Avaliação Qualitativa

- **Acessibilidade**: Excelente densidade de paradas
- **Eficiência**: Moderada eficiência energética
- **Balanceamento**: Moderadamente balanceado


## Configuração Utilizada

```json
{
  "algorithm_params": {
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.1,
    "Q": 1.0
  },
  "problem_params": {
    "capacity": 30,
    "max_route_length": 100,
    "max_route_attempts": 10
  },
  "quality_weights": {
    "w_c": 0.2,
    "w_r": 0.7,
    "w_d": 0.1
  },
  "n_ants": 10,
  "n_iterations": 50
}
```


## Conclusões e Recomendações

### Performance Geral

✅ **Excelente confiabilidade** com taxa de sucesso de 100.0%

### Qualidade das Soluções

✅ **Excelente qualidade** com 10000.0% de rotas válidas

### Comportamento de Convergência

✅ **Convergência rápida** na iteração 0 (0.0% do total)

### Recomendações

✅ **Nenhuma recomendação específica** - o algoritmo está performando bem em todos os aspectos avaliados.

### Adequação para Apresentação Acadêmica

Este relatório contém 0 visualizações e métricas abrangentes adequadas para:

- Validação científica da abordagem proposta
- Comparação com outros algoritmos da literatura
- Demonstração de eficácia em contexto de TCC
- Apresentação de resultados quantitativos robustos



---

*Relatório gerado automaticamente pelo Sistema de Métricas Rota_ACO*  
*Timestamp: 18/07/2025 05:12:50*
