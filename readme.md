# Rota_ACO

**Rota_ACO** é um projeto de pesquisa para otimização de rotas de transporte público em ambientes como universidades, combinando **heurísticas baseadas em colônia de formigas (ACO)** com pré-processamentos geográficos e topológicos inteligentes.

---

##  Visão Geral

O sistema é dividido em três etapas principais:

1. **Pré-processamento de dados**: Leitura de grafos via OpenStreetMap e detecção de paradas de ônibus, com identificação de pares de paradas opostas.
2. **Construção do meta-grafo**: Nós representam paradas de ônibus únicas (agrupadas por coordenadas); arestas representam caminhos mínimos que evitam conexões entre paradas opostas.
3. **Otimização via ACO**: Busca rotas eficientes considerando duas funções objetivo — distância percorrida total (ACS-TIME) e número de veículos (ACS-VEHICLE).

---

## Estrutura do Projeto

```
rota_aco/
├── src/
│   └── rota_aco/
│       ├── data/          # preprocessamento e detecção de opostos
│       ├── graph/         # construção de grafos e rotas candidatas
│       ├── aco/           # ACO + controlador multicolônia
│       ├── viz/           # visualizações (matplotlib, folium)
│       └── cli/           # interface de linha de comando (meta, dfs, acs)
├── graphml/               # grafos de entrada (.graphml)
├── tests/                 # testes unitários
├── output/                # Saida dos PNG, Html e plots de convergência
├── requirements.txt
└── README.md
```

---

##  Instalação

```bash
git clone https://github.com/seu-usuario/rota_aco.git
cd rota_aco
pip install -e src
$Env:PYTHONPATH = "$PWD/src"
```

---

##  Funcionalidades

- [x] Carregamento e limpeza de grafos OSM (`load_graph`)
- [x] Detecção de `bus_stop` e paradas opostas por:
  - Coordenadas arredondadas (`pre_process_opposites`)
  - Conexões `bus_access` (`find_opposites_by_access`)
- [x] Construção do meta-grafo sem conexões entre opostos
- [x] Geração de rotas candidatas via DFS (`find_k_shortest_paths`)
- [x] Heurística ACS-TIME: menor distância
- [x] Heurística ACS-VEHICLE: menor número de rotas
- [x] Controlador que mescla as duas abordagens
- [x] Visualizações interativas e estáticas
- [ ] Análise de convergência (em andamento)

---

## Abordagem ACS Multi-Colônia para Geração de Soluções Completas

A implementação atual do ACS (Ant Colony System) vai além da otimização de uma única rota. Em vez disso, o sistema foi projetado para gerar um **conjunto completo de rotas** que, juntas, atendem a toda a demanda de passageiros do sistema, respeitando restrições realistas.

O subcomando `acs` na CLI ativa essa lógica avançada, que opera da seguinte forma:

1.  **Duas Colônias, Dois Objetivos**: O sistema utiliza duas colônias de formigas que trabalham em paralelo, cada uma com um objetivo distinto:
    *   **`ACS-VEHICLE`**: Otimiza para o **menor número de rotas (veículos)**. O critério de desempate é a distância total percorrida.
    *   **`ACS-TIME`**: Otimiza para a **menor distância total percorrida** por todas as rotas. O critério de desempate é o número de veículos.

2.  **Cada Formiga Constrói uma Solução Completa**: Diferente de um ACO clássico onde cada formiga constrói uma única rota (um "caminho"), aqui, cada formiga constrói uma **solução inteira**. Uma solução é um conjunto de rotas (`[rota_1, rota_2, ..., rota_N]`) que, coletivamente, cobrem todos os pontos de demanda.

3.  **Construção da Solução (Intra-Formiga)**: O processo de uma única formiga é o seguinte:
    *   A formiga mantém um registro da demanda de passageiros ainda não atendida em cada parada.
    *   Ela inicia uma nova rota a partir do terminal de início.
    *   Usando feromônios e heurísticas, ela seleciona a próxima parada, sempre validando:
        *   **Capacidade do Veículo**: A rota não pode exceder a capacidade máxima de passageiros (70).
        *   **Paradas Opostas**: Uma rota não pode visitar uma parada e sua oposta direta.
    *   A rota atual é finalizada e retorna ao terminal de saída quando o veículo atinge sua capacidade ou não há mais paradas válidas para visitar.
    *   A formiga repete o processo, criando novas rotas até que toda a demanda de passageiros do grafo seja atendida.
    *   Ao final, a formiga apresenta seu conjunto de rotas como uma solução candidata.

4.  **Seleção da Melhor Solução**: Após todas as formigas de uma colônia gerarem suas soluções, a colônia seleciona a melhor entre elas com base em seu objetivo principal (número de veículos ou distância total). O resultado final do comando `acs` apresenta a melhor solução encontrada por cada uma das duas colônias.

Essa abordagem garante que as rotas geradas não sejam apenas individualmente eficientes, mas também complementares, formando um sistema de transporte coeso e funcional.

---

##  Exemplos de Execução

###  Visualização de Meta-Grafo com Folium (ACO Clássico)

```bash
python -m rota_aco.cli.run meta graphml/pequeno.graphml --precision 6 --start-lat -15.776 --start-lon -47.87161111 --exit-lat -15.760521 --exit-lon -47.8741429 --k-prune 5 --k-paths 3 --top-n 2 --meta-output meta_graph.png --folium --output meta_route.html
```

###  Geração de Rotas Candidatas (DFS)

```bash
python -m rota_aco.cli.run dfs graphml/pequeno.graphml --precision 6 --start-lat -15.776 --start-lon -47.8716 --exit-lat -15.76016667 --exit-lon -47.86705556 --k-prune 5 --k-paths 3 --top-n 2 --show-labels --meta-output dfs_meta.png
```

###  ACO Tradicional (com visualização)

```bash
python -m rota_aco.cli.run meta graphml/pequeno.graphml --precision 6 --start-lat -15.776 --start-lon -47.8716 --exit-lat -15.7601 --exit-lon -47.8670 --k-prune 5 --k-paths 3 --length-percentile 0.75 --top-n 2 --meta-output meta_graph.png --ants 10 --iterations 300 --diversify 10 --pheromone-q 100 --evaporation 0.1  --output final_route.png
```

###  ACS Multi-Colônia (veículos e distância, NOVO!)

```bash
python -m rota_aco.cli.run acs graphml/pequeno.graphml --start-lat -15.776 --start-lon -47.8716 --exit-lat -15.7605 --exit-lon -47.8741 --ants-time 20 --ants-vehicle 20 --iterations 50 --lambda 0.7 --output acs_route.png --verbose
```

```bash
python -m rota_aco.cli.run acs graphml/grafo.graphml --start-lat -15.77600000 --start-lon -47.87161111 --exit-lat -15.76111111 --exit-lon -47.87461111 --ants-time 20 --ants-vehicle 20 --iterations 50 --lambda 0.7 --verbose --output acs_route.png
```
---

##  Coordenadas Reais de Teste

| Entrada | Latitude | Longitude |  
|--------|----------|-----------|  
| FIOCRUZ L3        | -15.77052778 | -47.87008333 |  
| ENTRADA UNB L3    | -15.77600000 | -47.87161111 |  
| CORREIOS L4       | -15.77155556 | -47.86347222 |  
| POSTINHO L3       | -15.76111111 | -47.87461111 |  
| ENTRADA CO        | -15.76563889 | -47.85877778 |  
| FINAL PEQUENO     | -15.76016667 | -47.86705556 |  
| INÍCIO L2         | -15.77952778 | -47.87400000 |  
| FIM L2            | -15.74727778 | -47.88222222 |  

---

##  Teste Rápido

```bash
python tests/test_opposites.py graphml/grafo.graphml
```

---


Desenvolvido como parte do Trabalho de Conclusão de Curso – Universidade de Brasília (UnB)
