# Guia de Solução de Problemas - Sistema de Métricas

## Problemas Comuns e Soluções

### 1. Problemas de Instalação e Dependências

#### Erro: ModuleNotFoundError

**Problema:**
```
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'seaborn'
ModuleNotFoundError: No module named 'scipy'
```

**Solução:**
```bash
# Instalar dependências básicas
pip install matplotlib seaborn numpy pandas scipy

# Para funcionalidades avançadas
pip install psutil joblib

# Verificar instalação
python -c "import matplotlib, seaborn, numpy, pandas, scipy; print('Todas as dependências instaladas!')"
```

#### Erro: Versão incompatível do Python

**Problema:**
```
SyntaxError: invalid syntax (uso de f-strings ou type hints)
```

**Solução:**
- O sistema requer Python 3.7+
- Verifique sua versão: `python --version`
- Atualize se necessário

### 2. Problemas de Execução

#### Erro: Arquivo GraphML não encontrado

**Problema:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'graphml/grafo.graphml'
```

**Solução:**
```bash
# Verificar se arquivo existe
ls -la graphml/

# Usar caminho absoluto se necessário
python -m rota_aco.cli.run --graph /caminho/completo/para/grafo.graphml ...

# Verificar formato do arquivo
file graphml/grafo.graphml
```

#### Erro: Coordenadas inválidas

**Problema:**
```
ValueError: No nodes found near the specified coordinates
```

**Solução:**
1. Verificar se coordenadas estão dentro da área do grafo
2. Usar coordenadas com precisão adequada (5-6 casas decimais)
3. Verificar se grafo contém nós com atributos de latitude/longitude

```bash
# Exemplo com coordenadas válidas para Brasília
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics
```

#### Erro: Nenhuma parada de ônibus encontrada

**Problema:**
```
[ERRO] Nenhuma parada de ônibus ('bus_stop=true') encontrada no grafo.
```

**Solução:**
1. Verificar se o grafo contém nós marcados como paradas de ônibus
2. Verificar atributo correto no GraphML:

```xml
<node id="123">
    <data key="bus_stop">true</data>
    <data key="lat">-15.7801</data>
    <data key="lon">-47.9292</data>
</node>
```

### 3. Problemas de Memória

#### Erro: MemoryError ou sistema lento

**Problema:**
- Sistema fica lento durante execução
- Erro de memória insuficiente
- Processo é morto pelo sistema

**Soluções:**

1. **Usar modo rápido:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --fast-mode
```

2. **Reduzir parâmetros:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --iterations 20 \
    --ants 10 \
    --compare-runs 5
```

3. **Configurar limites de memória:**
```json
{
    "performance": {
        "max_iterations_to_store": 100,
        "enable_memory_optimization": true,
        "garbage_collection_frequency": 10
    }
}
```

4. **Monitorar uso de memória:**
```bash
# Instalar psutil se não tiver
pip install psutil

# Executar com monitoramento
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --verbose
```

### 4. Problemas de Visualização

#### Erro: Gráficos não são gerados

**Problema:**
- Nenhum arquivo de imagem é criado
- Erro ao salvar visualizações

**Soluções:**

1. **Verificar backend do matplotlib:**
```python
import matplotlib
print(matplotlib.get_backend())

# Se necessário, configurar backend
import matplotlib
matplotlib.use('Agg')  # Para sistemas sem display
```

2. **Verificar permissões de escrita:**
```bash
# Verificar permissões do diretório
ls -la output/metrics/

# Criar diretório se necessário
mkdir -p output/metrics/visualizations

# Verificar espaço em disco
df -h
```

3. **Testar formatos diferentes:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --visualization-formats png  # Apenas PNG
```

#### Erro: Fontes não encontradas

**Problema:**
```
UserWarning: Glyph missing from current font
```

**Solução:**
```python
# Configurar fontes do matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'

# Ou no arquivo de configuração
{
    "visualizations": {
        "font_family": "DejaVu Sans",
        "use_latex_fonts": false
    }
}
```

### 5. Problemas de Relatórios

#### Erro: Relatórios vazios ou incompletos

**Problema:**
- Arquivo de relatório é criado mas está vazio
- Seções faltando no relatório

**Soluções:**

1. **Verificar se execuções foram bem-sucedidas:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --verbose  # Para ver logs detalhados
```

2. **Verificar dados de execução:**
```bash
# Listar arquivos de dados
ls -la output/metrics/execution_data/

# Verificar conteúdo de um arquivo
cat output/metrics/execution_data/execution_*.json
```

3. **Usar configuração de debug:**
```json
{
    "debug": {
        "enable_detailed_logging": true,
        "save_intermediate_results": true
    }
}
```

#### Erro: Encoding de caracteres

**Problema:**
```
UnicodeDecodeError: 'ascii' codec can't decode byte
```

**Solução:**
```bash
# Definir encoding do sistema
export PYTHONIOENCODING=utf-8

# Ou no código Python
import locale
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
```

### 6. Problemas de Performance

#### Execução muito lenta

**Problema:**
- Execução demora muito tempo
- Sistema trava durante processamento

**Soluções:**

1. **Usar processamento paralelo:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --compare-runs 10 \
    --parallel-executions
```

2. **Otimizar parâmetros:**
```bash
# Reduzir complexidade
python -m rota_aco.cli.run \
    --graph graphml/pequeno.graphml \  # Usar grafo menor
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --iterations 30 \  # Menos iterações
    --ants 15          # Menos formigas
```

3. **Usar modo rápido:**
```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --fast-mode
```

### 7. Problemas de Configuração

#### Erro: Arquivo de configuração inválido

**Problema:**
```
JSONDecodeError: Expecting ',' delimiter
```

**Solução:**
1. **Validar JSON:**
```bash
# Usar ferramenta online ou
python -m json.tool config.json
```

2. **Exemplo de configuração válida:**
```json
{
    "convergence_analysis": {
        "convergence_threshold": 0.001,
        "stability_window": 50
    },
    "visualizations": {
        "figure_dpi": 300,
        "output_formats": ["png", "svg"]
    }
}
```

#### Erro: Configuração não aplicada

**Problema:**
- Configuração personalizada é ignorada
- Valores padrão são usados

**Solução:**
```bash
# Verificar caminho do arquivo
ls -la /caminho/para/config.json

# Usar caminho absoluto
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics-config /caminho/absoluto/para/config.json
```

### 8. Problemas de Reprodutibilidade

#### Resultados diferentes a cada execução

**Problema:**
- Resultados variam mesmo com mesmos parâmetros
- Não consegue reproduzir resultados

**Solução:**
```bash
# Usar semente fixa
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --seed 42

# Salvar dados para reprodução
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --seed 42 \
    --save-execution-data \
    --export-raw-data
```

## Diagnóstico Avançado

### Script de Diagnóstico

Crie um arquivo `diagnostico.py`:

```python
#!/usr/bin/env python3
"""Script de diagnóstico do sistema de métricas."""

import sys
import os
import json
import tempfile
from pathlib import Path

def check_dependencies():
    """Verifica dependências."""
    print("=== VERIFICAÇÃO DE DEPENDÊNCIAS ===")
    
    required_modules = [
        'matplotlib', 'seaborn', 'numpy', 'pandas', 'scipy'
    ]
    
    optional_modules = [
        'psutil', 'joblib'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module} - OBRIGATÓRIO")
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module} (opcional)")
        except ImportError:
            print(f"⚠️  {module} - opcional, mas recomendado")

def check_file_system():
    """Verifica sistema de arquivos."""
    print("\n=== VERIFICAÇÃO DO SISTEMA DE ARQUIVOS ===")
    
    # Verificar diretório atual
    cwd = Path.cwd()
    print(f"Diretório atual: {cwd}")
    
    # Verificar estrutura do projeto
    expected_files = [
        'src/rota_aco/cli/run.py',
        'src/rota_aco/metrics/__init__.py'
    ]
    
    for file_path in expected_files:
        if (cwd / file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    # Verificar permissões de escrita
    try:
        with tempfile.NamedTemporaryFile(dir=cwd, delete=True) as f:
            print("✅ Permissões de escrita OK")
    except Exception as e:
        print(f"❌ Problema de permissões: {e}")

def check_graphml_files():
    """Verifica arquivos GraphML."""
    print("\n=== VERIFICAÇÃO DE ARQUIVOS GRAPHML ===")
    
    graphml_dir = Path('graphml')
    if graphml_dir.exists():
        graphml_files = list(graphml_dir.glob('*.graphml'))
        if graphml_files:
            for file in graphml_files:
                print(f"✅ {file}")
        else:
            print("⚠️  Nenhum arquivo .graphml encontrado")
    else:
        print("❌ Diretório graphml/ não encontrado")

def test_basic_functionality():
    """Testa funcionalidade básica."""
    print("\n=== TESTE DE FUNCIONALIDADE BÁSICA ===")
    
    try:
        # Importar módulos principais
        sys.path.insert(0, 'src')
        from rota_aco.metrics.config import MetricsConfig
        from rota_aco.metrics.data_models import ExecutionData
        
        # Criar configuração
        config = MetricsConfig()
        print("✅ Configuração de métricas criada")
        
        # Testar criação de diretórios
        with tempfile.TemporaryDirectory() as temp_dir:
            config.base_output_dir = temp_dir
            config._ensure_directories()
            print("✅ Criação de diretórios OK")
        
    except Exception as e:
        print(f"❌ Erro na funcionalidade básica: {e}")

def main():
    """Função principal de diagnóstico."""
    print("DIAGNÓSTICO DO SISTEMA DE MÉTRICAS")
    print("=" * 50)
    
    check_dependencies()
    check_file_system()
    check_graphml_files()
    test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("DIAGNÓSTICO CONCLUÍDO")
    print("\nSe houver problemas marcados com ❌, resolva-os antes de usar o sistema.")

if __name__ == "__main__":
    main()
```

Execute o diagnóstico:
```bash
python diagnostico.py
```

### Logs Detalhados

Para debug avançado, use:

```bash
python -m rota_aco.cli.run \
    --graph graphml/grafo.graphml \
    --start-lat -15.7801 \
    --start-lon -47.9292 \
    --exit-lat -15.7901 \
    --exit-lon -47.9392 \
    --metrics \
    --verbose \
    2>&1 | tee debug.log
```

### Configuração de Debug

Crie `debug_config.json`:

```json
{
    "debug": {
        "enable_detailed_logging": true,
        "save_intermediate_results": true,
        "log_memory_usage": true,
        "log_execution_times": true
    },
    "performance": {
        "enable_profiling": true,
        "profile_output": "profile_results.txt"
    }
}
```

## Contato e Suporte

Se os problemas persistirem:

1. **Verifique a documentação completa** em `docs/metrics_system_guide.md`
2. **Execute o script de diagnóstico** para identificar problemas
3. **Colete logs detalhados** usando `--verbose`
4. **Documente o problema** com:
   - Comando executado
   - Mensagem de erro completa
   - Resultado do diagnóstico
   - Versão do Python e sistema operacional

## Problemas Conhecidos

### Windows
- Problemas com encoding UTF-8: use `chcp 65001` no prompt
- Caminhos longos: use caminhos relativos curtos

### macOS
- Problemas com matplotlib: `brew install python-tk`
- Permissões: use `sudo` apenas se necessário

### Linux
- Dependências do sistema: `sudo apt-get install python3-tk`
- Fontes: `sudo apt-get install fonts-dejavu`

## Atualizações e Melhorias

O sistema está em desenvolvimento ativo. Para atualizações:

1. Verifique a documentação mais recente
2. Execute testes após atualizações
3. Reporte bugs encontrados
4. Sugira melhorias baseadas no uso real