#!/usr/bin/env python3
"""
Demonstração das funcionalidades de métricas no CLI.

Este script mostra exemplos de como usar os novos flags de métricas
na interface de linha de comando do Rota_ACO.
"""

import os
import sys
import subprocess
import tempfile

def run_command(cmd, description):
    """Executa um comando e mostra o resultado."""
    print(f"\n{'='*60}")
    print(f"DEMONSTRAÇÃO: {description}")
    print(f"{'='*60}")
    print(f"Comando: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Comando executado com sucesso!")
            if result.stdout:
                print("Saída:")
                print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ Comando falhou!")
            if result.stderr:
                print("Erro:")
                print(result.stderr[:500] + "..." if len(result.stderr) > 500 else result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Comando expirou (timeout)")
    except Exception as e:
        print(f"❌ Erro ao executar comando: {e}")


def main():
    """Demonstra as funcionalidades de métricas do CLI."""
    print("DEMONSTRAÇÃO DAS FUNCIONALIDADES DE MÉTRICAS NO CLI")
    print("=" * 60)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("src/rota_aco/cli/run.py"):
        print("❌ Execute este script a partir do diretório raiz do projeto.")
        sys.exit(1)
    
    # Criar diretório temporário para outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Usando diretório temporário: {temp_dir}")
        
        # Comando base (seria necessário ter um arquivo GraphML real para execução completa)
        base_cmd = [
            sys.executable, "-m", "rota_aco.cli.run",
            "--graph", "graphml/pequeno.graphml",  # Assumindo que existe
            "--start-lat", "-15.7801",
            "--start-lon", "-47.9292", 
            "--exit-lat", "-15.7901",
            "--exit-lon", "-47.9392",
            "--iterations", "3",  # Poucas iterações para demo
            "--ants", "5"
        ]
        
        # 1. Demonstrar help com novos flags
        help_cmd = [sys.executable, "-m", "rota_aco.cli.run", "--help"]
        run_command(help_cmd, "Mostrando ajuda com novos flags de métricas")
        
        # 2. Demonstrar flag básico de métricas
        metrics_cmd = base_cmd + [
            "--metrics",
            "--report-output", os.path.join(temp_dir, "basic_metrics")
        ]
        run_command(metrics_cmd, "Execução básica com métricas habilitadas")
        
        # 3. Demonstrar análise de convergência
        convergence_cmd = base_cmd + [
            "--convergence-analysis",
            "--report-output", os.path.join(temp_dir, "convergence_analysis")
        ]
        run_command(convergence_cmd, "Análise detalhada de convergência")
        
        # 4. Demonstrar comparação de múltiplas execuções
        compare_cmd = base_cmd + [
            "--compare-runs", "3",
            "--statistical-tests",
            "--report-output", os.path.join(temp_dir, "comparison")
        ]
        run_command(compare_cmd, "Comparação de múltiplas execuções")
        
        # 5. Demonstrar modo acadêmico
        academic_cmd = base_cmd + [
            "--academic-mode",
            "--export-raw-data",
            "--visualization-formats", "png", "svg",
            "--report-output", os.path.join(temp_dir, "academic")
        ]
        run_command(academic_cmd, "Modo acadêmico com alta qualidade")
        
        # 6. Demonstrar modo rápido
        fast_cmd = base_cmd + [
            "--fast-mode",
            "--report-output", os.path.join(temp_dir, "fast")
        ]
        run_command(fast_cmd, "Modo rápido para testes")
        
        # 7. Demonstrar configuração personalizada
        config_file = os.path.join(temp_dir, "custom_config.json")
        with open(config_file, 'w') as f:
            f.write("""{
    "convergence_analysis": {
        "convergence_threshold": 0.005,
        "stability_window": 20
    },
    "visualizations": {
        "figure_dpi": 150,
        "output_formats": ["png"]
    },
    "reports": {
        "include_raw_data": true
    }
}""")
        
        config_cmd = base_cmd + [
            "--metrics-config", config_file,
            "--report-output", os.path.join(temp_dir, "custom_config")
        ]
        run_command(config_cmd, "Configuração personalizada via arquivo JSON")
        
        # 8. Demonstrar execução com semente para reprodutibilidade
        seed_cmd = base_cmd + [
            "--metrics",
            "--seed", "42",
            "--compare-runs", "2",
            "--report-output", os.path.join(temp_dir, "reproducible")
        ]
        run_command(seed_cmd, "Execução reprodutível com semente")
        
        print(f"\n{'='*60}")
        print("RESUMO DOS NOVOS FLAGS DE MÉTRICAS")
        print(f"{'='*60}")
        print("""
FLAGS BÁSICOS:
  --metrics                    Habilita coleta de métricas expandidas
  --report-output DIR          Define diretório de saída para relatórios
  --compare-runs N             Executa N vezes para análise comparativa
  --convergence-analysis       Análise detalhada de convergência
  --metrics-config FILE        Configuração personalizada via JSON

FLAGS AVANÇADOS:
  --statistical-tests          Habilita testes estatísticos
  --confidence-level LEVEL     Nível de confiança (padrão: 0.95)
  --export-raw-data           Inclui dados brutos nos relatórios
  --visualization-formats      Formatos de saída (png, svg, pdf)
  --academic-mode             Modo acadêmico: alta qualidade
  --fast-mode                 Modo rápido: análise simplificada
  --parallel-executions       Execuções paralelas (experimental)
  --seed NUMBER               Semente para reprodutibilidade
  --save-execution-data       Salva dados de execução (padrão: sim)
  --load-previous-data PATH   Carrega dados anteriores para comparação

EXEMPLOS DE USO:
  # Análise básica com métricas
  python -m rota_aco.cli.run --graph grafo.graphml ... --metrics
  
  # Comparação de 5 execuções com testes estatísticos
  python -m rota_aco.cli.run --graph grafo.graphml ... --compare-runs 5 --statistical-tests
  
  # Modo acadêmico para publicação
  python -m rota_aco.cli.run --graph grafo.graphml ... --academic-mode --export-raw-data
  
  # Execução rápida para testes
  python -m rota_aco.cli.run --graph grafo.graphml ... --fast-mode
  
  # Configuração personalizada
  python -m rota_aco.cli.run --graph grafo.graphml ... --metrics-config config.json
        """)
        
        print(f"\n{'='*60}")
        print("DEMONSTRAÇÃO CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Arquivos de exemplo criados em: {temp_dir}")
        print("Para execução real, certifique-se de ter um arquivo GraphML válido.")


if __name__ == "__main__":
    main()