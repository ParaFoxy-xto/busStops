"""
Testes para o sistema de coleta de dados.

Este módulo testa a funcionalidade do DataCollector e suas integrações
com os algoritmos ACO existentes.
"""

import pytest
import os
import tempfile
import json
from datetime import datetime
from unittest.mock import Mock, patch

from src.rota_aco.metrics.data_collector import DataCollector, ACODataCollectorWrapper
from src.rota_aco.metrics.data_models import Route, Solution, IterationData
from src.rota_aco.metrics.config import MetricsConfig
from src.rota_aco.metrics.exceptions import DataCollectionError, FileOperationError


class TestDataCollector:
    """Testes para a classe DataCollector."""
    
    def setup_method(self):
        """Setup para cada teste."""
        # Use temporary directory for tests
        self.temp_dir = tempfile.mkdtemp()
        self.config = MetricsConfig(base_output_dir=self.temp_dir)
        self.collector = DataCollector(self.config)
    
    def teardown_method(self):
        """Cleanup após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_route(self) -> Route:
        """Cria uma rota de exemplo para testes."""
        return Route(
            stops=[1, 2, 3],
            distances=[100.0, 150.0],
            passenger_load=[20, 15],
            total_distance=250.0,
            total_passengers=35,
            is_valid=True,
            capacity_violations=0,
            opposite_stops_violations=0
        )
    
    def create_sample_solution(self) -> Solution:
        """Cria uma solução de exemplo para testes."""
        route = self.create_sample_route()
        return Solution(
            routes=[route],
            total_vehicles=1,
            total_distance=250.0,
            total_passengers_served=35,
            fitness_time=250.0,
            fitness_vehicle=1250.0,
            is_feasible=True,
            generation_time=1.5
        )
    
    def test_start_execution(self):
        """Testa início de execução."""
        execution_id = self.collector.start_execution(
            algorithm_type="ACS-TIME",
            config={"ants": 10, "iterations": 50}
        )
        
        assert execution_id is not None
        assert self.collector.current_execution is not None
        assert self.collector.current_execution.algorithm_type == "ACS-TIME"
        assert self.collector.current_execution.config["ants"] == 10
    
    def test_start_execution_with_custom_id(self):
        """Testa início de execução com ID customizado."""
        custom_id = "test-execution-123"
        execution_id = self.collector.start_execution(
            algorithm_type="ACS-VEHICLE",
            config={"ants": 20},
            execution_id=custom_id
        )
        
        assert execution_id == custom_id
        assert self.collector.current_execution.execution_id == custom_id
    
    def test_start_execution_already_running(self):
        """Testa erro ao iniciar execução quando já há uma em andamento."""
        self.collector.start_execution("ACS-TIME", {})
        
        with pytest.raises(DataCollectionError, match="Já existe uma execução em andamento"):
            self.collector.start_execution("ACS-VEHICLE", {})
    
    def test_record_iteration(self):
        """Testa registro de dados de iteração."""
        self.collector.start_execution("ACS-TIME", {})
        solution = self.create_sample_solution()
        
        self.collector.record_iteration(
            iteration=1,
            best_fitness=250.0,
            avg_fitness=300.0,
            population_variance=25.0,
            best_solution=solution,
            additional_metrics={"custom_metric": 42}
        )
        
        assert len(self.collector.current_execution.iterations_data) == 1
        iteration_data = self.collector.current_execution.iterations_data[0]
        assert iteration_data.iteration == 1
        assert iteration_data.best_fitness == 250.0
        assert iteration_data.additional_metrics["custom_metric"] == 42
    
    def test_record_iteration_no_execution(self):
        """Testa erro ao registrar iteração sem execução em andamento."""
        solution = self.create_sample_solution()
        
        with pytest.raises(DataCollectionError, match="Nenhuma execução em andamento"):
            self.collector.record_iteration(1, 250.0, 300.0, 25.0, solution)
    
    def test_record_final_solution(self):
        """Testa registro de solução final."""
        self.collector.start_execution("ACS-TIME", {})
        solution = self.create_sample_solution()
        
        self.collector.record_final_solution(
            solution=solution,
            execution_time=45.5,
            success=True
        )
        
        assert self.collector.current_execution.final_solution == solution
        assert self.collector.current_execution.execution_time == 45.5
        assert self.collector.current_execution.success is True
        assert len(self.collector.current_execution.routes) == 1
    
    def test_record_final_solution_failure(self):
        """Testa registro de falha na execução."""
        self.collector.start_execution("ACS-TIME", {})
        
        self.collector.record_final_solution(
            solution=None,
            execution_time=10.0,
            success=False,
            error_message="Algorithm failed to converge"
        )
        
        assert self.collector.current_execution.final_solution is None
        assert self.collector.current_execution.success is False
        assert self.collector.current_execution.error_message == "Algorithm failed to converge"
    
    def test_finish_execution(self):
        """Testa finalização de execução."""
        self.collector.start_execution("ACS-TIME", {})
        solution = self.create_sample_solution()
        self.collector.record_final_solution(solution, 45.5, True)
        
        execution_data = self.collector.finish_execution(save_to_disk=False)
        
        assert execution_data is not None
        assert self.collector.current_execution is None
        assert len(self.collector.execution_history) == 1
        assert self.collector.execution_history[0] == execution_data
    
    def test_finish_execution_no_execution(self):
        """Testa erro ao finalizar sem execução em andamento."""
        with pytest.raises(DataCollectionError, match="Nenhuma execução em andamento"):
            self.collector.finish_execution()
    
    def test_save_and_load_execution_data(self):
        """Testa salvamento e carregamento de dados."""
        # Create and finish execution
        execution_id = self.collector.start_execution("ACS-TIME", {"ants": 10})
        solution = self.create_sample_solution()
        self.collector.record_final_solution(solution, 45.5, True)
        execution_data = self.collector.finish_execution(save_to_disk=True)
        
        # Load from disk
        loaded_data = self.collector.load_execution_data(execution_id)
        
        assert loaded_data.execution_id == execution_id
        assert loaded_data.algorithm_type == "ACS-TIME"
        assert loaded_data.config["ants"] == 10
        assert loaded_data.execution_time == 45.5
        assert loaded_data.success is True
    
    def test_load_nonexistent_execution(self):
        """Testa erro ao carregar execução inexistente."""
        with pytest.raises(FileOperationError, match="Dados de execução não encontrados"):
            self.collector.load_execution_data("nonexistent-id")
    
    def test_list_executions(self):
        """Testa listagem de execuções."""
        # Initially empty
        assert self.collector.list_executions() == []
        
        # Create and save executions
        id1 = self.collector.start_execution("ACS-TIME", {})
        self.collector.record_final_solution(self.create_sample_solution(), 30.0, True)
        self.collector.finish_execution()
        
        id2 = self.collector.start_execution("ACS-VEHICLE", {})
        self.collector.record_final_solution(self.create_sample_solution(), 25.0, True)
        self.collector.finish_execution()
        
        executions = self.collector.list_executions()
        assert len(executions) == 2
        assert id1 in executions
        assert id2 in executions
    
    def test_load_multiple_executions(self):
        """Testa carregamento de múltiplas execuções."""
        # Create executions
        ids = []
        for i in range(3):
            execution_id = self.collector.start_execution(f"ACS-TIME", {"run": i})
            self.collector.record_final_solution(self.create_sample_solution(), 30.0, True)
            self.collector.finish_execution()
            ids.append(execution_id)
        
        # Load all
        all_executions = self.collector.load_multiple_executions()
        assert len(all_executions) == 3
        
        # Load specific
        specific_executions = self.collector.load_multiple_executions([ids[0], ids[2]])
        assert len(specific_executions) == 2
        assert specific_executions[0].config["run"] in [0, 2]
        assert specific_executions[1].config["run"] in [0, 2]
    
    def test_callbacks(self):
        """Testa sistema de callbacks."""
        iteration_callback = Mock()
        solution_callback = Mock()
        
        self.collector.add_iteration_callback(iteration_callback)
        self.collector.add_solution_callback(solution_callback)
        
        # Start execution and record data
        self.collector.start_execution("ACS-TIME", {})
        solution = self.create_sample_solution()
        
        self.collector.record_iteration(1, 250.0, 300.0, 25.0, solution)
        self.collector.record_final_solution(solution, 45.5, True)
        
        # Check callbacks were called
        iteration_callback.assert_called_once()
        solution_callback.assert_called_once()
        
        # Clear callbacks
        self.collector.clear_callbacks()
        assert len(self.collector.iteration_callbacks) == 0
        assert len(self.collector.solution_callbacks) == 0
    
    def test_callback_error_handling(self):
        """Testa tratamento de erros em callbacks."""
        def failing_callback(data):
            raise Exception("Callback failed")
        
        self.collector.add_iteration_callback(failing_callback)
        
        # Should not raise exception even if callback fails
        self.collector.start_execution("ACS-TIME", {})
        solution = self.create_sample_solution()
        
        # This should not raise an exception
        self.collector.record_iteration(1, 250.0, 300.0, 25.0, solution)
    
    def test_get_execution_summary(self):
        """Testa geração de resumo de execuções."""
        # Initially empty
        summary = self.collector.get_execution_summary()
        assert summary['total_executions'] == 0
        assert summary['success_rate'] == 0.0
        
        # Add successful execution
        self.collector.start_execution("ACS-TIME", {})
        self.collector.record_final_solution(self.create_sample_solution(), 30.0, True)
        self.collector.finish_execution(save_to_disk=False)
        
        # Add failed execution
        self.collector.start_execution("ACS-VEHICLE", {})
        self.collector.record_final_solution(None, 15.0, False, "Error")
        self.collector.finish_execution(save_to_disk=False)
        
        summary = self.collector.get_execution_summary()
        assert summary['total_executions'] == 2
        assert summary['successful_executions'] == 1
        assert summary['success_rate'] == 0.5
        assert summary['algorithms']['ACS-TIME']['count'] == 1
        assert summary['algorithms']['ACS-TIME']['successful'] == 1
        assert summary['algorithms']['ACS-VEHICLE']['count'] == 1
        assert summary['algorithms']['ACS-VEHICLE']['successful'] == 0


class TestACODataCollectorWrapper:
    """Testes para o wrapper de integração."""
    
    def setup_method(self):
        """Setup para cada teste."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MetricsConfig(base_output_dir=self.temp_dir)
        self.collector = DataCollector(self.config)
        self.wrapper = ACODataCollectorWrapper(self.collector)
    
    def teardown_method(self):
        """Cleanup após cada teste."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wrap_controller_run_success(self):
        """Testa wrapper com execução bem-sucedida."""
        # Mock controller run method
        def mock_controller_run():
            return ([[1, 2, 3], [4, 5, 6]], 500.0, 2, 0.95)
        
        # Execute wrapped method
        result = self.wrapper.wrap_controller_run(
            mock_controller_run,
            "ACS-TIME",
            {"ants": 10, "iterations": 50}
        )
        
        # Check result is preserved
        assert result == ([[1, 2, 3], [4, 5, 6]], 500.0, 2, 0.95)
        
        # Check data was collected
        assert len(self.collector.execution_history) == 1
        execution = self.collector.execution_history[0]
        assert execution.algorithm_type == "ACS-TIME"
        assert execution.success is True
        assert execution.final_solution is not None
    
    def test_wrap_controller_run_failure(self):
        """Testa wrapper com execução que falha."""
        def mock_controller_run():
            raise Exception("Algorithm failed")
        
        # Should re-raise the exception
        with pytest.raises(Exception, match="Algorithm failed"):
            self.wrapper.wrap_controller_run(
                mock_controller_run,
                "ACS-TIME",
                {"ants": 10}
            )
        
        # But should still record the failure
        assert len(self.collector.execution_history) == 1
        execution = self.collector.execution_history[0]
        assert execution.success is False
        assert execution.error_message == "Algorithm failed"


if __name__ == "__main__":
    pytest.main([__file__])