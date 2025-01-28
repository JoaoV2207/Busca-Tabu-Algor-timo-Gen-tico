import numpy as np
from typing import Tuple, List, Optional
import time

class TabuList:
    """Lista Tabu circular com tamanho fixo"""
    def __init__(self, max_size: int):
        self.data = [(-1, -1) for _ in range(max_size)]
        self.insert_pos = 0
        self.max_size = max_size

    def append(self, pos_changed: Tuple[int, int]) -> None:
        """Adiciona uma nova posição à lista tabu"""
        self.data[self.insert_pos] = pos_changed
        self.insert_pos = (self.insert_pos + 1) % self.max_size

    def __contains__(self, value: Tuple[int, int]) -> bool:
        """Verifica se uma posição está na lista tabu"""
        return value in self.data

class MCLP_TabuSearch:
    """Implementação do Tabu Search para o MCLP"""
    def __init__(self, cover_matrix: np.ndarray, population: np.ndarray, num_bases: int):
        self.cover_matrix = cover_matrix
        self.population = population
        self.num_bases = num_bases
        self.best_solution = None
        self.best_cost = -float('inf')
        
    def generate_initial_solution(self) -> np.ndarray:
        """Gera solução inicial baseada na cobertura populacional"""
        coverage_scores = np.sum(self.cover_matrix * self.population[:, np.newaxis], axis=0)
        main_index = np.argsort(coverage_scores)[-self.num_bases:]
        solution = np.zeros(self.cover_matrix.shape[1])
        solution[main_index] = 1
        return solution

    def calculate_coverage(self, solution: np.ndarray) -> float:
        """Calcula a cobertura populacional total de uma solução"""
        coverage = self.cover_matrix[:, solution.astype(bool)]
        coverage = np.logical_or.reduce(coverage, axis=1)
        return np.sum(coverage * self.population)

    def get_neighbors(self, solution: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Gera vizinhança baseada em trocas 1-1"""
        zeros = np.where(solution == 0)[0]
        ones = np.where(solution == 1)[0]
        neighbors = []
        pos_changed = []
        
        for zero in zeros:
            for one in ones:
                neighbor = solution.copy()
                neighbor[zero] = 1
                neighbor[one] = 0
                neighbors.append(neighbor)
                pos_changed.append((min(zero, one), max(zero, one)))
                
        return neighbors, pos_changed

    def search(self, max_iterations: int, tabu_size: int, 
              time_limit: Optional[float] = None) -> Tuple[np.ndarray, float, int]:
        """
        Executa o algoritmo Tabu Search
        
        Parâmetros:
        - max_iterations: número máximo de iterações
        - tabu_size: tamanho da lista tabu
        - time_limit: limite de tempo em segundos (opcional)
        
        Retorna:
        - Melhor solução encontrada
        - Valor da função objetivo
        - Número de iterações realizadas
        """
        start_time = time.time()
        tabu_list = TabuList(tabu_size)
        
        # Inicialização
        current_solution = self.generate_initial_solution()
        self.best_solution = current_solution.copy()
        self.best_cost = self.calculate_coverage(current_solution)
        
        iteration = 0
        while iteration < max_iterations:
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
            # Gera e avalia vizinhança
            neighbors, positions = self.get_neighbors(current_solution)
            if not neighbors:
                break
                
            # Avalia todos os vizinhos
            neighbor_costs = [self.calculate_coverage(n) for n in neighbors]
            
            # Ordena vizinhos por custo
            sorted_indices = np.argsort(neighbor_costs)[::-1]
            
            # Seleciona melhor vizinho não-tabu
            moved = False
            for idx in sorted_indices:
                if positions[idx] not in tabu_list:
                    current_solution = neighbors[idx]
                    current_cost = neighbor_costs[idx]
                    tabu_list.append(positions[idx])
                    moved = True
                    
                    # Atualiza melhor solução se necessário
                    if current_cost > self.best_cost:
                        self.best_solution = current_solution.copy()
                        self.best_cost = current_cost
                    break
            
            if not moved:
                # Se todos os movimentos estão na lista tabu, escolhe o melhor
                best_idx = sorted_indices[0]
                current_solution = neighbors[best_idx]
                current_cost = neighbor_costs[best_idx]
                tabu_list.append(positions[best_idx])
                
                if current_cost > self.best_cost:
                    self.best_solution = current_solution.copy()
                    self.best_cost = current_cost
            
            iteration += 1
            
        return self.best_solution, self.best_cost, iteration