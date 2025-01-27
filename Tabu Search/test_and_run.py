import numpy as np
import json
import time
from tabu_search import MCLP_TabuSearch

def generate_scaled_test_cases():
    test_cases = dict()
    
    # Configurações de tamanhos para as matrizes
    sizes = [
        (100, 30),    # 100x30
        (200, 40),    # 200x40
        (300, 50),    # 300x50
        (400, 60),    # 400x60
        (500, 70),    # 500x70
        (600, 80),    # 600x80
        (700, 90),    # 700x90
        (800, 100),   # 800x100
        (900, 110),   # 900x110
        (1000, 120),  # 1000x120
        (1200, 130),  # 1200x130
        (1400, 140),  # 1400x140
        (1600, 150),  # 1600x150
        (1800, 160),  # 1800x160
        (2000, 170)   # 2000x170
    ]
    
    for rows, cols in sizes:
        case_name = f"matrix_{rows:04d}x{cols:03d}"
        
        # Cálculo do número de bases baseado no tamanho da matriz
        num_bases = max(5, cols // 10)
        
        # Criação da matriz com esparsidade variável
        sparsity = min(0.95, 0.85 + (rows / 20000))  # Aumenta esparsidade com o tamanho
        matriz = np.random.choice([0, 1], size=(rows, cols), p=[sparsity, 1-sparsity])
        
        # População com clusters
        populacao = create_clustered_population(rows)
        
        test_cases[case_name] = {
            'matriz': matriz,
            'populacao': populacao,
            'num_bases': num_bases,
            'descricao': f'Matriz {rows}x{cols} com {num_bases} bases'
        }
    
    return test_cases

def create_clustered_population(size):
    """Cria população com clusters e variação exponencial"""
    population = np.zeros(size)
    
    # População base com distribuição exponencial
    population = np.random.exponential(1000, size)
    
    # Hotspots
    num_hotspots = size // 50
    hotspot_indices = np.random.choice(size, num_hotspots, replace=False)
    population[hotspot_indices] = np.random.exponential(10000, num_hotspots)
    
    # Normalização para valores inteiros
    population = population.astype(int)
    population = np.maximum(population, 100)  # Mínimo de 100 pessoas
    
    return population

def save_test_cases(cases, filename='test_cases.json'):
    cases_json = {}
    for name, case in cases.items():
        cases_json[name] = {
            'matriz': case['matriz'].tolist(),
            'populacao': case['populacao'].tolist(),
            'num_bases': case['num_bases'],
            'descricao': case['descricao']
        }
    
    with open(filename, 'w') as f:
        json.dump(cases_json, f, indent=2)
    print(f"Casos de teste salvos em {filename}")

def load_test_cases(filename='test_cases.json'):
    with open(filename, 'r') as f:
        cases = json.load(f)
    
    loaded_cases = {}
    for name, case in cases.items():
        loaded_cases[name] = {
            'matriz': np.array(case['matriz']),
            'populacao': np.array(case['populacao']),
            'num_bases': case['num_bases'],
            'descricao': case['descricao']
        }
    return loaded_cases

def run_tabu_search(test_cases):
    results = {}
    
    for case_name, case in sorted(test_cases.items()):
        print(f"\nExecutando Tabu Search para caso: {case_name}")
        print(f"Descrição: {case['descricao']}")
        
        start_time = time.time()
        
        ts = MCLP_TabuSearch(
            cover_matrix=case['matriz'],
            population=case['populacao'],
            num_bases=case['num_bases']
        )
        
        # Define número de iterações baseado no tamanho da matriz
        matriz_shape = case['matriz'].shape
        max_iter = matriz_shape[0] * matriz_shape[1] // 50  # Uma iteração a cada 50 elementos
        tabu_size = min(100, matriz_shape[1] // 2)
        
        solution, cost, iterations = ts.search(
            max_iterations=max_iter,
            tabu_size=tabu_size,
            time_limit=300  # 5 minutos por caso
        )
        
        execution_time = time.time() - start_time
        
        results[case_name] = {
            'cobertura': float(cost),
            'cobertura_percentual': float((cost/case['populacao'].sum())*100),
            'iteracoes': iterations,
            'tempo_execucao': execution_time,
            'posicoes_bases': np.where(solution == 1)[0].tolist()
        }
        
        print(f"Dimensões da matriz: {case['matriz'].shape}")
        print(f"Número de bases: {case['num_bases']}")
        print(f"Cobertura populacional: {cost:,.0f}")
        print(f"Porcentagem coberta: {(cost/case['populacao'].sum())*100:.2f}%")
        print(f"Iterações: {iterations}")
        print(f"Tempo de execução: {execution_time:.2f}s")
    
    with open('resultados_tabu.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResultados salvos em resultados_tabu.json")

if __name__ == "__main__":
    print("Gerando casos de teste...")
    test_cases = generate_scaled_test_cases()
    save_test_cases(test_cases)
    
    print("\nCarregando casos de teste...")
    loaded_cases = load_test_cases()
    
    print("\nIniciando execução do Tabu Search...")
    run_tabu_search(loaded_cases)