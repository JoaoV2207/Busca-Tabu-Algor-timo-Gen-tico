import numpy as np
import json
import time
from tabu_search import MCLP_TabuSearch
from Algoritmo_Gen import solveGeneticAlgorithm, objectiveFunction

def generate_scaled_test_cases():
    test_cases = dict()
    sizes = [
        (100, 30), (200, 40), (300, 50), (400, 60), (500, 70),
        (600, 80), (700, 90), (800, 100), (900, 110), (1000, 120),
        (1200, 130), (1400, 140), (1600, 150), (1800, 160), (2000, 170)
    ]
    
    print("Gerando casos de teste:")
    for rows, cols in sizes:
        print(f"Gerando caso {rows}x{cols}...")
        case_name = f"matrix_{rows:04d}x{cols:03d}"
        num_bases = max(5, cols // 10)
        sparsity = min(0.95, 0.85 + (rows / 20000))
        matriz = np.random.choice([0, 1], size=(rows, cols), p=[sparsity, 1-sparsity])
        populacao = create_clustered_population(rows)
        
        test_cases[case_name] = {
            'matriz': matriz,
            'populacao': populacao,
            'num_bases': num_bases,
            'descricao': f'Matriz {rows}x{cols} com {num_bases} bases'
        }
    return test_cases

def create_clustered_population(size):
    population = np.zeros(size)
    population = np.random.exponential(1000, size)
    num_hotspots = size // 50
    hotspot_indices = np.random.choice(size, num_hotspots, replace=False)
    population[hotspot_indices] = np.random.exponential(10000, num_hotspots)
    population = population.astype(int)
    population = np.maximum(population, 100)
    return population

def save_test_cases(cases, filename='test_cases.json'):
    print(f"\nSalvando casos de teste em {filename}...")
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
    print("Casos de teste salvos com sucesso!")

def load_test_cases(filename='test_cases.json'):
    print(f"\nCarregando casos de teste de {filename}...")
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
    print("Casos de teste carregados com sucesso!")
    return loaded_cases

def calculate_metrics(solution, case, cost, iterations, execution_time):
    coverage_matrix = case['matriz'][:, solution == 1]
    covered_points = np.any(coverage_matrix, axis=1)
    
    return {
        # Métricas originais
        'cobertura': float(cost),
        'cobertura_percentual': float((cost/case['populacao'].sum())*100),
        'iteracoes': iterations,
        'tempo_execucao': execution_time,
        'posicoes_bases': np.where(solution == 1)[0].tolist(),
        'dimensao_matriz': list(case['matriz'].shape),
        'num_bases': case['num_bases'],
        'populacao_total': float(case['populacao'].sum()),
        'densidade_matriz': float(np.sum(case['matriz']) / (case['matriz'].shape[0] * case['matriz'].shape[1])),
        
        # Métricas adicionais
        'pontos_cobertos': int(np.sum(covered_points)),
        'porcentagem_pontos_cobertos': float(np.sum(covered_points) / len(covered_points) * 100),
        'celulas_cobertas': int(np.sum(coverage_matrix)),
        'porcentagem_celulas_cobertas': float(np.sum(coverage_matrix) / coverage_matrix.size * 100),
        'cobertura_media_por_base': float(cost / case['num_bases']),
        'pontos_por_base': float(case['matriz'].shape[0] / case['num_bases']),
        'populacao_media_por_ponto': float(case['populacao'].sum() / case['matriz'].shape[0]),
        'tempo_por_iteracao': float(execution_time / iterations),
        'iteracoes_por_segundo': float(iterations / execution_time)
    }

def print_metrics(metrics):
    print(f"Dimensões: {metrics['dimensao_matriz']}")
    print(f"Número de bases: {metrics['num_bases']}")
    print(f"Cobertura populacional: {metrics['cobertura']:,.0f}")
    print(f"Porcentagem coberta: {metrics['cobertura_percentual']:.2f}%")
    print(f"Pontos de demanda cobertos: {metrics['pontos_cobertos']} ({metrics['porcentagem_pontos_cobertos']:.2f}%)")
    print(f"Células cobertas: {metrics['celulas_cobertas']} ({metrics['porcentagem_celulas_cobertas']:.2f}%)")
    print(f"Cobertura média por base: {metrics['cobertura_media_por_base']:,.0f}")
    print(f"Iterações: {metrics['iteracoes']}")
    print(f"Tempo de execução: {metrics['tempo_execucao']:.2f}s")
    print(f"Iterações por segundo: {metrics['iteracoes_por_segundo']:.2f}")
    print(f"Tempo por iteração: {metrics['tempo_por_iteracao']*1000:.2f}ms")

def run_tabu_search(test_cases):
    results = {}
    
    print("\nRODANDO TABU SEARCH")
    print("===================")
    
    total_cases = len(test_cases)
    current_case = 1
    
    for case_name, case in sorted(test_cases.items()):
        print(f"\nCaso {current_case}/{total_cases}: {case_name}")
        print(f"Dimensões: {case['matriz'].shape}")
        print("Iniciando busca Tabu...")
        
        start_time = time.time()
        
        ts = MCLP_TabuSearch(
            cover_matrix=case['matriz'],
            population=case['populacao'],
            num_bases=case['num_bases']
        )
        
        matriz_shape = case['matriz'].shape
        max_iter = matriz_shape[0] * matriz_shape[1] // 50
        tabu_size = min(100, matriz_shape[1] // 2)
        
        solution, cost, iterations = ts.search(
            max_iterations=max_iter,
            tabu_size=tabu_size,
            time_limit=300
        )
        
        execution_time = time.time() - start_time
        
        results[case_name] = calculate_metrics(solution, case, cost, iterations, execution_time)
        print_metrics(results[case_name])
        print(f"Caso {current_case} concluído!")
        
        current_case += 1
    
    with open('resultados_tabu.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResultados do Tabu Search salvos em resultados_tabu.json")
    return results

def run_genetic_algorithm(test_cases):
    results = {}
    
    print("\nRODANDO ALGORITMO GENÉTICO")
    print("=========================")
    
    total_cases = len(test_cases)
    current_case = 1
    
    for case_name, case in sorted(test_cases.items()):
        print(f"\nCaso {current_case}/{total_cases}: {case_name}")
        print(f"Dimensões: {case['matriz'].shape}")
        print("Iniciando algoritmo genético...")
        
        start_time = time.time()
        ngen = 100
        
        rng = np.random.default_rng(seed=153)
        
        best = solveGeneticAlgorithm(
            pop_size=400,
            ngen=ngen,
            pmut=0.8,
            tournamentk=5,
            coverMatrix=case['matriz'],
            numPeopleHelpedList=case['populacao'],
            limit=case['num_bases'],
            rng=rng
        )
        
        execution_time = time.time() - start_time
        cost = objectiveFunction(best, case['matriz'], case['populacao'])
        
        results[case_name] = calculate_metrics(best, case, cost, ngen, execution_time)
        print_metrics(results[case_name])
        print(f"Caso {current_case} concluído!")
        
        current_case += 1
    
    with open('resultados_genetico.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResultados do Algoritmo Genético salvos em resultados_genetico.json")
    return results

if __name__ == "__main__":
    print("Iniciando execução do experimento...")
    test_cases = generate_scaled_test_cases()
    save_test_cases(test_cases)
    
    loaded_cases = load_test_cases()
    
    results_tabu = run_tabu_search(loaded_cases)
    results_genetic = run_genetic_algorithm(loaded_cases)
    
    print("\nExperimento concluído! Resultados salvos em:")
    print("- resultados_tabu.json")
    print("- resultados_genetico.json")