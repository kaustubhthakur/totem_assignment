import random
import time
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TSPPreferences:
    quality_over_speed: float = 0.6
    max_computation_time: float = 5.0
    problem_size_threshold: int = 15
    genetic_generations: int = 1000
    population_size: int = 100

class UserBehaviorAnalyzer:
    def __init__(self):
        self.solution_history = []
        self.algorithm_performance = {
            'dp_bitmask': {'calls': 0, 'avg_time': 0, 'avg_quality': 0},
            'genetic': {'calls': 0, 'avg_time': 0, 'avg_quality': 0},
            'nearest_neighbor': {'calls': 0, 'avg_time': 0, 'avg_quality': 0}
        }
    
    def record_solution(self, algorithm: str, time_taken: float, solution_quality: int, problem_size: int):
        self.solution_history.append({
            'algorithm': algorithm,
            'time': time_taken,
            'quality': solution_quality,
            'size': problem_size
        })
        
        perf = self.algorithm_performance[algorithm]
        perf['calls'] += 1
        perf['avg_time'] = (perf['avg_time'] * (perf['calls'] - 1) + time_taken) / perf['calls']
        perf['avg_quality'] = (perf['avg_quality'] * (perf['calls'] - 1) + solution_quality) / perf['calls']
    
    def get_best_algorithm(self, problem_size: int, time_budget: float) -> str:
        if problem_size <= 12:
            return 'dp_bitmask'
        
        genetic_perf = self.algorithm_performance['genetic']
        nn_perf = self.algorithm_performance['nearest_neighbor']
        
        if genetic_perf['calls'] > 0 and nn_perf['calls'] > 0:
            if time_budget > genetic_perf['avg_time'] and genetic_perf['avg_quality'] < nn_perf['avg_quality'] * 1.2:
                return 'genetic'
        
        return 'nearest_neighbor' if time_budget < 1.0 else 'genetic'

class AdaptiveTSPSolver:
    def __init__(self, preferences: TSPPreferences, behavior_analyzer: UserBehaviorAnalyzer):
        self.preferences = preferences
        self.behavior_analyzer = behavior_analyzer
        self.memoization_cache = {}
        
    def _calculate_distance_matrix(self, cities: List[Tuple[float, float]]) -> List[List[float]]:
        n = len(cities)
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = cities[i][0] - cities[j][0]
                    dy = cities[i][1] - cities[j][1]
                    distances[i][j] = math.sqrt(dx * dx + dy * dy)
        
        return distances
    
    def _solve_dp_bitmask(self, distances: List[List[float]]) -> Tuple[float, List[int]]:
        n = len(distances)
        if n > 20:
            raise ValueError("DP approach not suitable for large instances")
        
        memo = {}
        
        def dp(mask: int, pos: int) -> float:
            if mask == (1 << n) - 1:
                return distances[pos][0]
            
            if (mask, pos) in memo:
                return memo[(mask, pos)]
            
            min_cost = float('inf')
            for next_city in range(n):
                if not (mask & (1 << next_city)):
                    cost = distances[pos][next_city] + dp(mask | (1 << next_city), next_city)
                    min_cost = min(min_cost, cost)
            
            memo[(mask, pos)] = min_cost
            return min_cost
        
        optimal_cost = dp(1, 0)
        
        path = [0]
        mask = 1
        current_pos = 0
        
        while mask != (1 << n) - 1:
            best_next = -1
            best_cost = float('inf')
            
            for next_city in range(n):
                if not (mask & (1 << next_city)):
                    remaining_cost = memo.get((mask | (1 << next_city), next_city), float('inf'))
                    total_cost = distances[current_pos][next_city] + remaining_cost
                    
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_next = next_city
            
            path.append(best_next)
            mask |= (1 << best_next)
            current_pos = best_next
        
        path.append(0)
        return optimal_cost, path
    
    def _solve_genetic_algorithm(self, distances: List[List[float]]) -> Tuple[float, List[int]]:
        n = len(distances)
        population_size = min(self.preferences.population_size, n * 4)
        generations = min(self.preferences.genetic_generations, n * 50)
        
        def calculate_fitness(tour: List[int]) -> float:
            total_distance = 0
            for i in range(len(tour) - 1):
                total_distance += distances[tour[i]][tour[i + 1]]
            return total_distance
        
        def create_random_tour() -> List[int]:
            tour = list(range(1, n))
            random.shuffle(tour)
            return [0] + tour + [0]
        
        def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
            start = random.randint(1, n - 2)
            end = random.randint(start + 1, n - 1)
            
            child = [-1] * n
            child[start:end] = parent1[start:end]
            
            remaining = [city for city in parent2[1:-1] if city not in child[start:end]]
            
            idx = 0
            for i in range(1, n):
                if child[i] == -1:
                    child[i] = remaining[idx]
                    idx += 1
            
            return [0] + child[1:-1] + [0]
        
        def mutate(tour: List[int]) -> List[int]:
            if random.random() < 0.1:
                i, j = random.sample(range(1, n), 2)
                tour[i], tour[j] = tour[j], tour[i]
            return tour
        
        population = [create_random_tour() for _ in range(population_size)]
        
        for generation in range(generations):
            fitness_scores = [(calculate_fitness(tour), tour) for tour in population]
            fitness_scores.sort(key=lambda x: x[0])
            
            elite_size = population_size // 4
            new_population = [tour for _, tour in fitness_scores[:elite_size]]
            
            while len(new_population) < population_size:
                parent1, parent2 = random.choices(
                    [tour for _, tour in fitness_scores[:population_size//2]], 
                    k=2
                )
                child = order_crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population
        
        best_tour = min(population, key=calculate_fitness)
        return calculate_fitness(best_tour), best_tour
    
    def _solve_nearest_neighbor(self, distances: List[List[float]]) -> Tuple[float, List[int]]:
        n = len(distances)
        visited = [False] * n
        tour = [0]
        visited[0] = True
        total_distance = 0
        current_city = 0
        
        for _ in range(n - 1):
            nearest_city = -1
            nearest_distance = float('inf')
            
            for city in range(n):
                if not visited[city] and distances[current_city][city] < nearest_distance:
                    nearest_distance = distances[current_city][city]
                    nearest_city = city
            
            visited[nearest_city] = True
            tour.append(nearest_city)
            total_distance += nearest_distance
            current_city = nearest_city
        
        tour.append(0)
        total_distance += distances[current_city][0]
        
        return total_distance, tour
    
    def _improve_solution_2opt(self, tour: List[int], distances: List[List[float]]) -> Tuple[float, List[int]]:
        def calculate_tour_distance(t: List[int]) -> float:
            return sum(distances[t[i]][t[i + 1]] for i in range(len(t) - 1))
        
        best_tour = tour[:]
        best_distance = calculate_tour_distance(best_tour)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour) - 1):
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    new_distance = calculate_tour_distance(new_tour)
                    
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        tour = new_tour
                        improved = True
        
        return best_distance, best_tour
    
    def solve(self, cities: List[Tuple[float, float]]) -> Tuple[float, List[int]]:
        start_time = time.perf_counter()
        n = len(cities)
        
        if n < 2:
            return 0.0, [0] if n == 1 else []
        
        distances = self._calculate_distance_matrix(cities)
        
        algorithm = self.behavior_analyzer.get_best_algorithm(n, self.preferences.max_computation_time)
        
        try:
            if algorithm == 'dp_bitmask' and n <= self.preferences.problem_size_threshold:
                cost, path = self._solve_dp_bitmask(distances)
            elif algorithm == 'genetic':
                cost, path = self._solve_genetic_algorithm(distances)
            else:
                cost, path = self._solve_nearest_neighbor(distances)
                if self.preferences.quality_over_speed > 0.7:
                    cost, path = self._improve_solution_2opt(path, distances)
        
        except Exception:
            cost, path = self._solve_nearest_neighbor(distances)
        
        execution_time = time.perf_counter() - start_time
        self.behavior_analyzer.record_solution(algorithm, execution_time, int(cost), n)
        
        return cost, path
    
    def solve_from_distance_matrix(self, distances: List[List[float]]) -> Tuple[float, List[int]]:
        start_time = time.perf_counter()
        n = len(distances)
        
        algorithm = self.behavior_analyzer.get_best_algorithm(n, self.preferences.max_computation_time)
        
        try:
            if algorithm == 'dp_bitmask' and n <= self.preferences.problem_size_threshold:
                cost, path = self._solve_dp_bitmask(distances)
            elif algorithm == 'genetic':
                cost, path = self._solve_genetic_algorithm(distances)
            else:
                cost, path = self._solve_nearest_neighbor(distances)
                if self.preferences.quality_over_speed > 0.7:
                    cost, path = self._improve_solution_2opt(path, distances)
        
        except Exception:
            cost, path = self._solve_nearest_neighbor(distances)
        
        execution_time = time.perf_counter() - start_time
        self.behavior_analyzer.record_solution(algorithm, execution_time, int(cost), n)
        
        return cost, path

class TSPTestSuite:
    def __init__(self):
        self.test_results = []
    
    def run_comprehensive_tests(self):
        preferences = TSPPreferences()
        behavior_analyzer = UserBehaviorAnalyzer()
        solver = AdaptiveTSPSolver(preferences, behavior_analyzer)
        
        test_cases = [
            {
                'name': 'Small Instance (4 cities)',
                'cities': [(0, 0), (1, 0), (1, 1), (0, 1)]
            },
            {
                'name': 'Medium Instance (8 cities)',
                'cities': [(0, 0), (2, 0), (3, 1), (3, 3), (1, 4), (-1, 3), (-2, 1), (-1, 0)]
            },
            {
                'name': 'Larger Instance (12 cities)',
                'cities': [(i, random.uniform(0, 10)) for i in range(12)]
            }
        ]
        
        print("=== TSP Solver Test Results ===")
        
        for test in test_cases:
            print(f"\nTest: {test['name']}")
            print(f"Cities: {len(test['cities'])}")
            
            cost, path = solver.solve(test['cities'])
            
            print(f"Optimal Cost: {cost:.2f}")
            print(f"Path: {path}")
            
            self.test_results.append({
                'name': test['name'],
                'cost': cost,
                'path': path,
                'cities_count': len(test['cities'])
            })
        
        print(f"\nAlgorithm Performance Summary:")
        for algo, stats in behavior_analyzer.algorithm_performance.items():
            if stats['calls'] > 0:
                print(f"  {algo}: {stats['calls']} calls, avg time: {stats['avg_time']:.4f}s, avg quality: {stats['avg_quality']:.2f}")

def benchmark_algorithms():
    print("\n=== Algorithm Benchmark ===")
    preferences = TSPPreferences()
    behavior_analyzer = UserBehaviorAnalyzer()
    solver = AdaptiveTSPSolver(preferences, behavior_analyzer)
    
    random.seed(42)
    
    for size in [6, 10, 15]:
        cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(size)]
        
        print(f"\nBenchmarking {size}-city problem:")
        
        start_time = time.perf_counter()
        cost, path = solver.solve(cities)
        end_time = time.perf_counter()
        
        print(f"  Solution cost: {cost:.2f}")
        print(f"  Execution time: {end_time - start_time:.4f}s")
        print(f"  Path length: {len(path)}")

def demonstrate_adaptive_behavior():
    print("\n=== Adaptive Behavior Demonstration ===")
    
    preferences = TSPPreferences(quality_over_speed=0.8, max_computation_time=2.0)
    behavior_analyzer = UserBehaviorAnalyzer()
    solver = AdaptiveTSPSolver(preferences, behavior_analyzer)
    
    print("Testing adaptation over multiple problem instances...")
    
    for i in range(5):
        size = random.randint(8, 12)
        cities = [(random.uniform(0, 50), random.uniform(0, 50)) for _ in range(size)]
        
        recommended_algo = behavior_analyzer.get_best_algorithm(size, preferences.max_computation_time)
        cost, path = solver.solve(cities)
        
        print(f"Problem {i+1}: {size} cities, recommended: {recommended_algo}, cost: {cost:.2f}")

if __name__ == "__main__":
    test_suite = TSPTestSuite()
    test_suite.run_comprehensive_tests()
    benchmark_algorithms()
    demonstrate_adaptive_behavior()