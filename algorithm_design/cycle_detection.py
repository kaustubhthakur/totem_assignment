from typing import Dict, List, Tuple, Set
from collections import defaultdict
import time

class UserProfile:
    def __init__(self):
        self.graph_access_patterns = {'sparse_graphs': 0.7, 'dense_graphs': 0.3}
        self.performance_history = []
        self.algorithm_preferences = {'dfs_success_rate': 0.85, 'tarjan_success_rate': 0.78}
    
    def update_performance(self, execution_time: float, algorithm_used: str):
        self.performance_history.append({'time': execution_time, 'algorithm': algorithm_used})
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
    
    def get_avg_performance(self, algorithm: str) -> float:
        relevant_runs = [run['time'] for run in self.performance_history if run['algorithm'] == algorithm]
        return sum(relevant_runs) / len(relevant_runs) if relevant_runs else 0.1

class AdaptiveCycleDetector:
    def __init__(self, user_profile: UserProfile):
        self.user_profile = user_profile
        self.graph_cache = {}
        self.algorithm_stats = {'dfs_calls': 0, 'tarjan_calls': 0}
        
    def _select_algorithm(self, graph: Dict[int, List[int]]) -> str:
        node_count = len(graph)
        edge_count = sum(len(neighbors) for neighbors in graph.values())
        density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
        
        sparse_preference = self.user_profile.graph_access_patterns.get('sparse_graphs', 0.5)
        dfs_performance = self.user_profile.get_avg_performance('dfs')
        tarjan_performance = self.user_profile.get_avg_performance('tarjan')
        
        if density < 0.3 and sparse_preference > 0.6:
            return 'dfs'
        elif dfs_performance > 0 and tarjan_performance > 0:
            return 'dfs' if dfs_performance < tarjan_performance else 'tarjan'
        else:
            return 'dfs' if sparse_preference > 0.5 else 'tarjan'
    
    def _detect_cycle_dfs(self, graph: Dict[int, List[int]]) -> Tuple[bool, List[int]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = defaultdict(lambda: WHITE)
        parent = {}
        cycle_path = []
        
        def dfs_visit(node: int, path: List[int]) -> bool:
            colors[node] = GRAY
            
            for neighbor in graph.get(node, []):
                if colors[neighbor] == WHITE:
                    parent[neighbor] = node
                    if dfs_visit(neighbor, path + [node]):
                        return True
                elif colors[neighbor] == GRAY:
                    cycle_start = path.index(neighbor) if neighbor in path else len(path)
                    cycle_path.extend(path[cycle_start:] + [node, neighbor])
                    return True
            
            colors[node] = BLACK
            return False
        
        for node in graph:
            if colors[node] == WHITE:
                if dfs_visit(node, []):
                    return True, cycle_path
        
        return False, []
    
    def _detect_cycle_tarjan(self, graph: Dict[int, List[int]]) -> Tuple[bool, List[int]]:
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = set()
        cycles = []
        
        def strongconnect(node: int):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif neighbor in on_stack:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                
                if len(component) > 1:
                    cycles.append(component)
        
        for node in graph:
            if node not in index:
                strongconnect(node)
        
        return (True, cycles[0]) if cycles else (False, [])
    
    def detect_cycle(self, graph: Dict[int, List[int]]) -> Tuple[bool, List[int]]:
        start_time = time.perf_counter()
        
        graph_signature = str(sorted((k, sorted(v)) for k, v in graph.items()))
        if graph_signature in self.graph_cache:
            return self.graph_cache[graph_signature]
        
        algorithm = self._select_algorithm(graph)
        
        if algorithm == 'dfs':
            result = self._detect_cycle_dfs(graph)
            self.algorithm_stats['dfs_calls'] += 1
        else:
            result = self._detect_cycle_tarjan(graph)
            self.algorithm_stats['tarjan_calls'] += 1
        
        execution_time = time.perf_counter() - start_time
        self.user_profile.update_performance(execution_time, algorithm)
        self.graph_cache[graph_signature] = result
        
        return result
    
    def get_statistics(self) -> Dict:
        return {
            'cache_size': len(self.graph_cache),
            'algorithm_usage': self.algorithm_stats,
            'performance_history_length': len(self.user_profile.performance_history)
        }

def test_cycle_detector():
    user_profile = UserProfile()
    detector = AdaptiveCycleDetector(user_profile)
    
    test_cases = [
        {
            'name': 'Simple Cycle',
            'graph': {0: [1], 1: [2], 2: [0]}
        },
        {
            'name': 'No Cycle - Tree',
            'graph': {0: [1, 2], 1: [3, 4], 2: [5]}
        },
        {
            'name': 'Complex Graph with Cycle',
            'graph': {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: [1]}
        },
        {
            'name': 'Self Loop',
            'graph': {0: [0]}
        },
        {
            'name': 'Empty Graph',
            'graph': {}
        }
    ]
    
    print("=== Cycle Detection Test Results ===")
    for test in test_cases:
        has_cycle, cycle = detector.detect_cycle(test['graph'])
        print(f"{test['name']}: Cycle={'Yes' if has_cycle else 'No'}")
        if has_cycle and cycle:
            print(f"  Cycle path: {cycle}")
        print()
    
    print("Algorithm Statistics:", detector.get_statistics())

if __name__ == "__main__":
    test_cycle_detector()