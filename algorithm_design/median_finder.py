import heapq
import statistics
from typing import List, Dict, Optional
import time

class StreamAnalyzer:
    def __init__(self):
        self.stream_patterns = {
            'avg_insertion_rate': 100,
            'typical_range': (-1000, 1000),
            'distribution_type': 'normal'
        }
        self.performance_metrics = []
    
    def update_pattern(self, value: int, operation_time: float):
        self.performance_metrics.append(operation_time)
        if len(self.performance_metrics) > 1000:
            self.performance_metrics.pop(0)
    
    def get_avg_operation_time(self) -> float:
        return statistics.mean(self.performance_metrics) if self.performance_metrics else 0.001

class AdaptiveMedianFinder:
    def __init__(self, stream_analyzer: StreamAnalyzer):
        self.stream_analyzer = stream_analyzer
        self.max_heap = []
        self.min_heap = []
        self.total_elements = 0
        self.rebalance_frequency = 10
        self.optimization_mode = 'balanced'
        
    def _determine_optimization_mode(self):
        avg_time = self.stream_analyzer.get_avg_operation_time()
        if avg_time > 0.01:
            self.optimization_mode = 'memory_efficient'
            self.rebalance_frequency = max(5, self.rebalance_frequency // 2)
        elif avg_time < 0.001:
            self.optimization_mode = 'speed_optimized' 
            self.rebalance_frequency = min(50, self.rebalance_frequency * 2)
        else:
            self.optimization_mode = 'balanced'
    
    def _maintain_heap_property(self):
        if len(self.max_heap) > len(self.min_heap) + 1:
            value = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, value)
        elif len(self.min_heap) > len(self.max_heap) + 1:
            value = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -value)
    
    def add_number(self, num: int):
        start_time = time.perf_counter()
        
        if not self.max_heap and not self.min_heap:
            heapq.heappush(self.max_heap, -num)
        elif not self.min_heap:
            if num <= -self.max_heap[0]:
                heapq.heappush(self.max_heap, -num)
            else:
                heapq.heappush(self.min_heap, num)
        elif not self.max_heap:
            if num >= self.min_heap[0]:
                heapq.heappush(self.min_heap, num)
            else:
                heapq.heappush(self.max_heap, -num)
        else:
            median_estimate = self._get_current_median()
            if num <= median_estimate:
                heapq.heappush(self.max_heap, -num)
            else:
                heapq.heappush(self.min_heap, num)
        
        self.total_elements += 1
        
        if self.total_elements % self.rebalance_frequency == 0:
            self._maintain_heap_property()
            self._determine_optimization_mode()
        
        operation_time = time.perf_counter() - start_time
        self.stream_analyzer.update_pattern(num, operation_time)
    
    def _get_current_median(self) -> float:
        if not self.max_heap and not self.min_heap:
            return 0.0
        
        max_size = len(self.max_heap)
        min_size = len(self.min_heap)
        
        if max_size == min_size:
            if max_size == 0:
                return 0.0
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        elif max_size > min_size:
            return float(-self.max_heap[0])
        else:
            return float(self.min_heap[0])
    
    def find_median(self) -> float:
        return self._get_current_median()
    
    def get_stream_statistics(self) -> Dict:
        return {
            'total_elements': self.total_elements,
            'max_heap_size': len(self.max_heap),
            'min_heap_size': len(self.min_heap),
            'current_median': self.find_median(),
            'optimization_mode': self.optimization_mode,
            'rebalance_frequency': self.rebalance_frequency,
            'avg_operation_time': self.stream_analyzer.get_avg_operation_time()
        }
    
    def bulk_insert(self, numbers: List[int]):
        for num in numbers:
            self.add_number(num)
    
    def reset(self):
        self.max_heap.clear()
        self.min_heap.clear()
        self.total_elements = 0

class MedianFinderBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_performance(self, test_data: List[int], finder: AdaptiveMedianFinder) -> Dict:
        start_time = time.perf_counter()
        medians = []
        
        for num in test_data:
            finder.add_number(num)
            medians.append(finder.find_median())
        
        total_time = time.perf_counter() - start_time
        
        return {
            'total_time': total_time,
            'avg_time_per_operation': total_time / len(test_data),
            'final_median': medians[-1] if medians else 0,
            'median_sequence': medians[:10]
        }

def test_median_finder():
    stream_analyzer = StreamAnalyzer()
    median_finder = AdaptiveMedianFinder(stream_analyzer)
    benchmark = MedianFinderBenchmark()
    
    test_cases = [
        {
            'name': 'Sequential Numbers',
            'data': list(range(1, 21))
        },
        {
            'name': 'Random Numbers',
            'data': [64, 25, 12, 22, 11, 90, 5, 77, 30, 55]
        },
        {
            'name': 'Duplicates',
            'data': [1, 1, 2, 2, 3, 3, 4, 4]
        },
        {
            'name': 'Negative Numbers',
            'data': [-10, -5, 0, 5, 10, -3, 7, -8]
        },
        {
            'name': 'Large Stream',
            'data': list(range(100, 0, -1))
        }
    ]
    
    print("=== Adaptive Median Finder Test Results ===")
    
    for test in test_cases:
        median_finder.reset()
        print(f"\nTest: {test['name']}")
        print(f"Input: {test['data'][:10]}{'...' if len(test['data']) > 10 else ''}")
        
        performance = benchmark.benchmark_performance(test['data'], median_finder)
        stats = median_finder.get_stream_statistics()
        
        print(f"Final Median: {performance['final_median']}")
        print(f"Median Sequence: {performance['median_sequence']}")
        print(f"Performance: {performance['avg_time_per_operation']:.6f}s per operation")
        print(f"Optimization Mode: {stats['optimization_mode']}")
        print(f"Heap Sizes - Max: {stats['max_heap_size']}, Min: {stats['min_heap_size']}")
    
    print(f"\nOverall Statistics:")
    final_stats = median_finder.get_stream_statistics()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

def demonstrate_stream_processing():
    print("\n=== Stream Processing Demonstration ===")
    stream_analyzer = StreamAnalyzer()
    median_finder = AdaptiveMedianFinder(stream_analyzer)
    
    import random
    random.seed(42)
    
    print("Processing continuous stream...")
    for i in range(100):
        num = random.randint(-100, 100)
        median_finder.add_number(num)
        
        if i % 20 == 0:
            stats = median_finder.get_stream_statistics()
            print(f"Step {i}: Added {num}, Median: {stats['current_median']:.2f}, Mode: {stats['optimization_mode']}")

if __name__ == "__main__":
    test_median_finder()
    demonstrate_stream_processing()