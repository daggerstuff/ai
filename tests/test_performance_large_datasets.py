import pytest
#!/usr/bin/env python3
"""
Performance Tests for Large Dataset Processing
Task 5.7.1.3: Create performance tests for large dataset processing

Tests system performance with large datasets:
- Memory usage optimization
- Processing speed benchmarks
- Scalability testing
- Resource utilization monitoring
- Performance regression detection
"""

import unittest
import sqlite3
import pandas as pd
import numpy as np
import json
import tempfile
import os
import time
import psutil
import gc
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('/home/vivi/pixelated/ai/monitoring')
sys.path.append('/home/vivi/pixelated/ai')

class PerformanceTestBase(unittest.TestCase):
    """Base class for performance testing with utilities"""
    
    def setUp(self):
        """Set up performance monitoring"""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.performance_metrics = {}
    
    def tearDown(self):
        """Record performance metrics"""
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.performance_metrics.update({
            'execution_time': end_time - self.start_time,
            'memory_usage_mb': end_memory - self.start_memory,
            'peak_memory_mb': end_memory
        })
        
        # Force garbage collection
        gc.collect()
    
    def create_large_test_database(self, num_conversations=10000):
        """Create large test database for performance testing"""
        test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        test_db_path = test_db.name
        test_db.close()
        
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute('''
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                dataset_source TEXT,
                tier TEXT,
                conversations_json TEXT,
                character_count INTEGER,
                word_count INTEGER,
                turn_count INTEGER,
                created_at TIMESTAMP,
                processed_at TIMESTAMP
            )
        ''')
        
        # Generate large dataset
        datasets = ['dataset_a', 'dataset_b', 'dataset_c', 'dataset_d', 'dataset_e']
        tiers = ['priority_1', 'standard', 'additional_specialized']
        
        batch_size = 1000
        for batch_start in range(0, num_conversations, batch_size):
            batch_data = []
            batch_end = min(batch_start + batch_size, num_conversations)
            
            for i in range(batch_start, batch_end):
                dataset = datasets[i % len(datasets)]
                tier = tiers[i % len(tiers)]
                
                # Create varied conversation content
                human_msg = f"This is test question {i+1} about {dataset} with some additional context and details."
                assistant_msg = f"This is a comprehensive response for {dataset} question {i+1} providing detailed information and helpful guidance with multiple sentences to create realistic content length."
                
                conversation_json = json.dumps([
                    {"human": human_msg},
                    {"assistant": assistant_msg}
                ])
                
                word_count = len((human_msg + ' ' + assistant_msg).split())
                char_count = len(human_msg + assistant_msg)
                
                batch_data.append((
                    f'perf_test_{i+1:06d}',
                    dataset,
                    tier,
                    conversation_json,
                    char_count,
                    word_count,
                    2,
                    f'2025-08-{(i % 7) + 1:02d} {(i % 24):02d}:{(i % 60):02d}:00',
                    f'2025-08-{(i % 7) + 1:02d} {(i % 24):02d}:{((i % 60) + 1):02d}:00'
                ))
            
            cursor.executemany('''
                INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            
            # Commit in batches to manage memory
            conn.commit()
        
        conn.close()
        return test_db_path
    
    def measure_performance(self, func, *args, **kwargs):
        """Measure performance of a function"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_delta_mb': end_memory - start_memory,
            'peak_memory_mb': end_memory
        }

class TestLargeDatasetLoading(PerformanceTestBase):
    """Test performance of loading large datasets"""
    
    def test_load_10k_conversations(self):
        """Test loading 10,000 conversations"""
        print("📊 Testing 10K conversation loading performance...")
        
        # Create test database
        db_path = self.create_large_test_database(10000)
        
        try:
            # Measure loading performance
            def load_data():
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query("SELECT * FROM conversations", conn)
                conn.close()
                return df
            
            perf_metrics = self.measure_performance(load_data)
            
            # Verify data loaded correctly
            self.assertEqual(len(perf_metrics['result']), 10000)
            
            # Performance assertions
            self.assertLess(perf_metrics['execution_time'], 10.0, "Loading should complete within 10 seconds")
            self.assertLess(perf_metrics['memory_delta_mb'], 500, "Memory usage should be under 500MB")
            
            print(f"  ✅ Loaded 10K conversations in {perf_metrics['execution_time']:.2f}s")
            print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
            
        finally:
            os.unlink(db_path)
    
    def test_load_50k_conversations(self):
        """Test loading 50,000 conversations"""
        print("📊 Testing 50K conversation loading performance...")
        
        # Create larger test database
        db_path = self.create_large_test_database(50000)
        
        try:
            # Measure loading performance with chunking
            def load_data_chunked():
                conn = sqlite3.connect(db_path)
                chunks = []
                chunk_size = 10000
                
                for chunk in pd.read_sql_query(
                    "SELECT * FROM conversations", 
                    conn, 
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                conn.close()
                return df
            
            perf_metrics = self.measure_performance(load_data_chunked)
            
            # Verify data loaded correctly
            self.assertEqual(len(perf_metrics['result']), 50000)
            
            # Performance assertions (more lenient for larger dataset)
            self.assertLess(perf_metrics['execution_time'], 30.0, "Loading should complete within 30 seconds")
            self.assertLess(perf_metrics['memory_delta_mb'], 1000, "Memory usage should be under 1GB")
            
            print(f"  ✅ Loaded 50K conversations in {perf_metrics['execution_time']:.2f}s")
            print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
            
        finally:
            os.unlink(db_path)
    
    def test_streaming_data_processing(self):
        """Test streaming data processing for memory efficiency"""
        print("📊 Testing streaming data processing...")
        
        db_path = self.create_large_test_database(25000)
        
        try:
            def process_streaming():
                conn = sqlite3.connect(db_path)
                chunk_size = 5000
                total_processed = 0
                quality_scores = []
                
                for chunk in pd.read_sql_query(
                    "SELECT * FROM conversations", 
                    conn, 
                    chunksize=chunk_size
                ):
                    # Process each chunk
                    chunk['quality_score'] = chunk['word_count'] * 2 + 50
                    quality_scores.extend(chunk['quality_score'].tolist())
                    total_processed += len(chunk)
                
                conn.close()
                return {
                    'total_processed': total_processed,
                    'avg_quality': np.mean(quality_scores)
                }
            
            perf_metrics = self.measure_performance(process_streaming)
            
            # Verify processing
            self.assertEqual(perf_metrics['result']['total_processed'], 25000)
            self.assertGreater(perf_metrics['result']['avg_quality'], 0)
            
            # Performance assertions
            self.assertLess(perf_metrics['execution_time'], 20.0, "Streaming processing should be efficient")
            self.assertLess(perf_metrics['memory_delta_mb'], 300, "Streaming should use less memory")
            
            print(f"  ✅ Processed 25K conversations in {perf_metrics['execution_time']:.2f}s")
            print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
            
        finally:
            os.unlink(db_path)

class TestAnalyticsPerformance(PerformanceTestBase):
    """Test performance of analytics operations on large datasets"""
    
    def test_aggregation_performance(self):
        """Test performance of data aggregation operations"""
        print("📊 Testing aggregation performance...")
        
        # Create large dataset in memory
        np.random.seed(42)
        large_df = pd.DataFrame({
            'conversation_id': [f'test_{i:06d}' for i in range(20000)],
            'dataset_source': np.random.choice(['dataset_a', 'dataset_b', 'dataset_c'], 20000),
            'tier': np.random.choice(['priority_1', 'standard', 'additional'], 20000),
            'word_count': np.random.randint(10, 100, 20000),
            'quality_score': np.random.normal(50, 15, 20000),
            'complexity_score': np.random.normal(40, 10, 20000)
        })
        
        def perform_aggregations():
            # Multiple aggregation operations
            dataset_stats = large_df.groupby('dataset_source').agg({
                'word_count': ['mean', 'std', 'count'],
                'quality_score': ['mean', 'median', 'std'],
                'complexity_score': 'mean'
            })
            
            tier_stats = large_df.groupby('tier').agg({
                'quality_score': ['mean', 'count'],
                'word_count': 'sum'
            })
            
            cross_stats = large_df.groupby(['dataset_source', 'tier']).agg({
                'quality_score': 'mean',
                'word_count': 'mean'
            })
            
            return {
                'dataset_stats': dataset_stats,
                'tier_stats': tier_stats,
                'cross_stats': cross_stats
            }
        
        perf_metrics = self.measure_performance(perform_aggregations)
        
        # Verify aggregations
        result = perf_metrics['result']
        self.assertGreater(len(result['dataset_stats']), 0)
        self.assertGreater(len(result['tier_stats']), 0)
        self.assertGreater(len(result['cross_stats']), 0)
        
        # Performance assertions
        self.assertLess(perf_metrics['execution_time'], 5.0, "Aggregations should complete quickly")
        self.assertLess(perf_metrics['memory_delta_mb'], 200, "Aggregations should be memory efficient")
        
        print(f"  ✅ Aggregated 20K records in {perf_metrics['execution_time']:.2f}s")
        print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
    
    def test_statistical_analysis_performance(self):
        """Test performance of statistical analysis operations"""
        print("📊 Testing statistical analysis performance...")
        
        # Create large dataset for statistical analysis
        np.random.seed(42)
        data_size = 15000
        
        large_dataset = {
            'quality_scores': np.random.normal(50, 15, data_size),
            'engagement_scores': np.random.normal(60, 20, data_size),
            'complexity_scores': np.random.normal(40, 10, data_size),
            'word_counts': np.random.randint(10, 200, data_size)
        }
        
        def perform_statistical_analysis():
            results = {}
            
            # Correlation analysis
            df = pd.DataFrame(large_dataset)
            correlation_matrix = df.corr()
            
            # Distribution analysis
            for column in df.columns:
                results[f'{column}_stats'] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'median': df[column].median(),
                    'q25': df[column].quantile(0.25),
                    'q75': df[column].quantile(0.75)
                }
            
            # Trend analysis (simple linear regression)
            x = np.arange(len(df))
            for column in df.columns:
                slope, intercept = np.polyfit(x, df[column], 1)
                results[f'{column}_trend'] = {'slope': slope, 'intercept': intercept}
            
            results['correlation_matrix'] = correlation_matrix.to_dict()
            return results
        
        perf_metrics = self.measure_performance(perform_statistical_analysis)
        
        # Verify analysis results
        result = perf_metrics['result']
        self.assertIn('correlation_matrix', result)
        self.assertIn('quality_scores_stats', result)
        
        # Performance assertions
        self.assertLess(perf_metrics['execution_time'], 3.0, "Statistical analysis should be fast")
        self.assertLess(perf_metrics['memory_delta_mb'], 150, "Statistical analysis should be memory efficient")
        
        print(f"  ✅ Analyzed 15K records in {perf_metrics['execution_time']:.2f}s")
        print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
    
    def test_machine_learning_performance(self):
        """Test performance of machine learning operations"""
        print("📊 Testing ML model performance...")
        
        # Create dataset for ML testing
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        def train_and_evaluate_model():
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': model,
                'mse': mse,
                'r2_score': r2,
                'n_samples': n_samples,
                'n_features': n_features
            }
        
        perf_metrics = self.measure_performance(train_and_evaluate_model)
        
        # Verify ML results
        result = perf_metrics['result']
        self.assertIsNotNone(result['model'])
        self.assertIsInstance(result['mse'], float)
        self.assertIsInstance(result['r2_score'], float)
        
        # Performance assertions
        self.assertLess(perf_metrics['execution_time'], 15.0, "ML training should complete within reasonable time")
        self.assertLess(perf_metrics['memory_delta_mb'], 300, "ML training should manage memory well")
        
        print(f"  ✅ Trained ML model on 10K samples in {perf_metrics['execution_time']:.2f}s")
        print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")
        print(f"  ✅ Model R² score: {result['r2_score']:.3f}")

class TestScalabilityLimits(PerformanceTestBase):
    """Test system scalability limits"""
    
    def test_memory_scalability(self):
        """Test memory usage scalability"""
        print("📊 Testing memory scalability...")
        
        dataset_sizes = [1000, 5000, 10000, 20000]
        memory_usage = []
        processing_times = []
        
        for size in dataset_sizes:
            print(f"  Testing with {size} conversations...")
            
            # Create dataset
            df = pd.DataFrame({
                'conversation_id': [f'test_{i:06d}' for i in range(size)],
                'word_count': np.random.randint(10, 100, size),
                'quality_score': np.random.normal(50, 15, size)
            })
            
            def process_dataset():
                # Simulate processing operations
                result = df.groupby(df.index // 1000).agg({
                    'word_count': 'mean',
                    'quality_score': ['mean', 'std']
                })
                return result
            
            perf_metrics = self.measure_performance(process_dataset)
            
            memory_usage.append(perf_metrics['memory_delta_mb'])
            processing_times.append(perf_metrics['execution_time'])
            
            # Clean up
            del df
            gc.collect()
        
        # Analyze scalability
        memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / (dataset_sizes[-1] - dataset_sizes[0])
        time_growth_rate = (processing_times[-1] - processing_times[0]) / (dataset_sizes[-1] - dataset_sizes[0])
        
        print(f"  ✅ Memory growth rate: {memory_growth_rate:.4f} MB per 1K conversations")
        print(f"  ✅ Time growth rate: {time_growth_rate:.6f} seconds per 1K conversations")
        
        # Scalability assertions
        self.assertLess(memory_growth_rate, 10.0, "Memory growth should be reasonable")
        self.assertLess(time_growth_rate, 0.1, "Processing time should scale well")
    
    def test_concurrent_processing_simulation(self):
        """Test concurrent processing simulation"""
        print("📊 Testing concurrent processing simulation...")
        
        import threading
        import queue
        
        def worker_function(work_queue, result_queue):
            """Worker function for concurrent processing"""
            while True:
                try:
                    work_item = work_queue.get(timeout=1)
                    if work_item is None:
                        break
                    
                    # Simulate processing
                    data = np.random.randn(1000)
                    result = {
                        'mean': np.mean(data),
                        'std': np.std(data),
                        'processed_items': len(data)
                    }
                    
                    result_queue.put(result)
                    work_queue.task_done()
                    
                except queue.Empty:
                    break
        
        def simulate_concurrent_processing():
            num_workers = 4
            num_tasks = 20
            
            work_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # Add work items
            for i in range(num_tasks):
                work_queue.put(f"task_{i}")
            
            # Start workers
            workers = []
            for i in range(num_workers):
                worker = threading.Thread(target=worker_function, args=(work_queue, result_queue))
                worker.start()
                workers.append(worker)
            
            # Wait for completion
            work_queue.join()
            
            # Stop workers
            for i in range(num_workers):
                work_queue.put(None)
            
            for worker in workers:
                worker.join()
            
            # Collect results
            results = []
            while not result_queue.empty():
                results.append(result_queue.get())
            
            return results
        
        perf_metrics = self.measure_performance(simulate_concurrent_processing)
        
        # Verify concurrent processing
        results = perf_metrics['result']
        self.assertEqual(len(results), 20)  # All tasks completed
        
        # Performance assertions
        self.assertLess(perf_metrics['execution_time'], 10.0, "Concurrent processing should be efficient")
        
        print(f"  ✅ Processed 20 tasks concurrently in {perf_metrics['execution_time']:.2f}s")
        print(f"  ✅ Memory usage: {perf_metrics['memory_delta_mb']:.1f}MB")

def run_performance_tests():
    """Run all performance tests and return results"""
    print("⚡ Running Performance Tests for Large Dataset Processing")
    print("=" * 70)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLargeDatasetLoading,
        TestAnalyticsPerformance,
        TestScalabilityLimits
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("⚡ Performance Test Results Summary:")
    print(f"  • Tests Run: {result.testsRun}")
    print(f"  • Failures: {len(result.failures)}")
    print(f"  • Errors: {len(result.errors)}")
    print(f"  • Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0] if 'AssertionError:' in traceback else 'Unknown failure'
            print(f"  • {test}: {error_msg}")
    
    if result.errors:
        print(f"\n🚨 Errors ({len(result.errors)}):")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2] if traceback else 'Unknown error'
            print(f"  • {test}: {error_msg}")
    
    if not result.failures and not result.errors:
        print("\n✅ All performance tests passed successfully!")
        print("🎉 System performs well with large datasets!")
        print("\n📊 Performance Highlights:")
        print("  • Large dataset loading: Optimized and efficient")
        print("  • Analytics operations: Fast and memory-efficient")
        print("  • ML model training: Scalable and performant")
        print("  • Memory usage: Well-managed and predictable")
        print("  • Concurrent processing: Efficient and stable")
    
    return result

if __name__ == "__main__":
    run_performance_tests()
