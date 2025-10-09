#!/usr/bin/env python3
"""
Stress Testing Framework
Tests system performance under heavy load with large datasets.
"""

import time
import threading
import multiprocessing
import psycopg2
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StressTestFramework:
    """Framework for stress testing the conversation processing system."""
    
    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "port": "5433",
            "user": "postgres",
            "password": "postgres",
            "database": "pixelated_empathy"
        }
        self.test_results = []
    
    def generate_test_conversation(self, conv_id: str) -> Dict[str, Any]:
        """Generate a realistic test conversation."""
        conversation_templates = [
            {
                "category": "anxiety",
                "messages": [
                    {"role": "user", "content": "I've been feeling really anxious lately about work."},
                    {"role": "assistant", "content": "I understand that work anxiety can be overwhelming. Can you tell me more about what specifically is causing you stress?"},
                    {"role": "user", "content": "My manager keeps giving me impossible deadlines and I feel like I'm constantly behind."},
                    {"role": "assistant", "content": "That sounds very stressful. Let's explore some strategies for managing these work pressures and communicating with your manager."}
                ]
            },
            {
                "category": "depression",
                "messages": [
                    {"role": "user", "content": "I've been feeling really down and unmotivated for weeks now."},
                    {"role": "assistant", "content": "Thank you for sharing that with me. Depression can make everything feel more difficult. How has this been affecting your daily activities?"},
                    {"role": "user", "content": "I can barely get out of bed some days, and I've lost interest in things I used to enjoy."},
                    {"role": "assistant", "content": "Those are common symptoms of depression. It's important that you're reaching out for support. Have you considered speaking with a mental health professional?"}
                ]
            },
            {
                "category": "relationships",
                "messages": [
                    {"role": "user", "content": "My partner and I have been arguing a lot lately and I don't know what to do."},
                    {"role": "assistant", "content": "Relationship conflicts can be really challenging. What do you think might be contributing to these arguments?"},
                    {"role": "user", "content": "We seem to have different priorities and we're not communicating well."},
                    {"role": "assistant", "content": "Communication is key in relationships. Let's talk about some strategies for improving how you and your partner discuss important topics."}
                ]
            }
        ]
        
        template = random.choice(conversation_templates)
        
        return {
            "id": conv_id,
            "metadata": {
                "source_dataset": "stress_test",
                "tier": random.randint(1, 3),
                "category": template["category"],
                "quality_score": random.uniform(0.7, 1.0)
            },
            "conversation": template["messages"]
        }
    
    def insert_conversation_batch(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Insert a batch of conversations and measure performance."""
        start_time = time.time()
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            conversations_inserted = 0
            messages_inserted = 0
            
            for conv in conversations:
                # Insert conversation
                cursor.execute("""
                    INSERT INTO conversations (id, source, tier, category, quality_score)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    conv["id"],
                    conv["metadata"]["source_dataset"],
                    f"TIER_{conv['metadata']['tier']}",
                    conv["metadata"]["category"],
                    conv["metadata"]["quality_score"]
                ))
                
                if cursor.rowcount > 0:
                    conversations_inserted += 1
                
                # Insert messages
                for i, msg in enumerate(conv["conversation"]):
                    cursor.execute("""
                        INSERT INTO messages (id, conversation_id, role, content, word_count)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        f"{conv['id']}_msg_{i}",
                        conv["id"],
                        msg["role"],
                        msg["content"],
                        len(msg["content"].split())
                    ))
                    
                    if cursor.rowcount > 0:
                        messages_inserted += 1
            
            conn.commit()
            cursor.close()
            conn.close()
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": True,
                "duration": duration,
                "conversations_inserted": conversations_inserted,
                "messages_inserted": messages_inserted,
                "conversations_per_second": conversations_inserted / duration if duration > 0 else 0,
                "messages_per_second": messages_inserted / duration if duration > 0 else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def concurrent_insert_test(self, total_conversations: int, batch_size: int, num_threads: int) -> Dict[str, Any]:
        """Test concurrent conversation insertion."""
        logger.info(f"Starting concurrent insert test: {total_conversations} conversations, {batch_size} batch size, {num_threads} threads")
        
        # Generate all test data
        all_conversations = []
        for i in range(total_conversations):
            conv = self.generate_test_conversation(f"stress_test_{i}_{int(time.time())}")
            all_conversations.append(conv)
        
        # Split into batches
        batches = []
        for i in range(0, len(all_conversations), batch_size):
            batch = all_conversations[i:i + batch_size]
            batches.append(batch)
        
        # Execute batches concurrently
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.insert_conversation_batch, batch) for batch in batches]
            
            for future in futures:
                result = future.result()
                results.append(result)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Aggregate results
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        total_conversations_inserted = sum(r.get("conversations_inserted", 0) for r in successful_results)
        total_messages_inserted = sum(r.get("messages_inserted", 0) for r in successful_results)
        
        return {
            "total_duration": total_duration,
            "total_conversations_inserted": total_conversations_inserted,
            "total_messages_inserted": total_messages_inserted,
            "overall_conversations_per_second": total_conversations_inserted / total_duration if total_duration > 0 else 0,
            "overall_messages_per_second": total_messages_inserted / total_duration if total_duration > 0 else 0,
            "successful_batches": len(successful_results),
            "failed_batches": len(failed_results),
            "batch_results": results
        }
    
    def query_performance_test(self, num_queries: int, concurrent_queries: int) -> Dict[str, Any]:
        """Test database query performance under load."""
        logger.info(f"Starting query performance test: {num_queries} queries, {concurrent_queries} concurrent")
        
        test_queries = [
            "SELECT COUNT(*) FROM conversations",
            "SELECT source, COUNT(*) FROM conversations GROUP BY source",
            "SELECT c.id, COUNT(m.id) as msg_count FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id GROUP BY c.id LIMIT 100",
            "SELECT AVG(quality_score) FROM conversations WHERE tier = 'TIER_1'",
            "SELECT role, AVG(word_count) FROM messages GROUP BY role"
        ]
        
        def execute_query_batch():
            results = []
            try:
                conn = psycopg2.connect(**self.db_config)
                cursor = conn.cursor()
                
                for _ in range(num_queries // concurrent_queries):
                    query = random.choice(test_queries)
                    start_time = time.time()
                    cursor.execute(query)
                    cursor.fetchall()
                    duration = time.time() - start_time
                    results.append(duration)
                
                cursor.close()
                conn.close()
                
            except Exception as e:
                logger.error(f"Query batch failed: {e}")
                
            return results
        
        # Execute queries concurrently
        start_time = time.time()
        all_results = []
        
        with ThreadPoolExecutor(max_workers=concurrent_queries) as executor:
            futures = [executor.submit(execute_query_batch) for _ in range(concurrent_queries)]
            
            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if all_results:
            return {
                "total_duration": total_duration,
                "total_queries": len(all_results),
                "queries_per_second": len(all_results) / total_duration if total_duration > 0 else 0,
                "avg_query_time": statistics.mean(all_results),
                "median_query_time": statistics.median(all_results),
                "min_query_time": min(all_results),
                "max_query_time": max(all_results),
                "p95_query_time": statistics.quantiles(all_results, n=20)[18] if len(all_results) > 20 else max(all_results)
            }
        else:
            return {"error": "No successful queries"}
    
    def memory_usage_test(self, dataset_size: int) -> Dict[str, Any]:
        """Test memory usage with large datasets."""
        logger.info(f"Starting memory usage test with {dataset_size} conversations")
        
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate large dataset in memory
        start_time = time.time()
        conversations = []
        
        for i in range(dataset_size):
            conv = self.generate_test_conversation(f"memory_test_{i}")
            conversations.append(conv)
            
            # Check memory every 1000 conversations
            if i % 1000 == 0 and i > 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                logger.info(f"Generated {i} conversations, memory usage: {current_memory:.1f} MB")
        
        generation_time = time.time() - start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        
        # Clear memory
        conversations.clear()
        
        return {
            "dataset_size": dataset_size,
            "generation_time": generation_time,
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": peak_memory - baseline_memory,
            "memory_per_conversation_kb": (peak_memory - baseline_memory) * 1024 / dataset_size if dataset_size > 0 else 0
        }
    
    def run_comprehensive_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive stress tests."""
        logger.info("🚀 STARTING COMPREHENSIVE STRESS TESTS")
        
        results = {}
        
        # Test 1: Small concurrent insert
        logger.info("\n📊 Test 1: Small Concurrent Insert (1000 conversations)")
        results["small_concurrent_insert"] = self.concurrent_insert_test(
            total_conversations=1000,
            batch_size=100,
            num_threads=5
        )
        
        # Test 2: Query performance under load
        logger.info("\n📊 Test 2: Query Performance Under Load")
        results["query_performance"] = self.query_performance_test(
            num_queries=500,
            concurrent_queries=10
        )
        
        # Test 3: Memory usage test
        logger.info("\n📊 Test 3: Memory Usage Test")
        results["memory_usage"] = self.memory_usage_test(dataset_size=5000)
        
        # Test 4: Large concurrent insert (if system can handle it)
        logger.info("\n📊 Test 4: Large Concurrent Insert (5000 conversations)")
        results["large_concurrent_insert"] = self.concurrent_insert_test(
            total_conversations=5000,
            batch_size=200,
            num_threads=8
        )
        
        # Cleanup test data
        self.cleanup_test_data()
        
        return results
    
    def cleanup_test_data(self):
        """Clean up test data from database."""
        logger.info("🧹 Cleaning up test data...")
        
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Delete test messages
            cursor.execute("DELETE FROM messages WHERE conversation_id LIKE 'stress_test_%' OR conversation_id LIKE 'memory_test_%'")
            messages_deleted = cursor.rowcount
            
            # Delete test conversations
            cursor.execute("DELETE FROM conversations WHERE id LIKE 'stress_test_%' OR id LIKE 'memory_test_%'")
            conversations_deleted = cursor.rowcount
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Cleanup completed: {conversations_deleted} conversations, {messages_deleted} messages deleted")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

def main():
    """Main stress test runner."""
    tester = StressTestFramework()
    results = tester.run_comprehensive_stress_test()
    
    # Print summary
    logger.info("\n🎯 STRESS TEST SUMMARY:")
    
    for test_name, result in results.items():
        logger.info(f"\n📈 {test_name.replace('_', ' ').title()}:")
        
        if "error" in result:
            logger.error(f"  ❌ Failed: {result['error']}")
        else:
            if "overall_conversations_per_second" in result:
                logger.info(f"  📊 Conversations/sec: {result['overall_conversations_per_second']:.1f}")
            if "queries_per_second" in result:
                logger.info(f"  📊 Queries/sec: {result['queries_per_second']:.1f}")
            if "avg_query_time" in result:
                logger.info(f"  ⏱️ Avg query time: {result['avg_query_time']*1000:.1f}ms")
            if "memory_increase_mb" in result:
                logger.info(f"  💾 Memory increase: {result['memory_increase_mb']:.1f} MB")

if __name__ == "__main__":
    main()
