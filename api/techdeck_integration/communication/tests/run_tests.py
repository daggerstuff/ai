#!/usr/bin/env python3
"""
Test Runner for Pipeline Communication System.

This script provides a comprehensive test runner for the six-stage pipeline
communication system with HIPAA++ compliance and sub-50ms performance requirements.
"""

import asyncio
import sys
import os
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner for pipeline communication system."""
    
    def __init__(self, test_directory: str = ".", verbose: bool = False, 
                 performance_mode: bool = False, security_mode: bool = False):
        self.test_directory = Path(test_directory)
        self.verbose = verbose
        self.performance_mode = performance_mode
        self.security_mode = security_mode
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests in the test directory."""
        logger.info("Starting comprehensive pipeline communication tests...")
        
        self.start_time = time.time()
        
        # Discover test files
        test_files = self._discover_test_files()
        
        if not test_files:
            logger.warning("No test files found in directory: %s", self.test_directory)
            return {'status': 'no_tests', 'test_files': [], 'results': []}
        
        logger.info("Found %d test files", len(test_files))
        
        # Run tests
        results = []
        for test_file in test_files:
            result = await self._run_test_file(test_file)
            results.append(result)
            self.test_results.append(result)
        
        self.end_time = time.time()
        
        # Generate summary
        summary = self._generate_summary(results)
        
        logger.info("Test execution completed in %.2f seconds", 
                   self.end_time - self.start_time)
        
        return summary
    
    def _discover_test_files(self) -> List[Path]:
        """Discover test files in the test directory."""
        test_files = []
        
        # Look for test files
        patterns = ['test_*.py', '*_test.py']
        
        for pattern in patterns:
            test_files.extend(self.test_directory.glob(pattern))
        
        # Sort by name for consistent execution order
        return sorted(test_files)
    
    async def _run_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Run a single test file using pytest."""
        logger.info("Running test file: %s", test_file.name)
        
        import subprocess
        import tempfile
        
        start_time = time.time()
        
        try:
            # Build pytest command
            cmd = [
                sys.executable, '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=short',
                '--asyncio-mode=auto'
            ]
            
            # Add specific markers based on mode
            if self.performance_mode:
                cmd.extend(['-m', 'performance'])
            elif self.security_mode:
                cmd.extend(['-m', 'security or hipaa'])
            
            if self.verbose:
                cmd.append('-s')  # Show print statements
            
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.test_directory)
            )
            
            end_time = time.time()
            
            # Parse results
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            # Extract test count and timing
            test_count = self._extract_test_count(output)
            execution_time = end_time - start_time
            
            logger.info("Test file %s completed in %.2f seconds - %s",
                       test_file.name, execution_time, 
                       "PASSED" if success else "FAILED")
            
            return {
                'file': str(test_file),
                'name': test_file.name,
                'success': success,
                'return_code': result.returncode,
                'execution_time': execution_time,
                'test_count': test_count,
                'output': output,
                'error_output': result.stderr if not success else None
            }
            
        except Exception as e:
            end_time = time.time()
            logger.error("Error running test file %s: %s", test_file.name, str(e))
            
            return {
                'file': str(test_file),
                'name': test_file.name,
                'success': False,
                'return_code': -1,
                'execution_time': end_time - start_time,
                'test_count': 0,
                'output': '',
                'error_output': str(e)
            }
    
    def _extract_test_count(self, output: str) -> int:
        """Extract test count from pytest output."""
        import re
        
        # Look for patterns like "X passed" or "X failed"
        patterns = [
            r'(\d+) passed',
            r'(\d+) failed',
            r'(\d+) skipped',
            r'(\d+) xfailed'
        ]
        
        total = 0
        for pattern in patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                total += int(match)
        
        return total
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate test execution summary."""
        total_tests = sum(r['test_count'] for r in results)
        total_time = sum(r['execution_time'] for r in results)
        passed_files = sum(1 for r in results if r['success'])
        failed_files = sum(1 for r in results if not r['success'])
        
        # Performance analysis
        performance_metrics = self._analyze_performance(results)
        
        # Security compliance analysis
        security_compliance = self._analyze_security_compliance(results)
        
        summary = {
            'status': 'completed',
            'total_files': len(results),
            'passed_files': passed_files,
            'failed_files': failed_files,
            'total_tests': total_tests,
            'total_execution_time': total_time,
            'overall_success': passed_files == len(results),
            'success_rate': passed_files / len(results) if results else 0.0,
            'results': results,
            'performance_metrics': performance_metrics,
            'security_compliance': security_compliance,
            'timestamp': time.time()
        }
        
        return summary
    
    def _analyze_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics from test results."""
        execution_times = [r['execution_time'] for r in results]
        
        if not execution_times:
            return {'status': 'no_data'}
        
        return {
            'status': 'analyzed',
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'total_execution_time': sum(execution_times),
            'sub_50ms_compliance': all(t < 0.05 for t in execution_times),  # 50ms threshold
            'performance_grade': self._calculate_performance_grade(execution_times)
        }
    
    def _analyze_security_compliance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security compliance from test results."""
        security_tests = 0
        hipaa_tests = 0
        security_failures = 0
        
        for result in results:
            if 'security' in result['name'].lower() or 'hipaa' in result['name'].lower():
                if 'security' in result['name'].lower():
                    security_tests += 1
                if 'hipaa' in result['name'].lower():
                    hipaa_tests += 1
                
                if not result['success']:
                    security_failures += 1
        
        return {
            'security_tests_run': security_tests,
            'hipaa_tests_run': hipaa_tests,
            'security_failures': security_failures,
            'security_compliance_rate': (security_tests - security_failures) / security_tests if security_tests > 0 else 1.0,
            'hipaa_compliance_rate': (hipaa_tests - security_failures) / hipaa_tests if hipaa_tests > 0 else 1.0,
            'overall_compliance': security_failures == 0
        }
    
    def _calculate_performance_grade(self, execution_times: List[float]) -> str:
        """Calculate performance grade based on execution times."""
        avg_time = sum(execution_times) / len(execution_times)
        
        if avg_time < 0.01:  # < 10ms
            return 'A+'
        elif avg_time < 0.05:  # < 50ms
            return 'A'
        elif avg_time < 0.1:   # < 100ms
            return 'B'
        elif avg_time < 0.5:   # < 500ms
            return 'C'
        elif avg_time < 1.0:   # < 1s
            return 'D'
        else:
            return 'F'
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary."""
        print("\n" + "="*60)
        print("PIPELINE COMMUNICATION TEST SUMMARY")
        print("="*60)
        
        print(f"Total Test Files: {summary['total_files']}")
        print(f"Passed Files: {summary['passed_files']}")
        print(f"Failed Files: {summary['failed_files']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Overall Success: {'✅ PASSED' if summary['overall_success'] else '❌ FAILED'}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        # Performance metrics
        perf = summary.get('performance_metrics', {})
        if perf.get('status') == 'analyzed':
            print(f"\nPerformance Metrics:")
            print(f"  Average Execution Time: {perf['avg_execution_time']:.3f}s")
            print(f"  Min Execution Time: {perf['min_execution_time']:.3f}s")
            print(f"  Max Execution Time: {perf['max_execution_time']:.3f}s")
            print(f"  Sub-50ms Compliance: {'✅' if perf['sub_50ms_compliance'] else '❌'}")
            print(f"  Performance Grade: {perf['performance_grade']}")
        
        # Security compliance
        security = summary.get('security_compliance', {})
        print(f"\nSecurity Compliance:")
        print(f"  Security Tests: {security.get('security_tests_run', 0)}")
        print(f"  HIPAA Tests: {security.get('hipaa_tests_run', 0)}")
        print(f"  Security Failures: {security.get('security_failures', 0)}")
        print(f"  Security Compliance Rate: {security.get('security_compliance_rate', 1.0):.1%}")
        print(f"  HIPAA Compliance Rate: {security.get('hipaa_compliance_rate', 1.0):.1%}")
        print(f"  Overall Compliance: {'✅ COMPLIANT' if security.get('overall_compliance', True) else '❌ NON-COMPLIANT'}")
        
        # Failed tests details
        failed_results = [r for r in summary['results'] if not r['success']]
        if failed_results:
            print(f"\nFailed Test Files:")
            for result in failed_results:
                print(f"  ❌ {result['name']} (Return Code: {result['return_code']})")
                if result.get('error_output'):
                    print(f"     Error: {result['error_output'][:100]}...")
        
        print("\n" + "="*60)


async def run_specific_test(test_name: str, test_directory: str = ".", 
                           verbose: bool = False) -> Dict[str, Any]:
    """Run a specific test file."""
    runner = TestRunner(test_directory, verbose)
    
    test_file = Path(test_directory) / test_name
    if not test_file.exists():
        logger.error("Test file not found: %s", test_name)
        return {'status': 'not_found', 'error': f'Test file {test_name} not found'}
    
    result = await runner._run_test_file(test_file)
    
    print(f"\nTest Results for {test_name}:")
    print(f"Success: {'✅ PASSED' if result['success'] else '❌ FAILED'}")
    print(f"Execution Time: {result['execution_time']:.2f}s")
    print(f"Test Count: {result['test_count']}")
    
    if not result['success'] and result.get('error_output'):
        print(f"Error: {result['error_output']}")
    
    return result


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for pipeline communication system'
    )
    
    parser.add_argument(
        '--test-dir', '-d',
        default='.',
        help='Directory containing test files (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--performance', '-p',
        action='store_true',
        help='Run only performance tests'
    )
    
    parser.add_argument(
        '--security', '-s',
        action='store_true',
        help='Run only security/HIPAA compliance tests'
    )
    
    parser.add_argument(
        '--specific-test', '-t',
        help='Run a specific test file'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for test results (JSON format)'
    )
    
    args = parser.parse_args()
    
    async def run_tests():
        if args.specific_test:
            result = await run_specific_test(
                args.specific_test, 
                args.test_dir, 
                args.verbose
            )
            return result
        else:
            runner = TestRunner(
                args.test_dir, 
                args.verbose, 
                args.performance, 
                args.security
            )
            summary = await runner.run_all_tests()
            runner.print_summary(summary)
            return summary
    
    # Run tests
    try:
        result = asyncio.run(run_tests())
        
        # Save results if output file specified
        if args.output and isinstance(result, dict):
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info("Test results saved to: %s", args.output)
        
        # Exit with appropriate code
        if isinstance(result, dict) and result.get('overall_success', False):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Test execution failed: %s", str(e))
        sys.exit(1)


if __name__ == '__main__':
    main()