#!/usr/bin/env python3
"""
CPU Optimization and Management System

Provides CPU usage optimization with:
- Process priority management
- Resource limiting for intensive operations
- CPU usage monitoring and alerts
- Automatic throttling mechanisms
- Background process management
"""

import os
import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import subprocess

@dataclass
class ProcessInfo:
    """Information about a process."""
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    status: str

class CPUOptimizer:
    """Manages CPU optimization and resource limiting."""
    
    def __init__(self, target_cpu_threshold: float = 80.0):
        self.target_cpu_threshold = target_cpu_threshold
        self.logger = logging.getLogger(__name__)
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Process management
        self.managed_processes: Dict[int, ProcessInfo] = {}
        self.cpu_intensive_patterns = [
            'snyk', 'language-server', 'webstorm', 'firefox',
            'chrome', 'node', 'npm', 'webpack'
        ]
        
        self.logger.info("CPU Optimizer initialized")
    
    def get_system_cpu_usage(self) -> float:
        """Get current system CPU usage."""
        return psutil.cpu_percent(interval=1)
    
    def get_top_cpu_processes(self, limit: int = 10) -> List[ProcessInfo]:
        """Get top CPU-consuming processes."""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                info = proc.info
                if info['cpu_percent'] > 0:
                    processes.append(ProcessInfo(
                        pid=info['pid'],
                        name=info['name'] or 'unknown',
                        cpu_percent=info['cpu_percent'],
                        memory_percent=info['memory_percent'],
                        status=info['status']
                    ))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Sort by CPU usage
        processes.sort(key=lambda x: x.cpu_percent, reverse=True)
        return processes[:limit]
    
    def identify_cpu_intensive_processes(self) -> List[ProcessInfo]:
        """Identify processes that are likely CPU-intensive development tools."""
        top_processes = self.get_top_cpu_processes()
        intensive_processes = []
        
        for proc in top_processes:
            if any(pattern in proc.name.lower() for pattern in self.cpu_intensive_patterns):
                intensive_processes.append(proc)
        
        return intensive_processes
    
    def set_process_priority(self, pid: int, priority: int) -> bool:
        """Set process priority (nice value)."""
        try:
            proc = psutil.Process(pid)
            proc.nice(priority)
            self.logger.info(f"Set process {pid} ({proc.name()}) priority to {priority}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError) as e:
            self.logger.warning(f"Failed to set priority for process {pid}: {e}")
            return False
    
    def limit_process_cpu(self, pid: int, cpu_limit: int) -> bool:
        """Limit process CPU usage using cpulimit."""
        try:
            # Check if cpulimit is available
            subprocess.run(['which', 'cpulimit'], check=True, capture_output=True)
            
            # Apply CPU limit
            cmd = ['cpulimit', '-p', str(pid), '-l', str(cpu_limit), '-b']
            subprocess.Popen(cmd)
            
            self.logger.info(f"Applied {cpu_limit}% CPU limit to process {pid}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("cpulimit not available, using nice priority instead")
            return self.set_process_priority(pid, 10)  # Lower priority
    
    def optimize_development_tools(self) -> Dict[str, bool]:
        """Optimize CPU usage of development tools."""
        results = {}
        intensive_processes = self.identify_cpu_intensive_processes()
        
        self.logger.info(f"Found {len(intensive_processes)} CPU-intensive development processes")
        
        for proc in intensive_processes:
            if proc.cpu_percent > 50:  # Very high CPU usage
                self.logger.info(f"Optimizing high CPU process: {proc.name} (PID: {proc.pid}, CPU: {proc.cpu_percent:.1f}%)")
                
                # Try to limit CPU to 30%
                success = self.limit_process_cpu(proc.pid, 30)
                results[f"{proc.name}_{proc.pid}"] = success
                
            elif proc.cpu_percent > 20:  # Moderate CPU usage
                # Just lower priority
                success = self.set_process_priority(proc.pid, 5)
                results[f"{proc.name}_{proc.pid}"] = success
        
        return results
    
    def create_cpu_friendly_environment(self):
        """Set up CPU-friendly environment variables and settings."""
        
        # Set environment variables for lower resource usage
        cpu_friendly_env = {
            'NODE_OPTIONS': '--max-old-space-size=2048',  # Limit Node.js memory
            'PYTHONOPTIMIZE': '1',  # Enable Python optimizations
            'OMP_NUM_THREADS': '2',  # Limit OpenMP threads
            'NUMBA_NUM_THREADS': '2',  # Limit Numba threads
            'MKL_NUM_THREADS': '2',  # Limit Intel MKL threads
        }
        
        for key, value in cpu_friendly_env.items():
            os.environ[key] = value
            self.logger.info(f"Set {key}={value}")
        
        # Create CPU optimization config
        config_path = Path("/home/vivi/pixelated/ai/enterprise_config/cpu_optimization.conf")
        with open(config_path, 'w') as f:
            f.write("""# CPU Optimization Configuration
# This file contains settings for CPU usage optimization

# Process priority settings
development_tools_priority=10
background_processes_priority=15
data_processing_priority=0

# CPU limits (percentage)
max_single_process_cpu=30
system_cpu_threshold=80
emergency_cpu_threshold=95

# Threading limits
max_worker_threads=2
max_parallel_processes=2

# Memory limits
max_memory_per_process=2048MB
swap_usage_threshold=50%
""")
        
        self.logger.info(f"Created CPU optimization config: {config_path}")
    
    def monitor_and_optimize(self, interval: int = 30):
        """Continuously monitor and optimize CPU usage."""
        self.logger.info(f"Starting CPU monitoring with {interval}s interval")
        
        while self.monitoring:
            try:
                cpu_usage = self.get_system_cpu_usage()
                
                if cpu_usage > self.target_cpu_threshold:
                    self.logger.warning(f"High CPU usage detected: {cpu_usage:.1f}%")
                    
                    # Optimize development tools
                    optimization_results = self.optimize_development_tools()
                    
                    optimized_count = sum(optimization_results.values())
                    self.logger.info(f"Optimized {optimized_count} processes")
                    
                    # Wait a bit longer after optimization
                    time.sleep(interval * 2)
                else:
                    self.logger.debug(f"CPU usage normal: {cpu_usage:.1f}%")
                    time.sleep(interval)
                    
            except Exception as e:
                self.logger.error(f"Error in CPU monitoring: {e}")
                time.sleep(interval)
    
    def start_monitoring(self, interval: int = 30):
        """Start background CPU monitoring."""
        if self.monitoring:
            self.logger.warning("CPU monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self.monitor_and_optimize,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        self.logger.info("CPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background CPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("CPU monitoring stopped")
    
    def get_optimization_report(self) -> Dict[str, any]:
        """Get CPU optimization report."""
        cpu_usage = self.get_system_cpu_usage()
        top_processes = self.get_top_cpu_processes(5)
        intensive_processes = self.identify_cpu_intensive_processes()
        
        return {
            'current_cpu_usage': cpu_usage,
            'cpu_status': 'high' if cpu_usage > self.target_cpu_threshold else 'normal',
            'top_processes': [
                {
                    'name': proc.name,
                    'pid': proc.pid,
                    'cpu_percent': proc.cpu_percent,
                    'memory_percent': proc.memory_percent
                }
                for proc in top_processes
            ],
            'intensive_development_tools': len(intensive_processes),
            'optimization_available': len(intensive_processes) > 0,
            'recommendations': self._get_optimization_recommendations(cpu_usage, intensive_processes)
        }
    
    def _get_optimization_recommendations(self, cpu_usage: float, intensive_processes: List[ProcessInfo]) -> List[str]:
        """Get optimization recommendations."""
        recommendations = []
        
        if cpu_usage > 90:
            recommendations.append("CRITICAL: Consider closing unnecessary applications")
            recommendations.append("Consider restarting high-CPU development tools")
        elif cpu_usage > self.target_cpu_threshold:
            recommendations.append("Consider reducing IDE extensions/plugins")
            recommendations.append("Close unused browser tabs")
        
        if len(intensive_processes) > 3:
            recommendations.append("Multiple development tools running - consider consolidating")
        
        for proc in intensive_processes:
            if proc.cpu_percent > 100:  # Multi-core usage
                recommendations.append(f"Consider restarting {proc.name} (very high CPU usage)")
        
        if not recommendations:
            recommendations.append("CPU usage is within normal limits")
        
        return recommendations

def install_cpulimit():
    """Install cpulimit if not available."""
    try:
        subprocess.run(['which', 'cpulimit'], check=True, capture_output=True)
        print("✅ cpulimit already installed")
        return True
    except subprocess.CalledProcessError:
        print("📦 Installing cpulimit...")
        try:
            subprocess.run(['sudo', 'apt', 'update'], check=True, capture_output=True)
            subprocess.run(['sudo', 'apt', 'install', '-y', 'cpulimit'], check=True, capture_output=True)
            print("✅ cpulimit installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install cpulimit: {e}")
            return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔧 CPU OPTIMIZATION SYSTEM")
    print("=" * 50)
    
    # Install cpulimit if needed
    install_cpulimit()
    
    # Initialize optimizer
    optimizer = CPUOptimizer(target_cpu_threshold=75.0)
    
    # Set up CPU-friendly environment
    optimizer.create_cpu_friendly_environment()
    
    # Get current status
    report = optimizer.get_optimization_report()
    
    print(f"📊 Current CPU usage: {report['current_cpu_usage']:.1f}%")
    print(f"🎯 CPU status: {report['cpu_status']}")
    print(f"🔧 Development tools using high CPU: {report['intensive_development_tools']}")
    
    print("\n🔝 Top CPU processes:")
    for proc in report['top_processes']:
        print(f"   {proc['name']} (PID: {proc['pid']}): {proc['cpu_percent']:.1f}% CPU")
    
    print("\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"   • {rec}")
    
    # Apply optimizations if needed
    if report['optimization_available']:
        print("\n🚀 Applying CPU optimizations...")
        results = optimizer.optimize_development_tools()
        optimized = sum(results.values())
        print(f"✅ Optimized {optimized} processes")
        
        # Show new CPU usage
        time.sleep(3)
        new_cpu = optimizer.get_system_cpu_usage()
        print(f"📈 CPU usage after optimization: {new_cpu:.1f}%")
    
    print("\n✅ CPU optimization complete!")
    print("💡 Run with --monitor to start continuous monitoring")
