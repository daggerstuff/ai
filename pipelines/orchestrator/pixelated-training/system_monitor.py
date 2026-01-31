#!/usr/bin/env python3
"""
System Monitoring - GPU/Memory/Disk tracking during training
"""

import json
import threading
import time
from datetime import datetime

import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None

class SystemMonitor:
    def __init__(self, log_file="training_monitor.jsonl", interval=30):
        self.log_file = log_file
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """Start monitoring in background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                stats = self._collect_stats()
                self._log_stats(stats)
                time.sleep(self.interval)
            except Exception:
                time.sleep(self.interval)

    def _collect_stats(self):
        """Collect system statistics"""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage("/").percent,
                "disk_free_gb": psutil.disk_usage("/").free / (1024**3)
            }
        }

        # GPU stats if available
        if GPUtil:
            try:
                gpus = GPUtil.getGPUs()
                gpu_stats = []
                for gpu in gpus:
                    gpu_stats.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_used_mb": gpu.memoryUsed,
                        "memory_total_mb": gpu.memoryTotal,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature,
                        "load_percent": gpu.load * 100
                    })
                stats["gpu"] = gpu_stats
            except:
                stats["gpu"] = []

        return stats

    def _log_stats(self, stats):
        """Log statistics to file"""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")

        # Print alerts for critical conditions
        self._check_alerts(stats)

    def _check_alerts(self, stats):
        """Check for critical conditions"""
        system = stats["system"]

        # Memory alert
        if system["memory_percent"] > 90:
            pass

        # Disk space alert
        if system["disk_usage_percent"] > 90:
            pass

        # GPU alerts
        if "gpu" in stats:
            for gpu in stats["gpu"]:
                if gpu["memory_percent"] > 95:
                    pass
                if gpu["temperature"] > 85:
                    pass

# Global monitor instance
_monitor = None

def start_monitoring(log_file="training_monitor.jsonl", interval=30):
    """Start system monitoring"""
    global _monitor
    if _monitor is None:
        _monitor = SystemMonitor(log_file, interval)
    _monitor.start()

def stop_monitoring():
    """Stop system monitoring"""
    global _monitor
    if _monitor:
        _monitor.stop()

if __name__ == "__main__":
    # Standalone monitoring
    monitor = SystemMonitor()
    monitor.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
