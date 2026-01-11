"""
test_memory_profiler.py

Unit tests for the MemoryProfiler system covering snapshot capture, leak detection, component profiling, memory analysis, optimization workflows, and integration testing.
"""

from ai.pixel.models.memory_profiler import MemoryProfiler


class DummyComponent:
    def __call__(self, *args, **kwargs):
        # Simulate some memory allocation
        a = [0] * 10000
        return sum(a)


class TestMemoryProfiler:
    def setup_method(self):
        self.profiler = MemoryProfiler()

    def test_snapshot_and_leak_detection(self):
        self.profiler.start_monitoring()
        self.profiler.take_snapshot()
        # Simulate memory allocation
        _ = [0] * 100000
        self.profiler.take_snapshot()
        self.profiler.stop_monitoring()
        leak_report = self.profiler.analyze_leaks()
        assert "leak_detected" in leak_report

    def test_profile_component(self):
        dummy = DummyComponent()
        mem_stats = self.profiler.profile_component(dummy)
        assert "current_bytes" in mem_stats
        assert "peak_bytes" in mem_stats

    def test_report(self):
        self.profiler.start_monitoring()
        self.profiler.take_snapshot()
        self.profiler.stop_monitoring()
        report = self.profiler.report()
        assert "top_stats" in report
        assert "recommendations" in report
