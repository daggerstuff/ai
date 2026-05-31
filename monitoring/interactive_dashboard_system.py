<script>
    // Auto-refresh functionality
    let refreshInterval;

    function showRefreshIndicator() {
        document.getElementById('refreshIndicator').classList.add('show');
    }

    function hideRefreshIndicator() {
        document.getElementById('refreshIndicator').classList.remove('show');
    }

    function updateLastRefreshTime() {
        document.getElementById('lastUpdate').textContent =
            'Last updated: ' + new Date().toLocaleTimeString();
    }

    function startAutoRefresh(intervalMs = 30000) {
        refreshInterval = setInterval(() => {
            // ⚡ Bolt: Prevent unnecessary background polling and re-renders when tab is inactive
            if (document.hidden) return;
            // ⚡ Bolt: Trigger immediate refresh when tab becomes visible again
            document.addEventListener('visibilitychange', function() {
                if (!document.hidden) {
                    refreshDashboardData();
                }
            });
            showRefreshIndicator();
            refreshDashboardData();
        }, intervalMs);
    }

    function refreshDashboardData() {
        // This will be overridden by specific dashboard implementations
        setTimeout(() => {
            updateLastRefreshTime();
            hideRefreshIndicator();
        }, 1000);
    }

    // Handle tab visibility changes - refresh immediately when tab becomes visible again
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            // Tab became visible, refresh data immediately to avoid stale data
            refreshDashboardData();
        }
    });

    // Start auto-refresh when page loads
    document.addEventListener('DOMContentLoaded', function() {
        startAutoRefresh();
    });

    // Chart.js default configuration
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.position = 'top';
    Chart.defaults.plugins.title.display = true;
    Chart.defaults.plugins.title.font = {
        size: 16,
        weight: 'bold'
    };
</script>