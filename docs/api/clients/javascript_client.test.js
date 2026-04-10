import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PixelatedEmpathyAPI } from './javascript_client';

describe('PixelatedEmpathyAPI healthCheck', () => {
    it('should return true when health check succeeds', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        api._makeRequest = vi.fn().mockResolvedValue({ success: true });

        const isHealthy = await api.healthCheck();

        expect(api._makeRequest).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(true);
    });

    it('should return false when health check returns false success', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        api._makeRequest = vi.fn().mockResolvedValue({ success: false });

        const isHealthy = await api.healthCheck();

        expect(api._makeRequest).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });

    it('should return false when health check throws an error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        api._makeRequest = vi.fn().mockRejectedValue(new Error('Network error'));

        const isHealthy = await api.healthCheck();

        expect(api._makeRequest).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });
});

describe('PixelatedEmpathyAPI Rate Limiting', () => {
    it('should retry after 429 error and succeed', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        api._makeRequest = vi.fn()
            .mockRejectedValueOnce({ status: 429 })
            .mockResolvedValueOnce({ success: true });

        const result = await api.healthCheck();

        expect(api._makeRequest).toHaveBeenCalledTimes(2);
        expect(result).toBe(true);
    });
});

describe('PixelatedEmpathyAPI Methods', () => {
    let api;

    beforeEach(() => {
        api = new PixelatedEmpathyAPI('test_key');
        // Mock the internal request method
        api._makeRequest = vi.fn();
    });

    it('getConversations should handle pagination options correctly', async () => {
        api._makeRequest.mockResolvedValueOnce({ data: { conversations: [] } });

        await api.getConversations({ limit: 50, offset: 100, dataset: 'test_dataset' });

        expect(api._makeRequest).toHaveBeenCalledWith('GET', '/conversations', expect.objectContaining({
            params: expect.objectContaining({
                limit: 50,
                offset: 100,
                dataset: 'test_dataset'
            })
        }));
    });

    it('getConversation should call _makeRequest with correct endpoint', async () => {
        api._makeRequest.mockResolvedValueOnce({ data: { id: 'conv-123' } });

        const result = await api.getConversation('conv-123');

        expect(api._makeRequest).toHaveBeenCalledWith('GET', '/conversations/conv-123');
        expect(result).toEqual({ id: 'conv-123' });
    });
});

describe('PixelatedEmpathyAPI Method getConversations', () => {
    it('should correctly map minQuality to min_quality parameter', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let calledEndpoint = '';
        let calledOptions = {};

        api._makeRequest = async (method, endpoint, options) => {
            calledEndpoint = endpoint;
            calledOptions = options;
            return { data: { conversations: [] } };
        };

        await api.getConversations({
            limit: 50,
            offset: 10,
            dataset: 'test_dataset',
            tier: 'professional',
            minQuality: 0.8
        });

        expect(calledEndpoint).toBe('/conversations');
        expect(calledOptions.params).toEqual({
            limit: 50,
            offset: 10,
            dataset: 'test_dataset',
            tier: 'professional',
            min_quality: 0.8
        });
    });

    it('should use default limit and offset if not provided', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let calledOptions = {};
        api._makeRequest = async (method, endpoint, options) => {
            calledOptions = options;
            return { data: { conversations: [] } };
        };

        await api.getConversations();

        expect(calledOptions.params).toEqual({
            limit: 100,
            offset: 0
        });
    });
});

describe('PixelatedEmpathyAPI Method waitForJob', () => {
    it('should resolve immediately if job is already completed', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        api.getJobStatus = async (jobId) => {
            return { status: 'completed', progress: 100 };
        };

        const result = await api.waitForJob('job-123');
        expect(result).toEqual({ status: 'completed', progress: 100 });
    });

    it('should throw error if timeout is exceeded', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        api.getJobStatus = async (jobId) => {
            return { status: 'processing', progress: 50 };
        };

        const originalNow = Date.now;
        let now = 0;
        Date.now = () => now;
        api._sleep = async (ms) => {
            now += ms;
        };

        try {
            await expect(
                api.waitForJob('job-123', { timeout: 0.1, pollInterval: 0.01 }),
            ).rejects.toThrow(PixelatedEmpathyAPIError);
        } finally {
            Date.now = originalNow;
        }
    });
});

describe('PixelatedEmpathyAPI Method iterConversations', () => {
    it('should handle pagination correctly and yield items', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let callCount = 0;
        api.getConversations = async (options) => {
            callCount++;
            if (callCount === 1) {
                return { conversations: [{ id: 1 }, { id: 2 }] };
            } else if (callCount === 2) {
                return { conversations: [{ id: 3 }] };
            } else {
                return { conversations: [] };
            }
        };

        const results = [];
        for await (const conv of api.iterConversations({ batchSize: 2 })) {
            results.push(conv);
        }

        expect(results).toEqual([{ id: 1 }, { id: 2 }, { id: 3 }]);
        expect(callCount).toBe(2);
    });

    it('should exit early if batch returns 0 items', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let callCount = 0;
        api.getConversations = async (options) => {
            callCount++;
            return { conversations: [] };
        };

        const results = [];
        for await (const conv of api.iterConversations({ batchSize: 2 })) {
            results.push(conv);
        }

        expect(results).toEqual([]);
        expect(callCount).toBe(1);
    });
});

describe('PixelatedEmpathyAPI Method submitProcessingJob', () => {
    it('should correctly build job data payload', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let calledEndpoint = '';
        let calledOptions = {};

        api._makeRequest = async (method, endpoint, options) => {
            calledEndpoint = endpoint;
            calledOptions = options;
            return { data: { job_id: 'new-job-123' } };
        };

        await api.submitProcessingJob('my-dataset', 'export', { format: 'csv' });

        expect(calledEndpoint).toBe('/processing/submit');
        expect(calledOptions.data).toEqual({
            dataset_name: 'my-dataset',
            processing_type: 'export',
            parameters: { format: 'csv' }
        });
    });
});

describe('PixelatedEmpathyAPI Method exportData', () => {
    it('should map options correctly to exportData payload', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let calledOptions = {};

        api._makeRequest = async (method, endpoint, options) => {
            calledOptions = options;
            return { data: { export_url: 'http://example.com/export' } };
        };

        await api.exportData('my-dataset', { format: 'csv', tier: 'premium', minQuality: 0.9 });

        expect(calledOptions.data).toEqual({
            dataset: 'my-dataset',
            format: 'csv',
            tier: 'premium',
            min_quality: 0.9
        });
        expect(calledOptions.headers).toEqual({ 'Content-Type': 'application/x-www-form-urlencoded' });
    });
});

describe('PixelatedEmpathyAPI Method healthCheck', () => {
    it('should return true on successful request', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        api._makeRequest = async (method, endpoint) => {
            return { success: true };
        };

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(true);
    });

    it('should return false if request throws error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        api._makeRequest = async (method, endpoint) => {
            throw new Error('Network error');
        };

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(false);
    });
});

describe('PixelatedEmpathyAPI healthCheck', () => {
    it('should return true when API is healthy', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        // Mock _makeRequest to return success
        api._makeRequest = async () => ({ success: true });

        const result = await api.healthCheck();
        expect(result).toBe(true);
    });

    it('should return false when API throws an error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        // Mock _makeRequest to simulate a network or API error
        api._makeRequest = async () => {
            throw new Error('Network timeout');
        };

        const result = await api.healthCheck();
        expect(result).toBe(false);
    });
});
