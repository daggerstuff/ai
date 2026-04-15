import { describe, it, expect } from 'vitest';
import { PixelatedEmpathyAPI, PixelatedEmpathyAPIError, RateLimitError } from './javascript_client';

describe('PixelatedEmpathyAPI Error Handling', () => {
    it('should correctly expose error classes', () => {
        expect(typeof PixelatedEmpathyAPIError).toBe('function');
        expect(typeof RateLimitError).toBe('function');
    });

    it('should set error properties correctly in PixelatedEmpathyAPIError', () => {
        const err = new PixelatedEmpathyAPIError('Test error', 'ERR_CODE', 404);
        expect(err.message).toBe('Test error');
        expect(err.errorCode).toBe('ERR_CODE');
        expect(err.statusCode).toBe(404);
        expect(err.name).toBe('PixelatedEmpathyAPIError');
    });

    it('should set retryAfter correctly in RateLimitError', () => {
        const err = new RateLimitError(60);
        expect(err.message).toBe('Rate limit exceeded. Retry after 60 seconds.');
        expect(err.retryAfter).toBe(60);
        expect(err.name).toBe('RateLimitError');
    });
});

describe('PixelatedEmpathyAPI Constructor', () => {
    it('should initialize with default options', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.apiKey).toBe('test_key');
        expect(api.baseUrl).toBe('https://api.pixelatedempathy.com/v1');
        expect(api.timeout).toBe(30000);
        expect(api.maxRetries).toBe(3);
        expect(api.defaultHeaders).toEqual({
            Authorization: 'Bearer test_key',
            'Content-Type': 'application/json',
            'User-Agent': 'PixelatedEmpathyAPI-JavaScript/1.0.0',
        });
    });

    it('should allow overriding default options', () => {
        const api = new PixelatedEmpathyAPI('test_key', {
            baseUrl: 'https://test.api.com/v1/',
            timeout: 15000,
            maxRetries: 5,
        });
        expect(api.apiKey).toBe('test_key');
        expect(api.baseUrl).toBe('https://test.api.com/v1');
        expect(api.timeout).toBe(15000);
        expect(api.maxRetries).toBe(5);
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
        api._sleep = async (ms) => {}; // Mock sleep to avoid actual delay

        await expect(api.waitForJob('job-123', { timeout: 0.1, pollInterval: 0.01 })).rejects.toThrow(PixelatedEmpathyAPIError);
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
