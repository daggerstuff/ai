import { describe, it, expect, vi } from 'vitest';
import { PixelatedEmpathyAPI, PixelatedEmpathyAPIError, RateLimitError } from './javascript_client';

describe('PixelatedEmpathyAPI healthCheck', () => {
    it('should return true when health check succeeds', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValue({ success: true });

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(true);
    });

    it('should return false when health check returns false success', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValue({ success: false });

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });

    it('should return false when health check throws an error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockRejectedValue(new Error('Network error'));

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });
});

describe('PixelatedEmpathyAPI Rate Limiting', () => {
    it('should retry after 429 error and succeed', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const httpRequestSpy = vi.spyOn(api, '_httpRequest');
        httpRequestSpy
            .mockResolvedValueOnce({ statusCode: 429, headers: { 'retry-after': '0' }, body: '' })
            .mockResolvedValueOnce({ statusCode: 200, headers: {}, body: '{"success": true}' });

        const result = await api.healthCheck();

        expect(httpRequestSpy).toHaveBeenCalledTimes(2);
        expect(result).toBe(true);
    });

    it('should throw RateLimitError when retries exceed maxRetries', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        api.maxRetries = 2;
        const httpRequestSpy = vi.spyOn(api, '_httpRequest');
        httpRequestSpy
            .mockResolvedValue({ statusCode: 429, headers: { 'retry-after': '0' }, body: '' });
        const makeRequest = api._makeRequest.bind(api);

        await expect(makeRequest('GET', '/test')).rejects.toThrow(RateLimitError);
        expect(httpRequestSpy).toHaveBeenCalledTimes(3);
    });
});

describe('PixelatedEmpathyAPI Methods', () => {
    it('getConversations should handle pagination options correctly', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValueOnce({ data: { conversations: [] } });

        await api.getConversations({ limit: 50, offset: 100, dataset: 'test_dataset' });

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/conversations', {
            params: {
                limit: 50,
                offset: 100,
                dataset: 'test_dataset'
            }
        });
    });

    it('getConversation should call _makeRequest with correct endpoint', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValueOnce({ data: { id: 'conv-123' } });

        const result = await api.getConversation('conv-123');

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/conversations/conv-123');
        expect(result).toEqual({ id: 'conv-123' });
    });
});

describe('PixelatedEmpathyAPI Method getConversations', () => {
    it('should correctly map minQuality to min_quality parameter', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { conversations: [] } });

        await api.getConversations({
            limit: 50,
            offset: 10,
            dataset: 'test_dataset',
            tier: 'professional',
            minQuality: 0.8
        });

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/conversations', expect.objectContaining({
            params: {
                limit: 50,
                offset: 10,
                dataset: 'test_dataset',
                tier: 'professional',
                min_quality: 0.8
            }
        }));
    });

    it('should use default limit and offset if not provided', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { conversations: [] } });

        await api.getConversations();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/conversations', expect.objectContaining({
            params: {
                limit: 100,
                offset: 0
            }
        }));
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
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { job_id: 'new-job-123' } });

        await api.submitProcessingJob('my-dataset', 'export', { format: 'csv' });

        expect(makeRequestSpy).toHaveBeenCalledWith('POST', '/processing/submit', expect.objectContaining({
            data: {
                dataset_name: 'my-dataset',
                processing_type: 'export',
                parameters: { format: 'csv' }
            }
        }));
    });
});

describe('PixelatedEmpathyAPI Method exportData', () => {
    it('should map options correctly to exportData payload', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { export_url: 'http://example.com/export' } });

        await api.exportData('my-dataset', { format: 'csv', tier: 'premium', minQuality: 0.9 });

        expect(makeRequestSpy).toHaveBeenCalledWith('POST', '/export', expect.objectContaining({
            data: {
                dataset: 'my-dataset',
                format: 'csv',
                tier: 'premium',
                min_quality: 0.9
            },
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        }));
    });
});


describe('PixelatedEmpathyAPI Dataset Methods', () => {
    it('listDatasets should call _makeRequest with correct endpoint and return datasets', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        const mockDatasets = [{ name: 'test_1', conversations: 10 }, { name: 'test_2', conversations: 20 }];
        makeRequestSpy.mockResolvedValue({ data: { datasets: mockDatasets } });

        const result = await api.listDatasets();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets');
        expect(result).toEqual(mockDatasets);
    });

    it('listDatasets should return empty array if datasets is missing from response', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValue({ data: {} });

        const result = await api.listDatasets();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets');
        expect(result).toEqual([]);
    });

    it('getDatasetInfo should call _makeRequest with correct endpoint', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        const mockDatasetInfo = { name: 'test_dataset', count: 100 };
        makeRequestSpy.mockResolvedValue({ data: mockDatasetInfo });

        const result = await api.getDatasetInfo('test_dataset');

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets/test_dataset');
        expect(result).toEqual(mockDatasetInfo);
    });
});


describe('PixelatedEmpathyAPI Method searchConversations', () => {
    it('should correctly pass query and default options to _makeRequest', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { results: ['conversation1'] } });

        const result = await api.searchConversations('anxiety');

        expect(makeRequestSpy).toHaveBeenCalledWith('POST', '/search', expect.objectContaining({
            data: {
                query: 'anxiety',
                filters: {},
                limit: 100,
                offset: 0,
            }
        }));
        expect(result).toEqual({ results: ['conversation1'] });
    });

    it('should correctly merge provided options', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        const makeRequestSpy = vi.spyOn(api, '_makeRequest');
        makeRequestSpy.mockResolvedValue({ data: { results: [] } });

        await api.searchConversations('depression', {
            filters: { tier: 'professional' },
            limit: 10,
            offset: 20,
        });

        expect(makeRequestSpy).toHaveBeenCalledWith('POST', '/search', expect.objectContaining({
            data: {
                query: 'depression',
                filters: { tier: 'professional' },
                limit: 10,
                offset: 20,
            }
        }));
    });
});

describe('PixelatedEmpathyAPI Method getQualityMetrics', () => {
    it('should correctly map options to params', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { overall_statistics: {} } });

        await api.getQualityMetrics({
            dataset: 'test_dataset',
            tier: 'professional',
        });

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/quality/metrics', expect.objectContaining({
            params: {
                dataset: 'test_dataset',
                tier: 'professional',
            }
        }));
    });

    it('should handle missing options gracefully', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ data: { overall_statistics: {} } });

        await api.getQualityMetrics();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/quality/metrics', { params: {} });
    });
});
describe('PixelatedEmpathyAPI Method healthCheck', () => {
    it('should return true on successful request', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockResolvedValue({ success: true });

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(true);
    });

    it('should return false if request throws error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, '_makeRequest');

        makeRequestSpy.mockRejectedValue(new Error('Network error'));

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(false);
    });
});
