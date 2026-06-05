import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
    PixelatedEmpathyAPI,
    PixelatedEmpathyAPIError,
    RateLimitError,
    type RequestOptions,
} from './javascript_client';

describe('PixelatedEmpathyAPI healthCheck', () => {
    it('should return true when health check succeeds', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        makeRequestSpy.mockResolvedValue({ success: true });

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(true);
    });

    it('should return false when health check returns false success', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        makeRequestSpy.mockResolvedValue({ success: false });

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });

    it('should return false when health check throws an error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        makeRequestSpy.mockRejectedValue(new Error('Network error'));

        const isHealthy = await api.healthCheck();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/health');
        expect(isHealthy).toBe(false);
    });
});

describe('PixelatedEmpathyAPI Rate Limiting', () => {
    it('should retry after 429 error and succeed', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const httpRequestSpy = vi.spyOn(api, 'httpRequest');
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
        const httpRequestSpy = vi.spyOn(api, 'httpRequest');
        httpRequestSpy
            .mockResolvedValue({ statusCode: 429, headers: { 'retry-after': '0' }, body: '' });
        const makeRequest = api.makeRequest.bind(api);

        await expect(makeRequest('GET', '/test')).rejects.toThrow(RateLimitError);
        expect(httpRequestSpy).toHaveBeenCalledTimes(3);
    });
});

describe('PixelatedEmpathyAPI Methods', () => {
    it('getConversations should handle pagination options correctly', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
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

    it('getConversation should call makeRequest with correct endpoint', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        makeRequestSpy.mockResolvedValueOnce({ data: { id: 'conv-123' } });

        const result = await api.getConversation('conv-123');

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/conversations/conv-123');
        expect(result).toEqual({ id: 'conv-123' });
    });
});

describe('PixelatedEmpathyAPI Method getConversations', () => {
    it('should correctly map minQuality to min_quality parameter', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledEndpoint = '';
        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledEndpoint = endpoint;
            calledOptions = options ?? {};
            return { data: { conversations: [] } };
        });

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
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledOptions: Record<string, unknown> = {};
        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledOptions = options ?? {};
            return { data: { conversations: [] } };
        });

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

        api.getJobStatus = async (jobId: string) => {
            return { status: 'completed', progress: 100 };
        };

        const result = await api.waitForJob('job-123');
        expect(result).toEqual({ status: 'completed', progress: 100 });
    });

    it('should poll until job is completed', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        let callCount = 0;

        api.getJobStatus = async (jobId: string) => {
            callCount++;
            if (callCount === 1) return { status: 'processing', progress: 50 };
            return { status: 'completed', progress: 100 };
        };

        api.sleep = async (ms: number) => {};

        const result = await api.waitForJob('job-123', { timeout: 10, pollInterval: 0.01 });
        expect(result).toEqual({ status: 'completed', progress: 100 });
        expect(callCount).toBe(2);
    });

    it('should throw error if timeout is exceeded', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        api.getJobStatus = async (jobId: string) => {
            return { status: 'processing', progress: 50 };
        };

        const originalNow = Date.now;
        let now = 0;
        Date.now = () => now;
        api.sleep = async (ms: number) => {
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
        api.getConversations = async (options?: Record<string, unknown>) => {
            callCount++;
            if (callCount === 1) {
                return { conversations: [{ id: 1 }, { id: 2 }] };
            } else if (callCount === 2) {
                return { conversations: [{ id: 3 }] };
            } else {
                return { conversations: [] };
            }
        };

        const results: Array<Record<string, unknown>> = [];
        for await (const conv of api.iterConversations({ batchSize: 2 })) {
            results.push(conv);
        }

        expect(results).toEqual([{ id: 1 }, { id: 2 }, { id: 3 }]);
        expect(callCount).toBe(2);
    });

    it('should exit early if batch returns 0 items', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        let callCount = 0;
        api.getConversations = async (options?: Record<string, unknown>) => {
            callCount++;
            return { conversations: [] };
        };

        const results: Array<Record<string, unknown>> = [];
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
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledEndpoint = '';
        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledEndpoint = endpoint;
            calledOptions = options ?? {};
            return { data: { job_id: 'new-job-123' } };
        });

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
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledOptions = options ?? {};
            return { data: { export_url: 'http://example.com/export' } };
        });

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


describe('PixelatedEmpathyAPI Dataset Methods', () => {
    it('listDatasets should call makeRequest with correct endpoint and return datasets', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        const mockDatasets = [{ name: 'test_1', conversations: 10 }, { name: 'test_2', conversations: 20 }];
        makeRequestSpy.mockResolvedValue({ data: { datasets: mockDatasets } });

        const result = await api.listDatasets();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets');
        expect(result).toEqual(mockDatasets);
    });

    it('listDatasets should return empty array if datasets is missing from response', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        makeRequestSpy.mockResolvedValue({ data: {} });

        const result = await api.listDatasets();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets');
        expect(result).toEqual([]);
    });

    it('getDatasetInfo should call makeRequest with correct endpoint', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        const mockDatasetInfo = { name: 'test_dataset', count: 100 };
        makeRequestSpy.mockResolvedValue({ data: mockDatasetInfo });

        const result = await api.getDatasetInfo('test_dataset');

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/datasets/test_dataset');
        expect(result).toEqual(mockDatasetInfo);
    });
});


describe('PixelatedEmpathyAPI Method searchConversations', () => {
    it('should correctly pass query and default options to makeRequest', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledEndpoint = '';
        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledEndpoint = endpoint;
            calledOptions = options ?? {};
            return { data: { results: ['conversation1'] } };
        });

        const result = await api.searchConversations('anxiety');

        expect(calledEndpoint).toBe('/search');
        expect(calledOptions.data).toEqual({
            query: 'anxiety',
            filters: {},
            limit: 100,
            offset: 0,
        });
        expect(result).toEqual({ results: ['conversation1'] });
    });

    it('should correctly merge provided options', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        let calledOptions: Record<string, unknown> = {};
        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledOptions = options ?? {};
            return { data: { results: [] } };
        });

        await api.searchConversations('depression', {
            filters: { tier: 'professional' },
            limit: 10,
            offset: 20,
        });

        expect(calledOptions.data).toEqual({
            query: 'depression',
            filters: { tier: 'professional' },
            limit: 10,
            offset: 20,
        });
    });
});

describe('PixelatedEmpathyAPI Method getQualityMetrics', () => {
    it('should correctly map options to params', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledEndpoint = '';
        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledEndpoint = endpoint;
            calledOptions = options ?? {};
            return { data: { overall_statistics: {} } };
        });

        await api.getQualityMetrics({
            dataset: 'test_dataset',
            tier: 'professional',
        });

        expect(calledEndpoint).toBe('/quality/metrics');
        expect(calledOptions.params).toEqual({
            dataset: 'test_dataset',
            tier: 'professional',
        });
    });

    it('should handle missing options gracefully', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        let calledEndpoint = '';
        let calledOptions: Record<string, unknown> = {};

        makeRequestSpy.mockImplementation(async (method: string, endpoint: string, options?: RequestOptions) => {
            calledEndpoint = endpoint;
            calledOptions = options ?? {};
            return { data: { overall_statistics: {} } };
        });

        await api.getQualityMetrics();

        expect(calledEndpoint).toBe('/quality/metrics');
        expect(calledOptions.params).toEqual({});
    });
});
describe('PixelatedEmpathyAPI Method healthCheck', () => {
    it('should return true on successful request', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        makeRequestSpy.mockResolvedValue({ success: true });

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(true);
    });

    it('should return false if request throws error', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');

        makeRequestSpy.mockRejectedValue(new Error('Network error'));

        const isHealthy = await api.healthCheck();
        expect(isHealthy).toBe(false);
    });
});


describe('PixelatedEmpathyAPI Method validateConversationQuality', () => {
    it('should correctly pass conversation data to makeRequest', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        const mockResult = { quality_score: 0.95 };
        makeRequestSpy.mockResolvedValue({ data: mockResult });

        const testConversation = { id: 'test-1', text: 'Hello' };
        const result = await api.validateConversationQuality(testConversation);

        expect(makeRequestSpy).toHaveBeenCalledWith('POST', '/quality/validate', {
            data: testConversation
        });
        expect(result).toEqual(mockResult);
    });
});


describe('PixelatedEmpathyAPI Method getStatisticsOverview', () => {
    it('should correctly call makeRequest and return stats overview', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        const mockStats = { total_conversations: 2500000, datasets: 15 };
        makeRequestSpy.mockResolvedValue({ data: mockStats });

        const result = await api.getStatisticsOverview();

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/statistics/overview');
        expect(result).toEqual(mockStats);
    });
});

describe('PixelatedEmpathyAPI Method getJobStatus', () => {
    it('should correctly call makeRequest with jobId', async () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const makeRequestSpy = vi.spyOn(api, 'makeRequest');
        const mockStatus = { status: 'completed' };
        makeRequestSpy.mockResolvedValue({ data: mockStatus });

        const result = await api.getJobStatus('job-123');

        expect(makeRequestSpy).toHaveBeenCalledWith('GET', '/processing/jobs/job-123');
        expect(result).toEqual(mockStatus);
    });
});

describe('PixelatedEmpathyAPI Method safeParseResponse', () => {
    it('should return parsed object if valid JSON object is passed', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const jsonStr = '{"success": true, "data": {"id": 1}}';
        expect(api.safeParseResponse(jsonStr)).toEqual({ success: true, data: { id: 1 } });
    });

    it('should return error object if string is not valid JSON', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const invalidJsonStr = '<html><body>error</body></html>';
        expect(api.safeParseResponse(invalidJsonStr)).toEqual({ success: false, message: 'Invalid JSON response' });
    });

    it('should return error object if parsed JSON is not a plain object (e.g. array)', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const jsonStr = '[{"id": 1}]';
        expect(api.safeParseResponse(jsonStr)).toEqual({ success: false, message: 'Invalid JSON response' });
    });
});



describe('PixelatedEmpathyAPI Method toRecordArray edge cases', () => {
    it('should correctly handle undefined by returning empty array', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.toRecordArray(undefined)).toEqual([]);
    });
});

describe('PixelatedEmpathyAPI Method formatValue edge cases', () => {
    it('should correctly format undefined and null', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.formatValue(undefined)).toBe('');
        expect(api.formatValue(null)).toBe('');
    });

    it('should correctly format strings', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.formatValue('test string')).toBe('test string');
        expect(api.formatValue('')).toBe('');
    });

    it('should correctly format primitives (number, boolean, bigint)', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.formatValue(123)).toBe('123');
        expect(api.formatValue(0)).toBe('0');
        expect(api.formatValue(true)).toBe('true');
        expect(api.formatValue(false)).toBe('false');
        expect(api.formatValue(BigInt(9007199254740991))).toBe('9007199254740991');
    });

    it('should correctly format standard objects and arrays via JSON.stringify', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        expect(api.formatValue({ a: 1, b: 'test' })).toBe('{"a":1,"b":"test"}');
        expect(api.formatValue([1, 2, 3])).toBe('[1,2,3]');
    });

    it('should correctly format Symbols', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        // JSON.stringify(Symbol(...)) returns undefined, not a string
        expect(api.formatValue(Symbol('my-symbol'))).toBeUndefined();
        expect(api.formatValue(Symbol())).toBeUndefined();
    });

    it('should correctly format circular objects', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const circularObj: any = { a: 1 };
        circularObj.self = circularObj;
        expect(api.formatValue(circularObj)).toBe('[object Object]');
    });

    it('should correctly format functions', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const myFunc = function() { return 42; };
        // JSON.stringify(function) returns undefined
        expect(api.formatValue(myFunc)).toBeUndefined();
    });
});

describe('PixelatedEmpathyAPI Method safeParseResponse edge cases', () => {
    it('should return error object if parsed JSON is null', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const jsonStr = 'null';
        expect(api.safeParseResponse(jsonStr)).toEqual({ success: false, message: 'Invalid JSON response' });
    });

    it('should return error object if parsed JSON is primitive string', () => {
        const api = new PixelatedEmpathyAPI('test_key');
        const jsonStr = '"just a string"';
        expect(api.safeParseResponse(jsonStr)).toEqual({ success: false, message: 'Invalid JSON response' });
    });
});
