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

describe('PixelatedEmpathyAPI Methods', () => {
    it('should correctly pass options in getQualityMetrics', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        // Mock _makeRequest to capture arguments and avoid network calls
        let capturedMethod, capturedEndpoint, capturedOptions;
        api._makeRequest = async (method, endpoint, options) => {
            capturedMethod = method;
            capturedEndpoint = endpoint;
            capturedOptions = options;
            return { data: { success: true } };
        };

        const result = await api.getQualityMetrics({
            dataset: 'test-dataset',
            tier: 'professional'
        });

        expect(result).toEqual({ success: true });
        expect(capturedMethod).toBe('GET');
        expect(capturedEndpoint).toBe('/quality/metrics');
        expect(capturedOptions).toEqual({
            params: {
                dataset: 'test-dataset',
                tier: 'professional'
            }
        });
    });

    it('should correctly handle getQualityMetrics without options', async () => {
        const api = new PixelatedEmpathyAPI('test_key');

        // Mock _makeRequest
        let capturedOptions;
        api._makeRequest = async (method, endpoint, options) => {
            capturedOptions = options;
            return { data: { success: true } };
        };

        await api.getQualityMetrics();

        expect(capturedOptions).toEqual({
            params: {}
        });
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
