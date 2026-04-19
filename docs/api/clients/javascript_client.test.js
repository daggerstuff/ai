import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
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

    afterEach(() => {
        vi.restoreAllMocks();
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
