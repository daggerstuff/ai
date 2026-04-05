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

describe('PixelatedEmpathyAPI Query Builders', () => {
    it('should correctly format query options for getConversations', async () => {
        const api = new PixelatedEmpathyAPI('fake_key');

        // Mock _makeRequest to intercept params
        let capturedParams = null;
        api._makeRequest = async (method, endpoint, options) => {
            capturedParams = options.params;
            return { data: { conversations: [] } };
        };

        await api.getConversations({
            dataset: 'test-dataset',
            tier: 'professional',
            minQuality: 0.8,
            limit: 50,
            offset: 10
        });

        expect(capturedParams).toEqual({
            limit: 50,
            offset: 10,
            dataset: 'test-dataset',
            tier: 'professional',
            min_quality: 0.8
        });
    });

    it('should handle iterConversations with empty results', async () => {
        const api = new PixelatedEmpathyAPI('fake_key');

        // Mock _makeRequest to return empty immediately
        api._makeRequest = async (method, endpoint, options) => {
            return { data: { conversations: [] } };
        };

        const iterator = api.iterConversations({ batchSize: 10 });
        const results = [];
        for await (const conv of iterator) {
            results.push(conv);
        }

        expect(results).toEqual([]);
    });
});
