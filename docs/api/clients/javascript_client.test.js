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
