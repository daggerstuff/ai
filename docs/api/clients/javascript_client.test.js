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
