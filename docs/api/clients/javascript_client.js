/**
 * Pixelated Empathy AI - JavaScript/Node.js Client Library
 * Task 51: Complete API Documentation
 * 
 * Official JavaScript client for the Pixelated Empathy AI API.
 */

const https = require('https');
const http = require('http');
const { URL } = require('url');

/**
 * Custom error class for API errors
 */
class PixelatedEmpathyAPIError extends Error {
    constructor(message, errorCode = null, statusCode = null) {
        super(message);
        this.name = 'PixelatedEmpathyAPIError';
        this.errorCode = errorCode;
        this.statusCode = statusCode;
    }
}

/**
 * Custom error class for rate limit errors
 */
class RateLimitError extends PixelatedEmpathyAPIError {
    constructor(retryAfter) {
        super(`Rate limit exceeded. Retry after ${retryAfter} seconds.`);
        this.name = 'RateLimitError';
        this.retryAfter = retryAfter;
    }
}

/**
 * Official JavaScript client for the Pixelated Empathy AI API.
 * 
 * Provides access to 2.59M+ therapeutic conversations with enterprise-grade
 * quality validation, real-time processing, and advanced search capabilities.
 */
class PixelatedEmpathyAPI {
    /**
     * Initialize the API client.
     * 
     * @param {string} apiKey - Your API key from https://api.pixelatedempathy.com
     * @param {Object} options - Configuration options
     * @param {string} options.baseUrl - API base URL (default: production)
     * @param {number} options.timeout - Request timeout in milliseconds
     * @param {number} options.maxRetries - Maximum number of retries for failed requests
     */
    constructor(apiKey, options = {}) {
        this.apiKey = apiKey;
        this.baseUrl = (options.baseUrl || 'https://api.pixelatedempathy.com/v1').replace(/\/$/, '');
        this.timeout = options.timeout || 30000;
        this.maxRetries = options.maxRetries || 3;
        
        this.defaultHeaders = {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'User-Agent': 'PixelatedEmpathyAPI-JavaScript/1.0.0'
        };
    }

    /**
     * Make an HTTP request with error handling and retries
     * @private
     */
    async _makeRequest(method, endpoint, options = {}) {
        const url = new URL(endpoint, this.baseUrl);
        
        // Add query parameters
        if (options.params) {
            Object.keys(options.params).forEach(key => {
                if (options.params[key] !== undefined && options.params[key] !== null) {
                    url.searchParams.append(key, options.params[key]);
                }
            });
        }

        const requestOptions = {
            method: method.toUpperCase(),
            headers: { ...this.defaultHeaders, ...options.headers },
            timeout: this.timeout
        };

        // Add request body
        if (options.data) {
            if (options.headers && options.headers['Content-Type'] === 'application/x-www-form-urlencoded') {
                requestOptions.body = new URLSearchParams(options.data).toString();
            } else {
                requestOptions.body = JSON.stringify(options.data);
            }
        }

        for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
            try {
                const response = await this._httpRequest(url, requestOptions);
                
                // Handle rate limiting
                if (response.statusCode === 429) {
                    const retryAfter = parseInt(response.headers['retry-after'] || '60');
                    if (attempt < this.maxRetries) {
                        console.warn(`Rate limited. Retrying after ${retryAfter} seconds...`);
                        await this._sleep(retryAfter * 1000);
                        continue;
                    } else {
                        throw new RateLimitError(retryAfter);
                    }
                }

                // Parse response
                let data;
                try {
                    data = JSON.parse(response.body);
                } catch (e) {
                    data = { success: false, message: 'Invalid JSON response' };
                }

                // Handle API errors
                if (response.statusCode >= 400) {
                    const errorMessage = data.error?.message || 'Unknown error';
                    const errorCode = data.error?.code || 'UNKNOWN_ERROR';
                    throw new PixelatedEmpathyAPIError(errorMessage, errorCode, response.statusCode);
                }

                return {
                    success: data.success || false,
                    data: data.data,
                    message: data.message || '',
                    timestamp: data.timestamp || '',
                    error: data.error
                };

            } catch (error) {
                if (error instanceof PixelatedEmpathyAPIError || error instanceof RateLimitError) {
                    throw error;
                }

                if (attempt < this.maxRetries) {
                    console.warn(`Request failed (attempt ${attempt + 1}): ${error.message}`);
                    await this._sleep(Math.pow(2, attempt) * 1000); // Exponential backoff
                    continue;
                } else {
                    throw new PixelatedEmpathyAPIError(`Request failed: ${error.message}`);
                }
            }
        }
    }

    /**
     * Make HTTP request using Node.js built-in modules
     * @private
     */
    _httpRequest(url, options) {
        return new Promise((resolve, reject) => {
            const client = url.protocol === 'https:' ? https : http;
            
            const req = client.request(url, {
                method: options.method,
                headers: options.headers,
                timeout: options.timeout
            }, (res) => {
                let body = '';
                res.on('data', chunk => body += chunk);
                res.on('end', () => {
                    resolve({
                        statusCode: res.statusCode,
                        headers: res.headers,
                        body: body
                    });
                });
            });

            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            if (options.body) {
                req.write(options.body);
            }
            
            req.end();
        });
    }

    /**
     * Sleep for specified milliseconds
     * @private
     */
    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Dataset methods

    /**
     * List all available datasets.
     * @returns {Promise<Array>} List of dataset information objects
     */
    async listDatasets() {
        const response = await this._makeRequest('GET', '/datasets');
        return response.data.datasets || [];
    }

    /**
     * Get detailed information about a specific dataset.
     * @param {string} datasetName - Name of the dataset
     * @returns {Promise<Object>} Dataset information object
     */
    async getDatasetInfo(datasetName) {
        const response = await this._makeRequest('GET', `/datasets/${datasetName}`);
        return response.data;
    }

    // Conversation methods

    /**
     * Get conversations with optional filtering.
     * @param {Object} options - Filtering options
     * @param {string} options.dataset - Filter by dataset name
     * @param {string} options.tier - Filter by quality tier
     * @param {number} options.minQuality - Minimum quality score (0.0-1.0)
     * @param {number} options.limit - Maximum number of results (1-1000)
     * @param {number} options.offset - Offset for pagination
     * @returns {Promise<Object>} Object with conversations list and pagination info
     */
    async getConversations(options = {}) {
        const params = {
            limit: options.limit || 100,
            offset: options.offset || 0
        };
        
        if (options.dataset) params.dataset = options.dataset;
        if (options.tier) params.tier = options.tier;
        if (options.minQuality !== undefined) params.min_quality = options.minQuality;

        const response = await this._makeRequest('GET', '/conversations', { params });
        return response.data;
    }

    /**
     * Get a specific conversation by ID.
     * @param {string} conversationId - Unique conversation identifier
     * @returns {Promise<Object>} Conversation details object
     */
    async getConversation(conversationId) {
        const response = await this._makeRequest('GET', `/conversations/${conversationId}`);
        return response.data;
    }

    /**
     * Iterate through all conversations with automatic pagination.
     * @param {Object} options - Filtering options
     * @param {number} options.batchSize - Number of conversations to fetch per request
     * @returns {AsyncGenerator<Object>} Async generator yielding conversation objects
     */
    async* iterConversations(options = {}) {
        const batchSize = options.batchSize || 100;
        let offset = 0;

        while (true) {
            const batch = await this.getConversations({
                ...options,
                limit: batchSize,
                offset: offset
            });

            const conversations = batch.conversations || [];
            if (conversations.length === 0) {
                break;
            }

            for (const conversation of conversations) {
                yield conversation;
            }

            offset += conversations.length;

            // Check if we've reached the end
            if (conversations.length < batchSize) {
                break;
            }
        }
    }

    // Quality methods

    /**
     * Get quality metrics for datasets or tiers.
     * @param {Object} options - Filtering options
     * @param {string} options.dataset - Filter by dataset name
     * @param {string} options.tier - Filter by quality tier
     * @returns {Promise<Object>} Quality metrics object
     */
    async getQualityMetrics(options = {}) {
        const params = {};
        if (options.dataset) params.dataset = options.dataset;
        if (options.tier) params.tier = options.tier;

        const response = await this._makeRequest('GET', '/quality/metrics', { params });
        return response.data;
    }

    /**
     * Validate the quality of a conversation using NLP-based assessment.
     * @param {Object} conversation - Conversation object with id, messages, etc.
     * @returns {Promise<Object>} Quality validation results object
     */
    async validateConversationQuality(conversation) {
        const response = await this._makeRequest('POST', '/quality/validate', { data: conversation });
        return response.data;
    }

    // Processing methods

    /**
     * Submit a processing job for dataset analysis or export.
     * @param {string} datasetName - Name of the dataset to process
     * @param {string} processingType - Type of processing (quality_validation, export, analysis)
     * @param {Object} parameters - Processing parameters object
     * @returns {Promise<Object>} Job information object
     */
    async submitProcessingJob(datasetName, processingType, parameters = {}) {
        const jobData = {
            dataset_name: datasetName,
            processing_type: processingType,
            parameters: parameters
        };

        const response = await this._makeRequest('POST', '/processing/submit', { data: jobData });
        return response.data;
    }

    /**
     * Get the status of a processing job.
     * @param {string} jobId - Unique job identifier
     * @returns {Promise<Object>} Job status object
     */
    async getJobStatus(jobId) {
        const response = await this._makeRequest('GET', `/processing/jobs/${jobId}`);
        return response.data;
    }

    /**
     * Wait for a processing job to complete.
     * @param {string} jobId - Unique job identifier
     * @param {Object} options - Wait options
     * @param {number} options.pollInterval - Seconds between status checks
     * @param {number} options.timeout - Maximum time to wait in seconds
     * @returns {Promise<Object>} Final job status object
     */
    async waitForJob(jobId, options = {}) {
        const pollInterval = (options.pollInterval || 30) * 1000;
        const timeout = (options.timeout || 3600) * 1000;
        const startTime = Date.now();

        while (Date.now() - startTime < timeout) {
            const status = await this.getJobStatus(jobId);

            if (['completed', 'failed', 'cancelled'].includes(status.status)) {
                return status;
            }

            console.log(`Job ${jobId} status: ${status.status} (${status.progress || 0}%)`);
            await this._sleep(pollInterval);
        }

        throw new PixelatedEmpathyAPIError(`Job ${jobId} did not complete within ${timeout / 1000} seconds`);
    }

    // Search methods

    /**
     * Search conversations using advanced filters and full-text search.
     * @param {string} query - Search query string
     * @param {Object} options - Search options
     * @param {Object} options.filters - Search filters object
     * @param {number} options.limit - Maximum number of results
     * @param {number} options.offset - Offset for pagination
     * @returns {Promise<Object>} Search results object
     */
    async searchConversations(query, options = {}) {
        const searchData = {
            query: query,
            filters: options.filters || {},
            limit: options.limit || 100,
            offset: options.offset || 0
        };

        const response = await this._makeRequest('POST', '/search', { data: searchData });
        return response.data;
    }

    // Statistics methods

    /**
     * Get comprehensive statistics about the API and datasets.
     * @returns {Promise<Object>} Statistics overview object
     */
    async getStatisticsOverview() {
        const response = await this._makeRequest('GET', '/statistics/overview');
        return response.data;
    }

    // Export methods

    /**
     * Export data in specified format with optional filtering.
     * @param {string} dataset - Dataset name to export
     * @param {Object} options - Export options
     * @param {string} options.format - Export format (jsonl, csv, parquet, huggingface, openai)
     * @param {string} options.tier - Filter by quality tier
     * @param {number} options.minQuality - Minimum quality score
     * @returns {Promise<Object>} Export information object
     */
    async exportData(dataset, options = {}) {
        const exportData = {
            dataset: dataset,
            format: options.format || 'jsonl'
        };
        
        if (options.tier) exportData.tier = options.tier;
        if (options.minQuality !== undefined) exportData.min_quality = options.minQuality;

        const response = await this._makeRequest('POST', '/export', {
            data: exportData,
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
        });
        return response.data;
    }

    // Utility methods

    /**
     * Check if the API is healthy.
     * @returns {Promise<boolean>} True if API is healthy, false otherwise
     */
    async healthCheck() {
        try {
            const response = await this._makeRequest('GET', '/health');
            return response.success;
        } catch (error) {
            return false;
        }
    }
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PixelatedEmpathyAPI,
        PixelatedEmpathyAPIError,
        RateLimitError
    };
}

// Example usage
async function example() {
    // Initialize client
    const api = new PixelatedEmpathyAPI('your_api_key_here');

    try {
        // Check API health
        const isHealthy = await api.healthCheck();
        if (!isHealthy) {
            console.log('API is not available');
            return;
        }

        // List datasets
        const datasets = await api.listDatasets();
        console.log(`Available datasets: ${datasets.length}`);
        datasets.forEach(dataset => {
            console.log(`  - ${dataset.name}: ${dataset.conversations} conversations`);
        });

        // Get professional conversations
        const conversations = await api.getConversations({ tier: 'professional', limit: 5 });
        console.log(`\nFound ${conversations.conversations.length} professional conversations`);

        // Search for anxiety-related conversations
        const searchResults = await api.searchConversations('anxiety therapy techniques', {
            filters: { tier: 'professional', min_quality: 0.7 }
        });
        console.log(`\nFound ${searchResults.total_matches} matching conversations`);

        // Get quality metrics
        const metrics = await api.getQualityMetrics();
        console.log(`\nOverall quality metrics:`);
        console.log(`  Average quality: ${metrics.overall_statistics.average_quality}`);
        console.log(`  Total conversations: ${metrics.overall_statistics.total_conversations}`);

        // Example conversation quality validation
        const sampleConversation = {
            id: 'test_conv_001',
            messages: [
                { role: 'user', content: "I'm feeling anxious about my job interview tomorrow." },
                { role: 'assistant', content: "It's completely natural to feel anxious before an important interview. Can you tell me what specific aspects are making you most worried?" }
            ],
            quality_score: 0.0,
            tier: 'unknown'
        };

        const validationResult = await api.validateConversationQuality(sampleConversation);
        console.log(`\nConversation quality validation:`);
        console.log(`  Overall quality: ${validationResult.validation_results.overall_quality}`);
        console.log(`  Tier classification: ${validationResult.tier_classification}`);

    } catch (error) {
        console.error('Error:', error.message);
        if (error instanceof RateLimitError) {
            console.log(`Rate limited. Retry after ${error.retryAfter} seconds.`);
        }
    }
}

// Run example if this file is executed directly
if (require.main === module) {
    example();
}
