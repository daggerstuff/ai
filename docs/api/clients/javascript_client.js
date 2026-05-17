/**
 * Pixelated Empathy AI - JavaScript/Node.js Client Library
 * Task 51: Complete API Documentation
 *
 * Official JavaScript client for the Pixelated Empathy AI API.
 */

import * as https from 'https'

/**
 * @typedef {{ [key: string]: unknown }} GenericRecord
 */
/**
 * @typedef {{ baseUrl?: string, timeout?: number, maxRetries?: number }} ClientOptions
 */
/**
 * @typedef {{ params?: GenericRecord, data?: GenericRecord | unknown[] | null, headers?: Record<string, string>, timeout?: number }} RequestOptions
 */
/**
 * @typedef {{ statusCode: number, headers: Record<string, string | string[] | undefined>, body: string }} HttpResponse
 */
/**
 * @typedef {{ success?: boolean, data?: GenericRecord | unknown[], message?: string, timestamp?: string, error?: unknown }} ApiResponse
 */
/**
 * @typedef {{ dataset?: string, tier?: string, minQuality?: number }} ConversationFilters
 */
/**
 * @typedef {{ limit?: number, offset?: number, dataset?: string, tier?: string, minQuality?: number }} GetConversationsOptions
 */
/**
 * @typedef {{ batchSize?: number }} IterConversationsOptions
 */
/**
 * @typedef {{ dataset?: string, tier?: string }} QualityMetricsOptions
 */
/**
 * @typedef {{ query?: string, filters?: GenericRecord, limit?: number, offset?: number }} SearchOptions
 */
/**
 * @typedef {{ pollInterval?: number, timeout?: number }} WaitOptions
 */
/**
 * @typedef {{ format?: string, tier?: string, minQuality?: number }} ExportOptions
 */
/**
 * Custom error class for API errors
 */
class PixelatedEmpathyAPIError extends Error {
  /**
   * @param {string} message
   * @param {string | number | null} [errorCode]
   * @param {number | null} [statusCode]
   */
  constructor(message, errorCode = null, statusCode = null) {
    super(message)
    this.name = 'PixelatedEmpathyAPIError'
    this.errorCode = errorCode
    this.statusCode = statusCode
  }
}

/**
 * Custom error class for rate limit errors
 */
class RateLimitError extends PixelatedEmpathyAPIError {
  /**
   * @param {number} retryAfter
   */
  constructor(retryAfter) {
    super(`Rate limit exceeded. Retry after ${retryAfter} seconds.`)
    this.name = 'RateLimitError'
    this.retryAfter = retryAfter
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
   * @param {ClientOptions} [options]
   */
  constructor(apiKey, options = {}) {
    /** @type {ClientOptions} */
    const clientOptions = options

    this.apiKey = apiKey
    this.baseUrl = (
      clientOptions.baseUrl ?? 'https://api.pixelatedempathy.com/v1'
    ).replace(/\/$/, '')
    this.timeout = clientOptions.timeout ?? 30000
    this.maxRetries = clientOptions.maxRetries ?? 3

    this.defaultHeaders = {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'User-Agent': 'PixelatedEmpathyAPI-JavaScript/1.0.0',
    }

    this['_makeRequest'] = this.makeRequest.bind(this)
    this['_httpRequest'] = this.httpRequest.bind(this)
    this['_sleep'] = this.sleep.bind(this)
  }

  /**
   * Make an HTTP request with error handling and retries
   * @private
   * @param {string} method
   * @param {string} endpoint
   * @param {RequestOptions} [options]
   * @returns {Promise<ApiResponse>}
   */
  async makeRequest(method, endpoint, options = {}) {
    /** @type {RequestOptions} */
    const requestOptions = options
    /** @type {string} */
    let requestUrl = this.baseUrl + endpoint

    // Add query parameters
    if (this.isPlainObject(requestOptions.params)) {
      const query = new URLSearchParams()
      for (const [key, value] of Object.entries(requestOptions.params)) {
        if (value !== undefined && value !== null) {
          query.append(key, this.formatValue(value))
        }
      }
      const queryString = query.toString()
      if (queryString) {
        requestUrl = `${requestUrl}${requestUrl.includes('?') ? '&' : '?'}${queryString}`
      }
    }

    /** @type {{ method: string, headers: Record<string, string>, timeout: number, body?: string }} */
    const requestConfig = {
      method: method.toUpperCase(),
      headers: { ...this.defaultHeaders, ...(requestOptions.headers ?? {}) },
      timeout: this.timeout,
    }

    // Add request body
    if (requestOptions.data !== undefined && requestOptions.data !== null) {
      if (
            requestOptions.headers?.['Content-Type'] ===
              'application/x-www-form-urlencoded'
      ) {
        requestConfig.body = new URLSearchParams(
            this.normalizePayload(
            typeof requestOptions.data === 'string'
              ? requestOptions.data
              : this.isPlainObject(requestOptions.data)
                ? requestOptions.data
                : { data: requestOptions.data },
          ),
        ).toString()
      } else {
        requestConfig.body =
          typeof requestOptions.data === 'string'
            ? requestOptions.data
            : JSON.stringify(requestOptions.data)
      }
    }

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await this.httpRequest(requestUrl, requestConfig)

        // Handle rate limiting
        if (response.statusCode === 429) {
          const retryAfterHeader = response.headers['retry-after']
          const retryAfterCandidate = Array.isArray(retryAfterHeader)
            ? retryAfterHeader[0]
            : retryAfterHeader
          let retryAfter = parseInt(retryAfterCandidate ?? '60', 10)
          if (Number.isNaN(retryAfter)) {
            retryAfter = 60
          }
          if (attempt < this.maxRetries) {
            console.warn(
              `Rate limited. Retrying after ${retryAfter} seconds...`,
            )
            await this.sleep(retryAfter * 1000)
            continue
          } else {
            throw new RateLimitError(retryAfter)
          }
        }

        // Parse response
        const data = this.safeParseResponse(response.body)

        // Handle API errors
        if (response.statusCode >= 400) {
          const errorPayload = this.toRecord(data.error)
          const errorMessage =
            typeof errorPayload.message === 'string'
              ? errorPayload.message
              : 'Unknown error'
          const errorCode =
            typeof errorPayload.code === 'string' ? errorPayload.code : 'UNKNOWN_ERROR'
          throw new PixelatedEmpathyAPIError(
            errorMessage,
            errorCode,
            response.statusCode,
          )
        }

        const payload = this.toRecord(data.data)
        return {
          success: data.success === true,
          data: payload,
          message: typeof data.message === 'string' ? data.message : '',
          timestamp: typeof data.timestamp === 'string' ? data.timestamp : '',
          error: data.error ?? null,
        }
      } catch (error) {
        const caughtError = this.toError(error)
        if (
          caughtError instanceof PixelatedEmpathyAPIError ||
          caughtError instanceof RateLimitError
        ) {
          throw caughtError
        }

        if (attempt < this.maxRetries) {
          console.warn(
            `Request failed (attempt ${attempt + 1}): ${caughtError.message}`,
          )
          await this.sleep(Math.pow(2, attempt) * 1000) // Exponential backoff
          continue
        } else {
          throw new PixelatedEmpathyAPIError(`Request failed: ${caughtError.message}`)
        }
      }
    }

    throw new PixelatedEmpathyAPIError('Request failed after retries')
  }

  /**
   * Convert values to strings for query/body payload serialization.
   * @param {unknown} value
   * @returns {string}
   */
  formatValue(value) {
    if (value === undefined || value === null) {
      return ''
    }
    if (typeof value === 'string') {
      return value
    }
    if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
      return String(value)
    }

    try {
      return JSON.stringify(value)
    } catch {
      if (typeof value === 'object') {
        return '[object Object]'
      }
      if (typeof value === 'symbol') {
        return value.description ?? ''
      }
      if (typeof value === 'function') {
        return value.toString()
      }
      return ''
    }
  }

  /**
   * Convert payload data into URLSearchParams-friendly values.
   * @param {GenericRecord | string} data
   * @returns {GenericRecord}
   */
  normalizePayload(data) {
    if (typeof data === 'string') {
      return { data }
    }

    /** @type {GenericRecord} */
    const payload = {}
    for (const [key, value] of Object.entries(data)) {
      if (value !== undefined && value !== null) {
        payload[key] = this.formatValue(value)
      }
    }

    return payload
  }

  /**
   * @param {string} rawBody
   * @returns {ApiResponse}
   */
  safeParseResponse(rawBody) {
    try {
      /** @type {unknown} */
      const parsed = JSON.parse(rawBody)
      return this.isPlainObject(parsed) ? parsed : { success: false, message: 'Invalid JSON response' }
    } catch {
      return { success: false, message: 'Invalid JSON response' }
    }
  }

  /**
   * @param {unknown} error
   * @returns {Error}
   */
  toError(error) {
    if (error instanceof Error) {
      return error
    }

    return new Error(typeof error === 'string' ? error : 'Unknown error')
  }

  /**
   * @param {unknown} value
   * @returns {value is GenericRecord}
   */
  isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
  }

  /**
   * @param {unknown} value
   * @param {GenericRecord} [fallback]
   * @returns {GenericRecord}
   */
  toRecord(value, fallback = {}) {
    return this.isPlainObject(value) ? value : fallback
  }

  /**
   * @param {unknown} value
   * @returns {Array<GenericRecord>}
   */
  toRecordArray(value) {
    if (!Array.isArray(value)) {
      return []
    }

    return value.filter((entry) => this.isPlainObject(entry))
  }

  /**
   * Make HTTP request using the Fetch API
   * @private
   * @param {string} url
   * @param {{ method: string, headers: Record<string, string>, timeout: number, body?: string }} requestConfig
   * @returns {Promise<HttpResponse>}
   */
  async httpRequest(url, requestConfig) {
    const requestConfigOptions = {
      method: requestConfig.method,
      headers: requestConfig.headers,
      body: requestConfig.body,
    }

    const controller = new AbortController()
    const timeout = setTimeout(() => {
      controller.abort()
    }, requestConfig.timeout)

    try {
      /** @type {Response} */
      const response = await fetch(url, {
        ...requestConfigOptions,
        signal: controller.signal,
      })
      clearTimeout(timeout)

      const responseBody = await response.text()
      /** @type {Record<string, string>} */
      const headers = {}
      response.headers.forEach((value, key) => {
        headers[key] = value
      })

      return {
        statusCode: response.status,
        headers,
        body: responseBody,
      }
    } catch (error) {
      clearTimeout(timeout)
      const requestError = this.toError(error)
      if (requestError.name === 'AbortError') {
        throw new Error('Request timeout')
      }
      throw requestError
    }
  }

  /**
   * Sleep for specified milliseconds
   * @private
   * @param {number} ms
   * @returns {Promise<void>}
   */
  async sleep(ms) {
    await new Promise((resolve) => setTimeout(resolve, ms))
  }

  // Dataset methods

  /**
   * List all available datasets.
   * @returns {Promise<Array<Object>>} List of dataset information objects
   */
  async listDatasets() {
    const response = await this.makeRequest('GET', '/datasets')
    const payload = this.toRecord(response.data)
    return this.toRecordArray(payload.datasets)
  }

  /**
   * Get detailed information about a specific dataset.
   * @param {string} datasetName - Name of the dataset
   * @returns {Promise<Object>} Dataset information object
   */
  async getDatasetInfo(datasetName) {
    const response = await this.makeRequest('GET', `/datasets/${datasetName}`)
    return this.toRecord(response.data)
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
   * @param {Object} [options]
   * @returns {Promise<Object>} Object with conversations list and pagination info
   */
  async getConversations(options = {}) {
    /** @type {GetConversationsOptions} */
    const conversationOptions = options
    /** @type {Record<string, number | string>} */
    const params = {
      limit: conversationOptions.limit ?? 100,
      offset: conversationOptions.offset ?? 0,
    }

    if (conversationOptions.dataset) params.dataset = conversationOptions.dataset
    if (conversationOptions.tier) params.tier = conversationOptions.tier
    if (conversationOptions.minQuality !== undefined)
      params.min_quality = conversationOptions.minQuality

    const response = await this.makeRequest('GET', '/conversations', {
      params,
    })
    const payload = this.toRecord(response.data)
    return {
      conversations: this.toRecordArray(payload.conversations),
      ...payload,
    }
  }

  /**
   * Get a specific conversation by ID.
   * @param {string} conversationId - Unique conversation identifier
   * @returns {Promise<Object>} Conversation details object
   */
  async getConversation(conversationId) {
    const response = await this.makeRequest(
      'GET',
      `/conversations/${conversationId}`,
    )
    return this.toRecord(response.data)
  }

  /**
   * Iterate through all conversations with automatic pagination.
   * @param {{ batchSize?: number }} [options]
   * @returns {AsyncGenerator<Object>} Async generator yielding conversation objects
   */
  async *iterConversations(options = {}) {
    /** @type {IterConversationsOptions} */
    const iterOptions = options
    const batchSize = iterOptions.batchSize ?? 100

    for (let offset = 0;;) {
      const batch = await this.getConversations({
        ...options,
        limit: batchSize,
        offset: offset,
      })

      const conversations = this.toRecordArray(batch.conversations)
      if (conversations.length === 0) {
        break
      }

      for (const conversation of conversations) {
        yield conversation
      }

      // Check if we've reached the end
      if (conversations.length < batchSize) {
        break
      }

      offset += conversations.length
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
    /** @type {QualityMetricsOptions} */
    const metricsOptions = options
    /** @type {Record<string, string>} */
    const params = {}
    if (metricsOptions.dataset) params.dataset = metricsOptions.dataset
    if (metricsOptions.tier) params.tier = metricsOptions.tier

    const response = await this.makeRequest('GET', '/quality/metrics', {
      params,
    })
    return this.toRecord(response.data)
  }

  /**
   * Validate the quality of a conversation using NLP-based assessment.
   * @param {Object} conversation - Conversation object with id, messages, etc.
   * @returns {Promise<Object>} Quality validation results object
   */
  async validateConversationQuality(conversation) {
    const response = await this.makeRequest('POST', '/quality/validate', {
      data: conversation,
    })
    return this.toRecord(response.data)
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
    /** @type {GenericRecord} */
    const jobParameters = parameters
    /** @type {GenericRecord} */
    const jobData = {
      dataset_name: datasetName,
      processing_type: processingType,
      parameters: jobParameters,
    }

    const response = await this.makeRequest('POST', '/processing/submit', {
      data: jobData,
    })
    return this.toRecord(response.data)
  }

  /**
   * Get the status of a processing job.
   * @param {string} jobId - Unique job identifier
   * @returns {Promise<{ status: string, progress?: number | null }>}
   */
  async getJobStatus(jobId) {
    const response = await this.makeRequest('GET', `/processing/jobs/${jobId}`)
    const payload = this.toRecord(response.data)
    return {
      status: typeof payload.status === 'string' ? payload.status : 'unknown',
      progress: typeof payload.progress === 'number' ? payload.progress : undefined,
    }
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
    /** @type {WaitOptions} */
    const waitOptions = options
    const pollInterval = (waitOptions.pollInterval ?? 30) * 1000
    const timeout = (waitOptions.timeout ?? 3600) * 1000
    const startTime = Date.now()

    while (Date.now() - startTime < timeout) {
      const currentStatus = await this.getJobStatus(jobId)
      const statusValue = currentStatus.status
      const progress = currentStatus.progress

      if (
        statusValue === 'completed' ||
        statusValue === 'failed' ||
        statusValue === 'cancelled'
      ) {
        return currentStatus
      }

      console.log(
        `Job ${jobId} status: ${statusValue} (${progress ?? 0}%)`,
      )
      await this.sleep(pollInterval)
    }

    throw new PixelatedEmpathyAPIError(
      `Job ${jobId} did not complete within ${timeout / 1000} seconds`,
    )
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
    /** @type {SearchOptions} */
    const searchOptions = options
    /** @type {GenericRecord} */
    const searchData = {
      query: query,
      filters: searchOptions.filters ?? {},
      limit: searchOptions.limit ?? 100,
      offset: searchOptions.offset ?? 0,
    }

    const response = await this.makeRequest('POST', '/search', {
      data: searchData,
    })
    return this.toRecord(response.data)
  }

  // Statistics methods

  /**
   * Get comprehensive statistics about the API and datasets.
   * @returns {Promise<Object>} Statistics overview object
   */
  async getStatisticsOverview() {
    const response = await this.makeRequest('GET', '/statistics/overview')
    return this.toRecord(response.data)
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
    /** @type {ExportOptions} */
    const exportOptions = options
    /** @type {GenericRecord} */
    const exportData = {
      dataset: dataset,
      format: exportOptions.format ?? 'jsonl',
    }

    if (exportOptions.tier) exportData.tier = exportOptions.tier
    if (exportOptions.minQuality !== undefined)
      exportData.min_quality = exportOptions.minQuality

    const response = await this.makeRequest('POST', '/export', {
      data: exportData,
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    })
    return this.toRecord(response.data)
  }

  // Utility methods

  /**
   * Check if the API is healthy.
   * @returns {Promise<boolean>} True if API is healthy, false otherwise
   */
  async healthCheck() {
    try {
      const response = await this.makeRequest('GET', '/health')
      return response.success
    } catch (error) {
      return false
    }
  }
}

export { PixelatedEmpathyAPI, PixelatedEmpathyAPIError, RateLimitError }

// Example usage has been moved to the JavaScript SDK README and test examples.
