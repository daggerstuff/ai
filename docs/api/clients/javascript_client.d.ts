export type RequestOptions = {
  params?: Record<string, unknown>;
  data?: Record<string, unknown> | unknown[];
  headers?: Record<string, string>;
  [key: string]: unknown;
};

export type ApiResponse = {
  success?: boolean;
  data?: Record<string, unknown> | unknown[];
  message?: string;
  timestamp?: string;
  error?: unknown;
};

export type HttpResponse = {
  statusCode: number;
  headers: Record<string, string | string[] | undefined>;
  body: string;
};

export type SleepOptions = {
  timeout?: number;
};

export class PixelatedEmpathyAPIError extends Error {
  errorCode: unknown;
  statusCode: unknown;
  constructor(message: string, errorCode?: unknown, statusCode?: unknown);
}

export class RateLimitError extends PixelatedEmpathyAPIError {
  retryAfter: unknown;
  constructor(retryAfter: number);
}

export class PixelatedEmpathyAPI {
  constructor(apiKey: string, options?: { baseUrl?: string; timeout?: number; maxRetries?: number });

  apiKey: string;
  baseUrl: string;
  timeout: number;
  maxRetries: number;
  defaultHeaders: Record<string, string>;

  _makeRequest(
    method: string,
    endpoint: string,
    options?: RequestOptions,
  ): Promise<ApiResponse>;
  _httpRequest(url: URL, options: Record<string, unknown>): Promise<HttpResponse>;
  _sleep(ms: number): Promise<void>;

  listDatasets(): Promise<Array<Record<string, unknown>>>;
  getDatasetInfo(datasetName: string): Promise<Record<string, unknown>>;

  getConversations(options?: {
    limit?: number;
    offset?: number;
    dataset?: string;
    tier?: string;
    minQuality?: number;
  }): Promise<{ conversations: Array<Record<string, unknown>>; [key: string]: unknown }>;

  getConversation(conversationId: string): Promise<Record<string, unknown>>;

  iterConversations(options?: {
    batchSize?: number;
    [key: string]: unknown;
  }): AsyncGenerator<Record<string, unknown>, void>;

  submitProcessingJob(
    datasetName: string,
    processingType: string,
    parameters?: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;

  exportData(
    dataset: string,
    options?: {
      format?: string;
      tier?: string;
      minQuality?: number;
      [key: string]: unknown;
    },
  ): Promise<Record<string, unknown>>;

  getJobStatus(jobId: string): Promise<{ status: string; progress?: number }>;
  waitForJob(
    jobId: string,
    options?: { pollInterval?: number; timeout?: number },
  ): Promise<{ status: string; progress?: number }>;

  searchConversations(
    query: string,
    options?: { filters?: Record<string, unknown>; limit?: number; offset?: number },
  ): Promise<Record<string, unknown>>;

  getQualityMetrics(options?: {
    dataset?: string;
    tier?: string;
  }): Promise<Record<string, unknown>>;

  healthCheck(): Promise<boolean>;
}
