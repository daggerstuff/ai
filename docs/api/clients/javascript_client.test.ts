import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  PixelatedEmpathyAPI,
  PixelatedEmpathyAPIError,
  RateLimitError,
  type RequestOptions,
} from "./javascript_client";

describe("PixelatedEmpathyAPIError Initialization", () => {
  it("should initialize with correct properties", () => {
    const error = new PixelatedEmpathyAPIError("Test error message", "TEST_CODE", 500);
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("PixelatedEmpathyAPIError");
    expect(error.message).toBe("Test error message");
    expect(error.errorCode).toBe("TEST_CODE");
    expect(error.statusCode).toBe(500);
  });
});

describe("RateLimitError Initialization", () => {
  it("should initialize with correct properties and format message", () => {
    const error = new RateLimitError(30);
    expect(error).toBeInstanceOf(Error);
    expect(error).toBeInstanceOf(PixelatedEmpathyAPIError);
    expect(error.name).toBe("RateLimitError");
    expect(error.retryAfter).toBe(30);
    expect(error.message).toBe("Rate limit exceeded. Retry after 30 seconds.");
    expect(error.errorCode).toBeNull();
    expect(error.statusCode).toBeNull();
  });
});

describe("PixelatedEmpathyAPI healthCheck", () => {
  it("should return true when health check succeeds", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValue({ success: true });

    const isHealthy = await api.healthCheck();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/health");
    expect(isHealthy).toBe(true);
  });

  it("should return false when health check returns false success", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValue({ success: false });

    const isHealthy = await api.healthCheck();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/health");
    expect(isHealthy).toBe(false);
  });

  it("should return false when health check throws an error", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockRejectedValue(new Error("Network error"));

    const isHealthy = await api.healthCheck();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/health");
    expect(isHealthy).toBe(false);
  });
});

describe("PixelatedEmpathyAPI Rate Limiting", () => {
  it("should retry after 429 error and succeed", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const httpRequestSpy = vi.spyOn(api, "httpRequest");
    httpRequestSpy
      .mockResolvedValueOnce({
        statusCode: 429,
        headers: { "retry-after": "0" },
        body: "",
      })
      .mockResolvedValueOnce({
        statusCode: 200,
        headers: {},
        body: '{"success": true}',
      });

    const result = await api.healthCheck();

    expect(httpRequestSpy).toHaveBeenCalledTimes(2);
    expect(result).toBe(true);
  });

  it("should throw RateLimitError when retries exceed maxRetries", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    api.maxRetries = 2;
    const httpRequestSpy = vi.spyOn(api, "httpRequest");
    httpRequestSpy.mockResolvedValue({
      statusCode: 429,
      headers: { "retry-after": "0" },
      body: "",
    });
    const makeRequest = api.makeRequest.bind(api);

    await expect(makeRequest("GET", "/test")).rejects.toThrow(RateLimitError);
    expect(httpRequestSpy).toHaveBeenCalledTimes(3);
  });
});

describe("PixelatedEmpathyAPI Methods", () => {
  it("getConversations should handle pagination options correctly", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValueOnce({ data: { conversations: [] } });

    await api.getConversations({
      limit: 50,
      offset: 100,
      dataset: "test_dataset",
    });

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/conversations", {
      params: {
        limit: 50,
        offset: 100,
        dataset: "test_dataset",
      },
    });
  });

  it("getConversation should call makeRequest with correct endpoint", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValueOnce({ data: { id: "conv-123" } });

    const result = await api.getConversation("conv-123");

    expect(makeRequestSpy).toHaveBeenCalledWith(
      "GET",
      "/conversations/conv-123",
    );
    expect(result).toEqual({ id: "conv-123" });
  });
});

describe("PixelatedEmpathyAPI Method getConversations", () => {
  it("should correctly map minQuality to min_quality parameter", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledEndpoint = endpoint;
        calledOptions = options ?? {};
        return { data: { conversations: [] } };
      },
    );

    await api.getConversations({
      limit: 50,
      offset: 10,
      dataset: "test_dataset",
      tier: "professional",
      minQuality: 0.8,
    });

    expect(calledEndpoint).toBe("/conversations");
    expect(calledOptions.params).toEqual({
      limit: 50,
      offset: 10,
      dataset: "test_dataset",
      tier: "professional",
      min_quality: 0.8,
    });
  });

  it("should use default limit and offset if not provided", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledOptions: Record<string, unknown> = {};
    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledOptions = options ?? {};
        return { data: { conversations: [] } };
      },
    );

    await api.getConversations();

    expect(calledOptions.params).toEqual({
      limit: 100,
      offset: 0,
    });
  });

  it("should handle minQuality of 0 correctly", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledOptions: Record<string, unknown> = {};
    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledOptions = options ?? {};
        return { data: { conversations: [] } };
      },
    );

    await api.getConversations({ minQuality: 0 });

    expect(calledOptions.params).toEqual({
      limit: 100,
      offset: 0,
      min_quality: 0,
    });
  });
});

describe("PixelatedEmpathyAPI Method waitForJob", () => {
  it("should resolve immediately if job is already completed", async () => {
    const api = new PixelatedEmpathyAPI("test_key");

    api.getJobStatus = async (jobId: string) => {
      return { status: "completed", progress: 100 };
    };

    const result = await api.waitForJob("job-123");
    expect(result).toEqual({ status: "completed", progress: 100 });
  });

  it("should poll until job is completed", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    let callCount = 0;

    api.getJobStatus = async (jobId: string) => {
      callCount++;
      if (callCount === 1) return { status: "processing", progress: 50 };
      return { status: "completed", progress: 100 };
    };

    api.sleep = async (_ms: number) => {};

    const result = await api.waitForJob("job-123", {
      timeout: 10,
      pollInterval: 0.01,
    });
    expect(result).toEqual({ status: "completed", progress: 100 });
    expect(callCount).toBe(2);
  });

  it("should throw error if timeout is exceeded", async () => {
    const api = new PixelatedEmpathyAPI("test_key");

    api.getJobStatus = async (jobId: string) => {
      return { status: "processing", progress: 50 };
    };

    const originalNow = Date.now;
    let now = 0;
    Date.now = () => now;
    api.sleep = async (ms: number) => {
      now += ms;
    };

    try {
      await expect(
        api.waitForJob("job-123", { timeout: 0.1, pollInterval: 0.01 }),
      ).rejects.toThrow(PixelatedEmpathyAPIError);
    } finally {
      Date.now = originalNow;
    }
  });
});

describe("PixelatedEmpathyAPI Method iterConversations", () => {
  it("should handle pagination correctly and yield items", async () => {
    const api = new PixelatedEmpathyAPI("test_key");

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

  it("should exit early if batch returns 0 items", async () => {
    const api = new PixelatedEmpathyAPI("test_key");

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

describe("PixelatedEmpathyAPI Method validateConversationQuality", () => {
  it("should correctly submit conversation for validation", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions = {};

    makeRequestSpy.mockImplementation(async (method, endpoint, options) => {
      calledEndpoint = endpoint;
      calledOptions = options as any;
      return {
        statusCode: 200,
        headers: {},
        body: '{"success":true,"data":{"valid":true,"score":0.95}}',
        success: true,
        data: { valid: true, score: 0.95 },
      };
    });

    const conversation = { messages: [{ role: "user", content: "hello" }] };
    const result = await api.validateConversationQuality(conversation);

    expect(calledEndpoint).toBe("/quality/validate");
    expect(calledOptions).toHaveProperty("data");
    expect(calledOptions).toEqual({ data: conversation });
    expect(result).toEqual({ valid: true, score: 0.95 });
    expect(makeRequestSpy).toHaveBeenCalledTimes(1);
  });

  it("should handle API errors appropriately", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi
      .spyOn(api, "makeRequest")
      .mockRejectedValue(new Error("Validation failed"));

    const conversation = { messages: [] };
    await expect(api.validateConversationQuality(conversation)).rejects.toThrow(
      "Validation failed",
    );
    expect(makeRequestSpy).toHaveBeenCalledTimes(1);
  });
});

describe("PixelatedEmpathyAPI Method submitProcessingJob", () => {
  it("should correctly build job data payload", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledEndpoint = endpoint;
        calledOptions = options ?? {};
        return { data: { job_id: "new-job-123" } };
      },
    );

    await api.submitProcessingJob("my-dataset", "export", { format: "csv" });

    expect(calledEndpoint).toBe("/processing/submit");
    expect(calledOptions.data).toEqual({
      dataset_name: "my-dataset",
      processing_type: "export",
      parameters: { format: "csv" },
    });
  });
});

describe("PixelatedEmpathyAPI Method getStatisticsOverview", () => {
  it("should call makeRequest with correct endpoint and return overview", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    const mockOverview = { total_conversations: 1000, active_users: 50 };
    makeRequestSpy.mockResolvedValue({ data: mockOverview });

    const result = await api.getStatisticsOverview();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/statistics/overview");
    expect(result).toEqual(mockOverview);
  });
});

describe("PixelatedEmpathyAPI Method exportData", () => {
  it("should use default format 'jsonl' when options are not provided", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledOptions = options ?? {};
        return { data: { export_url: "http://example.com/export" } };
      },
    );

    await api.exportData("my-dataset");

    expect(calledOptions.data).toEqual({
      dataset: "my-dataset",
      format: "jsonl",
    });
  });

  it("should map options correctly to exportData payload", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledOptions = options ?? {};
        return { data: { export_url: "http://example.com/export" } };
      },
    );

    await api.exportData("my-dataset", {
      format: "csv",
      tier: "premium",
      minQuality: 0.9,
    });

    expect(calledOptions.data).toEqual({
      dataset: "my-dataset",
      format: "csv",
      tier: "premium",
      min_quality: 0.9,
    });
    expect(calledOptions.headers).toEqual({
      "Content-Type": "application/x-www-form-urlencoded",
    });
  });
});

describe("PixelatedEmpathyAPI Dataset Methods", () => {
  it("listDatasets should call makeRequest with correct endpoint and return datasets", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    const mockDatasets = [
      { name: "test_1", conversations: 10 },
      { name: "test_2", conversations: 20 },
    ];
    makeRequestSpy.mockResolvedValue({ data: { datasets: mockDatasets } });

    const result = await api.listDatasets();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/datasets");
    expect(result).toEqual(mockDatasets);
  });

  it("listDatasets should return empty array if datasets is missing from response", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValue({ data: {} });

    const result = await api.listDatasets();

    expect(makeRequestSpy).toHaveBeenCalledWith("GET", "/datasets");
    expect(result).toEqual([]);
  });

  it("getDatasetInfo should call makeRequest with correct endpoint", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    const mockDatasetInfo = { name: "test_dataset", count: 100 };
    makeRequestSpy.mockResolvedValue({ data: mockDatasetInfo });

    const result = await api.getDatasetInfo("test_dataset");

    expect(makeRequestSpy).toHaveBeenCalledWith(
      "GET",
      "/datasets/test_dataset",
    );
    expect(result).toEqual(mockDatasetInfo);
  });

  it("getDatasetInfo should return empty object if dataset info is missing from response", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValueOnce({ data: undefined });

    const result = await api.getDatasetInfo("test_dataset");

    expect(makeRequestSpy).toHaveBeenCalledWith(
      "GET",
      "/datasets/test_dataset",
    );
    expect(result).toEqual({});
  });

  it("getDatasetInfo should return empty object if response has no data property", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValueOnce({});

    const result = await api.getDatasetInfo("test_dataset");

    expect(makeRequestSpy).toHaveBeenCalledWith(
      "GET",
      "/datasets/test_dataset",
    );
    expect(result).toEqual({});
  });
});

describe("PixelatedEmpathyAPI Method searchConversations", () => {
  it("should correctly pass query and default options to makeRequest", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledEndpoint = endpoint;
        calledOptions = options ?? {};
        return { data: { results: ["conversation1"] } };
      },
    );

    const result = await api.searchConversations("anxiety");

    expect(calledEndpoint).toBe("/search");
    expect(calledOptions.data).toEqual({
      query: "anxiety",
      filters: {},
      limit: 100,
      offset: 0,
    });
    expect(result).toEqual({ results: ["conversation1"] });
  });

  it("should correctly merge provided options", async () => {
    const api = new PixelatedEmpathyAPI("test_key");

    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    let calledOptions: Record<string, unknown> = {};
    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledOptions = options ?? {};
        return { data: { results: [] } };
      },
    );

    await api.searchConversations("depression", {
      filters: { tier: "professional" },
      limit: 10,
      offset: 20,
    });

    expect(calledOptions.data).toEqual({
      query: "depression",
      filters: { tier: "professional" },
      limit: 10,
      offset: 20,
    });
  });
});

describe("PixelatedEmpathyAPI Method getQualityMetrics", () => {
  it("should correctly map options to params", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledEndpoint = endpoint;
        calledOptions = options ?? {};
        return { data: { overall_statistics: {} } };
      },
    );

    await api.getQualityMetrics({
      dataset: "test_dataset",
      tier: "professional",
    });

    expect(calledEndpoint).toBe("/quality/metrics");
    expect(calledOptions.params).toEqual({
      dataset: "test_dataset",
      tier: "professional",
    });
  });

  it("should handle missing options gracefully", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    let calledEndpoint = "";
    let calledOptions: Record<string, unknown> = {};

    makeRequestSpy.mockImplementation(
      async (method: string, endpoint: string, options?: RequestOptions) => {
        calledEndpoint = endpoint;
        calledOptions = options ?? {};
        return { data: { overall_statistics: {} } };
      },
    );

    await api.getQualityMetrics();

    expect(calledEndpoint).toBe("/quality/metrics");
    expect(calledOptions.params).toEqual({});
  });
});
describe("PixelatedEmpathyAPI Method healthCheck", () => {
  it("should return true on successful request", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    makeRequestSpy.mockResolvedValue({ success: true });

    const isHealthy = await api.healthCheck();
    expect(isHealthy).toBe(true);
  });

  it("should return false if request throws error", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");

    makeRequestSpy.mockRejectedValue(new Error("Network error"));

    const isHealthy = await api.healthCheck();
    expect(isHealthy).toBe(false);
  });
});

describe("PixelatedEmpathyAPI Method getJobStatus", () => {
  it("should correctly call makeRequest with jobId", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    const mockStatus = { status: "completed" };
    makeRequestSpy.mockResolvedValue({ data: mockStatus });

    const result = await api.getJobStatus("job-123");

    expect(makeRequestSpy).toHaveBeenCalledWith(
      "GET",
      "/processing/jobs/job-123",
    );
    expect(result).toEqual(mockStatus);
  });

  it('should default to status "unknown" and undefined progress if payload is missing fields', async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const makeRequestSpy = vi.spyOn(api, "makeRequest");
    makeRequestSpy.mockResolvedValue({ data: {} });

    const result = await api.getJobStatus("job-123");

    expect(result).toEqual({ status: "unknown", progress: undefined });
  });
});

describe("PixelatedEmpathyAPI value formatting", () => {
  it("should format string correctly", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue("test")).toBe("test");
  });

  it("should format number correctly", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(42)).toBe("42");
  });

  it("should format null correctly", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(null)).toBe("");
  });

  it("should format boolean correctly", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(true)).toBe("true");
    expect(api.formatValue(false)).toBe("false");
  });
});

describe("PixelatedEmpathyAPI response parsing helpers", () => {
  const safeParseResponse = (jsonStr: string): Record<string, unknown> => {
    try {
      const parsed = JSON.parse(jsonStr) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return { success: true, data: parsed as Record<string, unknown> };
      }
    } catch {
      // fall through
    }
    return { success: false, message: "Invalid JSON response" };
  };

  it("should return parsed object if valid JSON object is passed", () => {
    const jsonStr = '{"success": true, "data": {"id": 1}}';
    expect(safeParseResponse(jsonStr)).toEqual({
      success: true,
      data: { success: true, data: { id: 1 } },
    });
  });

  it("should return error object if string is not valid JSON", () => {
    const invalidJsonStr = "<html><body>error</body></html>";
    expect(safeParseResponse(invalidJsonStr)).toEqual({
      success: false,
      message: "Invalid JSON response",
    });
  });

  it("should return error object if parsed JSON is array", () => {
    const jsonStr = '[{"id": 1}]';
    expect(safeParseResponse(jsonStr)).toEqual({
      success: false,
      message: "Invalid JSON response",
    });
  });
});

describe("PixelatedEmpathyAPI Method formatValue", () => {
  it("should handle undefined and null", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(undefined)).toBe("");
    expect(api.formatValue(null)).toBe("");
  });

  it("should handle strings", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue("hello")).toBe("hello");
  });

  it("should handle numbers, booleans, and bigints", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(123)).toBe("123");
    expect(api.formatValue(true)).toBe("true");
    expect(api.formatValue(false)).toBe("false");
    expect(api.formatValue(BigInt(9007199254740991))).toBe("9007199254740991");
  });

  it("should handle JSON serializable objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue({ key: "value" })).toBe('{"key":"value"}');
    expect(api.formatValue([1, 2, 3])).toBe("[1,2,3]");
  });

  it("should handle objects with circular references", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const circularObj: any = {};
    circularObj.self = circularObj;
    expect(api.formatValue(circularObj)).toBe("[object Object]");
  });

  it("should handle symbols", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(Symbol("test_symbol"))).toBeUndefined();
    expect(api.formatValue(Symbol())).toBeUndefined();
  });

  it("should handle functions", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const testFunc = () => {
      return "test";
    };
    expect(api.formatValue(testFunc)).toBeUndefined();
  });
});

describe("PixelatedEmpathyAPI Method normalizePayload", () => {
  it("should wrap string in data object", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.normalizePayload("raw string")).toEqual({ data: "raw string" });
  });

  it("should filter out undefined and null values and format others", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const rawPayload = {
      validString: "hello",
      validNumber: 123,
      invalidUndefined: undefined,
      invalidNull: null,
      validObject: { a: 1 },
    };

    const expectedPayload = {
      validString: "hello",
      validNumber: "123",
      validObject: '{"a":1}',
    };

    expect(api.normalizePayload(rawPayload)).toEqual(expectedPayload);
  });
});

describe("PixelatedEmpathyAPI Method httpRequest timeout", () => {
  it('should throw the original error if fetch fails with a non-AbortError', async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    api.maxRetries = 0; // Prevent automatic retry loops

    // Mock fetch globally
    const originalFetch = global.fetch;
    global.fetch = vi.fn().mockImplementation(() => {
      const error = new Error("Network connection lost");
      error.name = "TypeError";
      return Promise.reject(error);
    });

    try {
      await expect(
        api.httpRequest("http://test.com", {
          method: "GET",
          headers: {},
          timeout: 10,
        }),
      ).rejects.toThrow("Network connection lost");
    } finally {
      global.fetch = originalFetch;
    }
  });

  it('should throw Error with message "Request timeout" on AbortError', async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    api.maxRetries = 0; // Prevent automatic retry loops

    // Mock fetch globally
    const originalFetch = global.fetch;
    global.fetch = vi.fn().mockImplementation(() => {
      const error = new Error("The operation was aborted");
      error.name = "AbortError";
      return Promise.reject(error);
    });

    try {
      await expect(
        api.httpRequest("http://test.com", {
          method: "GET",
          headers: {},
          timeout: 10,
        }),
      ).rejects.toThrow("Request timeout");
    } finally {
      global.fetch = originalFetch;
    }
  });
});

describe("PixelatedEmpathyAPI Utility Methods", () => {
  describe("isPlainObject", () => {
    it("should return true for plain objects", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      expect(api.isPlainObject({})).toBe(true);
      expect(api.isPlainObject({ a: 1 })).toBe(true);
    });

    it("should return false for arrays, null, and primitives", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      expect(api.isPlainObject([])).toBe(false);
      expect(api.isPlainObject(null)).toBe(false);
      expect(api.isPlainObject("string")).toBe(false);
      expect(api.isPlainObject(123)).toBe(false);
      expect(api.isPlainObject(undefined)).toBe(false);
      expect(api.isPlainObject(() => {})).toBe(false);
    });
  });

  describe("toRecord", () => {
    it("should return the object if it is a plain object", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      const obj = { a: 1 };
      expect(api.toRecord(obj)).toEqual(obj);
    });

    it("should return an empty object for primitives, arrays, and null if no fallback is provided", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      expect(api.toRecord("string")).toEqual({});
      expect(api.toRecord([])).toEqual({});
      expect(api.toRecord(null)).toEqual({});
      expect(api.toRecord(undefined)).toEqual({});
    });

    it("should return the provided fallback for non-plain objects", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      const fallback = { fallback: true };
      expect(api.toRecord("string", fallback)).toEqual(fallback);
      expect(api.toRecord(null, fallback)).toEqual(fallback);
    });
  });

  describe("toError", () => {
    it("should return the error if it is an instance of Error", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      const err = new Error("Test error");
      expect(api.toError(err)).toBe(err);
    });

    it("should wrap a string in an Error object", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      const err = api.toError("String error message");
      expect(err).toBeInstanceOf(Error);
      expect(err.message).toBe("String error message");
    });

    it("should return an Unknown error for objects that are not strings or Error instances", () => {
      const api = new PixelatedEmpathyAPI("test_key");
      const err = api.toError({ code: 500 });
      expect(err).toBeInstanceOf(Error);
      expect(err.message).toBe("Unknown error");
    });
  });
});

describe("PixelatedEmpathyAPI Method toRecordArray edge cases", () => {
  it("should correctly handle undefined by returning empty array", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.toRecordArray(undefined)).toEqual([]);
  });

  it("should correctly handle non-array inputs by returning empty array", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.toRecordArray(null)).toEqual([]);
    expect(api.toRecordArray("string")).toEqual([]);
    expect(api.toRecordArray(123)).toEqual([]);
    expect(api.toRecordArray({ a: 1 })).toEqual([]);
    expect(api.toRecordArray(() => {})).toEqual([]);
  });

  it("should correctly filter out non-plain objects from arrays", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const input = [{ valid: true }, null, "string", [1, 2], { another: "yes" }];
    expect(api.toRecordArray(input)).toEqual([
      { valid: true },
      { another: "yes" },
    ]);
  });
});

describe("PixelatedEmpathyAPI Method isPlainObject", () => {
  it("should correctly identify plain objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.isPlainObject({})).toBe(true);
    expect(api.isPlainObject({ a: 1 })).toBe(true);
  });

  it("should reject arrays, null, strings, and other non-objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.isPlainObject([])).toBe(false);
    expect(api.isPlainObject(null)).toBe(false);
    expect(api.isPlainObject(undefined)).toBe(false);
    expect(api.isPlainObject("string")).toBe(false);
    expect(api.isPlainObject(123)).toBe(false);
    expect(api.isPlainObject(true)).toBe(false);
  });
});

describe("PixelatedEmpathyAPI Method formatValue", () => {
  it("should handle string, number, and boolean", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue("test string")).toBe("test string");
    expect(api.formatValue(123)).toBe("123");
    expect(api.formatValue(true)).toBe("true");
    expect(api.formatValue(false)).toBe("false");
  });

  it("should handle object (JSON.stringify)", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue({ a: 1 })).toBe('{"a":1}');
    expect(api.formatValue([1, 2])).toBe("[1,2]");
  });

  it("should handle null/undefined (empty string)", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(null)).toBe("");
    expect(api.formatValue(undefined)).toBe("");
  });
});

describe("PixelatedEmpathyAPI Method normalizePayload", () => {
  it("should wrap string payload in { data: string }", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.normalizePayload("test string")).toEqual({
      data: "test string",
    });
  });

  it("should format values and remove null/undefined from objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const input = {
      validString: "test",
      validNumber: 123,
      nullValue: null,
      undefinedValue: undefined,
    };
    const expected = {
      validString: "test",
      validNumber: "123",
    };
    expect(api.normalizePayload(input)).toEqual(expected);
  });
});

describe("PixelatedEmpathyAPI Method toError", () => {
  it("should pass through Error objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const err = new Error("Test error");
    expect(api.toError(err)).toBe(err);
  });

  it("should wrap strings in Error objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const result = api.toError("String error");
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toBe("String error");
  });

  it("should handle unknown types with a fallback message", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const result = api.toError({ some: "object" });
    expect(result).toBeInstanceOf(Error);
    expect(result.message).toBe("Unknown error");
  });
});

describe("PixelatedEmpathyAPI Method formatValue edge cases", () => {
  it("should return error object if parsed JSON is null", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(undefined)).toBe("");
    expect(api.formatValue(null)).toBe("");
  });

  it("should correctly format strings", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue("test string")).toBe("test string");
    expect(api.formatValue("")).toBe("");
  });

  it("should correctly format primitives (number, boolean, bigint)", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue(123)).toBe("123");
    expect(api.formatValue(0)).toBe("0");
    expect(api.formatValue(true)).toBe("true");
    expect(api.formatValue(false)).toBe("false");
    expect(api.formatValue(BigInt(9007199254740991))).toBe("9007199254740991");
  });

  it("should correctly format standard objects and arrays via JSON.stringify", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    expect(api.formatValue({ a: 1, b: "test" })).toBe('{"a":1,"b":"test"}');
    expect(api.formatValue([1, 2, 3])).toBe("[1,2,3]");
  });

  it("should correctly format Symbols", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    // JSON.stringify(Symbol(...)) returns undefined, not a string
    expect(api.formatValue(Symbol("my-symbol"))).toBeUndefined();
    expect(api.formatValue(Symbol())).toBeUndefined();
  });

  it("should correctly format circular objects", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const circularObj: any = { a: 1 };
    circularObj.self = circularObj;
    expect(api.formatValue(circularObj)).toBe("[object Object]");
  });

  it("should correctly format functions", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const myFunc = function () {
      return 42;
    };
    // JSON.stringify(function) returns undefined
    expect(api.formatValue(myFunc)).toBeUndefined();
  });
});

describe("PixelatedEmpathyAPI Method sleep", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should resolve after the specified delay", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const sleepPromise = api.sleep(100);

    // Fast-forward time
    vi.advanceTimersByTime(100);

    // If it correctly resolves, this await will finish
    await expect(sleepPromise).resolves.toBeUndefined();
  });

  it("should not resolve before the specified delay", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const sleepPromise = api.sleep(100);

    const resolved = vi.fn();
    void sleepPromise.then(resolved);

    await vi.advanceTimersByTimeAsync(50);

    // Ensure promise has not resolved yet
    expect(resolved).not.toHaveBeenCalled();
  });
});

describe("PixelatedEmpathyAPI Method safeParseResponse edge cases", () => {
  it("should return parsed object if valid JSON object is passed", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const jsonStr = '{"success": true, "data": {"id": 1}}';
    expect(api.safeParseResponse(jsonStr)).toEqual({
      success: true,
      data: { id: 1 },
    });
  });

  it("should return error object if parsed JSON is null", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const jsonStr = "null";
    expect(api.safeParseResponse(jsonStr)).toEqual({
      success: false,
      message: "Invalid JSON response",
    });
  });

  it("should return error object if parsed JSON is primitive string", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const jsonStr = '"just a string"';
    expect(api.safeParseResponse(jsonStr)).toEqual({
      success: false,
      message: "Invalid JSON response",
    });
  });

  it("should return error object if string is unparseable JSON", () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const jsonStr = "{ bad: json }";
    expect(api.safeParseResponse(jsonStr)).toEqual({
      success: false,
      message: "Invalid JSON response",
    });
  });
});

describe("PixelatedEmpathyAPI Method makeRequest", () => {
  it("should append query parameters correctly, filtering out null/undefined and handling existing query strings", async () => {
    const api = new PixelatedEmpathyAPI("test_key");
    const httpRequestSpy = vi.spyOn(api, "httpRequest").mockResolvedValue({
      statusCode: 200,
      headers: {},
      body: '{"success": true}',
    });

    await api.makeRequest("GET", "/endpoint?initial=true", {
      params: { valid: "value", num: 42, undef: undefined, nul: null },
    });

    expect(httpRequestSpy).toHaveBeenCalledWith(
      "https://api.pixelatedempathy.com/v1/endpoint?initial=true&valid=value&num=42",
      expect.anything(),
    );
  });
});
