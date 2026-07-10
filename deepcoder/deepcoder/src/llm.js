/**
 * LLM Client — OpenAI-compatible wrapper for NVIDIA DeepSeek API.
 * Stripped down: no streaming, no tool definitions (we use text-based protocol).
 */
import OpenAI from 'openai';

const NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1';

export class LLMClient {
  constructor(options = {}) {
    const apiKey = options.apiKey || process.env.NVIDIA_API_KEY || process.env.DEEPSEEK_API_KEY;
    if (!apiKey) {
      throw new Error(
        'NVIDIA API key required. Set NVIDIA_API_KEY environment variable or pass apiKey option.'
      );
    }

    this.model = options.model || 'deepseek-ai/deepseek-v4-flash';
    this.maxTokens = options.maxTokens ?? 4096;
    this.temperature = options.temperature ?? 0.3;

    this.client = new OpenAI({
      baseURL: options.baseURL || NVIDIA_BASE_URL,
      apiKey,
    });
  }

  /** Non-streaming completion call */
  async complete(messages, options = {}) {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      max_tokens: options.maxTokens ?? this.maxTokens,
      temperature: options.temperature ?? this.temperature,
    });
    return response;
  }
}
