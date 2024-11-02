export type ErrorType = 'model' | 'initialization' | 'input' | 'unknown' | 'timeout' | 'network';

export interface ErrorInfo {
  type: ErrorType;
  message: string;
  timestamp: Date;
}

export interface ModelStatus {
  isReady: boolean;
  isLoading: boolean;
  error: string | null;
  progress: number;
  lastError: ErrorInfo | null;
}

export interface LLMError {
  type: ErrorType;
  message: string;
  timestamp: Date;
}

export interface LLMResponse {
  text: string;
  error?: LLMError;
  timestamp?: Date;
}

export interface ModelInterface {
  getStatus(): ModelStatus;
  generateResponse(prompt: string, context: string): Promise<LLMResponse>;
}