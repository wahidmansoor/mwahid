import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js environment
env.useBrowserCache = true;
env.allowLocalModels = false;
env.backends.onnx.wasm.numThreads = 4;

// Model state management
let model: any = null;
let isInitializing = false;
let initializationError: Error | null = null;

export interface ModelStatus {
  isReady: boolean;
  isLoading: boolean;
  error: string | null;
  progress: number;
}

let modelStatus: ModelStatus = {
  isReady: false,
  isLoading: false,
  error: null,
  progress: 0
};

export function getModelStatus(): ModelStatus {
  return { ...modelStatus };
}

async function initializeModel(): Promise<any> {
  if (model) return model;
  if (isInitializing) {
    throw new Error('Model initialization already in progress');
  }

  try {
    isInitializing = true;
    modelStatus = {
      isReady: false,
      isLoading: true,
      error: null,
      progress: 0
    };

    model = await pipeline(
      'text-generation',
      'Xenova/LaMini-Flan-T5-783M',
      {
        quantized: true,
        progress_callback: (progress: any) => {
          modelStatus.progress = Math.round(progress.progress * 100);
        }
      }
    );

    modelStatus = {
      isReady: true,
      isLoading: false,
      error: null,
      progress: 100
    };

    return model;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown initialization error';
    modelStatus = {
      isReady: false,
      isLoading: false,
      error: errorMessage,
      progress: 0
    };
    initializationError = error instanceof Error ? error : new Error(errorMessage);
    throw initializationError;
  } finally {
    isInitializing = false;
  }
}

export interface LLMResponse {
  text: string;
  error?: {
    type: 'model' | 'initialization' | 'input' | 'unknown';
    message: string;
  };
}

export async function getLLMResponse(
  prompt: string,
  context: string
): Promise<LLMResponse> {
  if (!prompt.trim()) {
    return {
      text: '',
      error: {
        type: 'input',
        message: 'Prompt cannot be empty'
      }
    };
  }

  try {
    const modelInstance = await initializeModel();

    if (!modelInstance) {
      throw new Error('Model initialization failed');
    }

    const fullPrompt = `Context: You are an oncology AI assistant. You have access to medical knowledge about cancer treatments, protocols, and guidelines.
Current context: ${context}

User query: ${prompt}

Please provide a helpful, accurate, and concise response.`;

    const result = await modelInstance(fullPrompt, {
      max_new_tokens: 512,
      temperature: 0.7,
      top_p: 0.95,
      repetition_penalty: 1.1,
      do_sample: true,
      num_beams: 1,
    });

    if (!result?.[0]?.generated_text) {
      throw new Error('Model generated empty response');
    }

    return {
      text: result[0].generated_text.trim()
    };
  } catch (error) {
    console.error('LLM Error:', error);
    
    return {
      text: '',
      error: {
        type: error === initializationError ? 'initialization' : 'model',
        message: error instanceof Error 
          ? error.message 
          : 'An unexpected error occurred. Please try again.'
      }
    };
  }
}