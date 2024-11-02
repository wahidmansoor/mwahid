import { pipeline, env } from '@xenova/transformers';

class ModelManager {
  private static instance: ModelManager;
  private model: any = null;
  private isInitializing = false;
  
  private status = {
    isReady: false,
    isLoading: false,
    error: null as string | null,
    progress: 0
  };

  private constructor() {
    env.useBrowserCache = true;
    env.allowLocalModels = false;
    env.backends.onnx.wasm.numThreads = 1;
  }

  public static getInstance(): ModelManager {
    if (!ModelManager.instance) {
      ModelManager.instance = new ModelManager();
    }
    return ModelManager.instance;
  }

  public getStatus() {
    return { ...this.status };
  }

  private async initializeModel() {
    if (this.model) return this.model;
    if (this.isInitializing) return null;

    try {
      this.isInitializing = true;
      this.status = {
        isReady: false,
        isLoading: true,
        error: null,
        progress: 0
      };

      this.model = await pipeline(
        'text2text-generation',
        'Xenova/LaMini-Flan-T5-248M',
        {
          quantized: true,
          progress_callback: (progress: any) => {
            this.status.progress = Math.round(progress.progress * 100);
          }
        }
      );

      this.status = {
        isReady: true,
        isLoading: false,
        error: null,
        progress: 100
      };

      return this.model;
    } catch (error) {
      this.status = {
        isReady: false,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Model initialization failed',
        progress: 0
      };
      return null;
    } finally {
      this.isInitializing = false;
    }
  }

  private isChemotherapyQuery(prompt: string): boolean {
    const keywords = ['chemo', 'chemotherapy', 'protocol', 'regimen', 'treatment'];
    return keywords.some(keyword => prompt.toLowerCase().includes(keyword));
  }

  public async generateResponse(prompt: string, context: string) {
    if (!prompt.trim()) {
      return {
        text: '',
        error: 'Empty prompt'
      };
    }

    try {
      const modelInstance = this.model || await this.initializeModel();
      if (!modelInstance) {
        throw new Error('Model not available');
      }

      const result = await modelInstance(
        `Answer this oncology question. Context: ${context}. Question: ${prompt}`,
        {
          max_new_tokens: 128,
          temperature: 0.7,
          do_sample: true
        }
      );

      return {
        text: result[0].generated_text,
        metadata: {
          isChemotherapyQuery: this.isChemotherapyQuery(prompt)
        }
      };
    } catch (error) {
      return {
        text: '',
        error: error instanceof Error ? error.message : 'Failed to generate response'
      };
    }
  }
}

export default ModelManager;