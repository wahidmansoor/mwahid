import Replicate from 'replicate';

const replicate = new Replicate({
  auth: import.meta.env.VITE_REPLICATE_API_TOKEN,
});

export interface LlamaResponse {
  text: string;
  error?: string;
}

export async function getLlamaResponse(
  prompt: string,
  context: string
): Promise<LlamaResponse> {
  try {
    const output = await replicate.run(
      "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
      {
        input: {
          prompt: `Context: You are an oncology AI assistant. You have access to medical knowledge about cancer treatments, protocols, and guidelines. The current context is: ${context}\n\nUser: ${prompt}\n\nAssistant:`,
          temperature: 0.75,
          top_p: 0.9,
          max_length: 500,
          repetition_penalty: 1.2
        }
      }
    );

    return {
      text: Array.isArray(output) ? output.join('') : String(output)
    };
  } catch (error) {
    return {
      text: '',
      error: error instanceof Error ? error.message : 'Failed to get response'
    };
  }
}