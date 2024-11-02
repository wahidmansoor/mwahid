import { useState, useEffect } from 'react';
import ModelManager from '../ModelManager';

export function useModel() {
  const [status, setStatus] = useState({
    isReady: false,
    isLoading: false,
    error: null as string | null,
    progress: 0
  });

  useEffect(() => {
    const modelManager = ModelManager.getInstance();
    const intervalId = setInterval(() => {
      setStatus(modelManager.getStatus());
    }, 1000);

    return () => clearInterval(intervalId);
  }, []);

  const generateResponse = async (prompt: string, context: string) => {
    const modelManager = ModelManager.getInstance();
    return modelManager.generateResponse(prompt, context);
  };

  return { status, generateResponse };
}