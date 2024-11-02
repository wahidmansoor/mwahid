export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  error?: boolean;
}

export interface MessageProps {
  message: Message;
}

export interface ChatInputProps {
  onSend: (message: string) => void;
  isDisabled: boolean;
  placeholder: string;
  modelStatus: {
    isLoading: boolean;
    error: string | null;
    progress: number;
  };
  onToggleQuestions: () => void;
}