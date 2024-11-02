import React, { useState } from 'react';
import { Send, HelpCircle } from 'lucide-react';
import { ChatInputProps } from './types';

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSend, 
  isDisabled, 
  placeholder,
  modelStatus,
  onToggleQuestions 
}) => {
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim() || isDisabled) return;
    onSend(input);
    setInput('');
  };

  return (
    <div className="border-t bg-white p-4">
      <div className="max-w-4xl mx-auto space-y-2">
        <button
          onClick={onToggleQuestions}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
        >
          <HelpCircle className="w-4 h-4" />
          Show suggested questions
        </button>
        <div className="flex gap-4">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            placeholder={placeholder}
            disabled={isDisabled}
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || isDisabled}
            className={`px-4 py-2 rounded-lg flex items-center gap-2 ${
              input.trim() && !isDisabled
                ? 'bg-blue-600 text-white hover:bg-blue-700' 
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
          >
            <Send className="w-4 h-4" />
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInput;