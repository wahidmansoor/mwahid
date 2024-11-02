import React, { useState, useCallback } from 'react';
import { Loader } from 'lucide-react';
import StagingSystem from './staging/StagingSystem';
import TreatmentOptions from './treatment/TreatmentOptions';
import ProtocolReference from './protocols/ProtocolReference';
import ProtocolViewer from './protocols/ProtocolViewer';
import { useModel } from '../lib/llm/hooks/useModel';
import ChatMessage from './chat/ChatMessage';
import ChatInput from './chat/ChatInput';
import ErrorMessage from './chat/ErrorMessage';
import LeadingQuestions from './chat/LeadingQuestions';
import type { Message } from './chat/types';

interface ChatInterfaceProps {
  unit: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ unit }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showStaging, setShowStaging] = useState(false);
  const [showProtocols, setShowProtocols] = useState(false);
  const [showQuestions, setShowQuestions] = useState(true);
  const [stagingResult, setStagingResult] = useState<{
    cancerType: string;
    stagingData: Record<string, string>;
  } | null>(null);

  const { status: modelStatus, generateResponse } = useModel();

  const handleRetry = useCallback(async () => {
    const lastUserMessage = messages.findLast(m => m.role === 'user');
    if (lastUserMessage) {
      await handleSend(lastUserMessage.content);
    }
  }, [messages]);

  const getPlaceholderText = (unit: string) => {
    if (modelStatus.isLoading) {
      return `Loading AI model (${modelStatus.progress}%)...`;
    }
    if (modelStatus.error) {
      return 'AI model failed to load. Please refresh the page.';
    }
    
    return 'Type your question here...';
  };

  const handleStagingComplete = (data: { cancerType: string; stagingData: Record<string, string> }) => {
    setStagingResult(data);
    const stagingMessage = `Cancer Staging Summary for ${data.cancerType.toUpperCase()}:\n\n` +
      Object.entries(data.stagingData)
        .map(([key, value]) => `${key}: ${value}`)
        .join('\n');

    const newMessage: Message = {
      role: 'assistant',
      content: stagingMessage,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, newMessage]);
    setShowStaging(false);
    setShowQuestions(false);
  };

  const handleSend = async (input: string) => {
    if (!input.trim() || isLoading || modelStatus.isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setShowQuestions(false);

    // Check for protocol-related queries
    if (input.toLowerCase().includes('protocol') || 
        input.toLowerCase().includes('chemo') || 
        input.toLowerCase().includes('treatment')) {
      setShowProtocols(true);
      setIsLoading(false);
      return;
    }

    if (input.toLowerCase().includes('staging') || input.toLowerCase().includes('diagnos')) {
      setIsLoading(false);
      setShowStaging(true);
      return;
    }

    try {
      const response = await generateResponse(input, unit);
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.error 
          ? `I apologize, but I encountered an error: ${response.error}`
          : response.text,
        timestamp: new Date(),
        error: !!response.error
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: 'I apologize, but I encountered an unexpected error. Please try again.',
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleQuestions = () => {
    setShowQuestions(!showQuestions);
  };

  return (
    <div className="flex-1 flex flex-col bg-gray-50">
      {modelStatus.error && (
        <ErrorMessage 
          error={{
            type: 'model',
            message: modelStatus.error,
            timestamp: new Date()
          }}
          onRetry={handleRetry}
        />
      )}

      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 && showQuestions && (
          <div className="space-y-6">
            <div className="text-center text-gray-500">
              <p className="text-lg font-medium mb-2">Welcome to the {unit.charAt(0).toUpperCase() + unit.slice(1)} Unit</p>
              <p className="text-sm">Select a question below or type your own:</p>
            </div>
            <LeadingQuestions unit={unit} onQuestionSelect={handleSend} />
          </div>
        )}
        
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
        
        {showProtocols && (
          <div className="mt-4 bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Treatment Protocols</h3>
            <ProtocolViewer category={stagingResult?.cancerType} />
          </div>
        )}
        
        {isLoading && (
          <div className="flex items-center justify-start mb-4">
            <div className="bg-white rounded-lg px-4 py-3 shadow-sm">
              <Loader className="w-5 h-5 animate-spin text-blue-600" />
            </div>
          </div>
        )}
        
        {showStaging && (
          <div className="mb-4">
            <StagingSystem onStagingComplete={handleStagingComplete} />
          </div>
        )}
        
        {stagingResult && !showStaging && (
          <div className="mb-4">
            <TreatmentOptions 
              cancerType={stagingResult.cancerType}
              stagingData={stagingResult.stagingData}
            />
          </div>
        )}

        {showQuestions && messages.length > 0 && (
          <div className="mt-6">
            <LeadingQuestions unit={unit} onQuestionSelect={handleSend} />
          </div>
        )}
      </div>

      <ChatInput
        onSend={handleSend}
        isDisabled={isLoading || modelStatus.isLoading}
        placeholder={getPlaceholderText(unit)}
        modelStatus={modelStatus}
        onToggleQuestions={toggleQuestions}
      />
    </div>
  );
}

export default ChatInterface;