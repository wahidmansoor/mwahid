import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import type { ErrorInfo } from '../../lib/llm/types';

interface ErrorMessageProps {
  error: ErrorInfo;
  onRetry?: () => void;
}

const ErrorMessage: React.FC<ErrorMessageProps> = ({ error, onRetry }) => {
  const getErrorMessage = () => {
    switch (error.type) {
      case 'initialization':
        return 'Failed to initialize the AI model. Please try refreshing the page.';
      case 'network':
        return 'Network connection issue. Please check your internet connection.';
      case 'timeout':
        return 'The request took too long to process. Please try again.';
      default:
        return error.message || 'An unexpected error occurred.';
    }
  };

  return (
    <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-4">
      <div className="flex items-start">
        <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
        <div className="ml-3 flex-1">
          <p className="text-sm text-red-700">{getErrorMessage()}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-2 flex items-center gap-2 text-sm text-red-600 hover:text-red-800"
            >
              <RefreshCw className="w-4 h-4" />
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;