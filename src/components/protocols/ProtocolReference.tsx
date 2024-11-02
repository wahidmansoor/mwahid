import React from 'react';
import { ExternalLink, BookOpen } from 'lucide-react';

interface ProtocolReferenceProps {
  protocolType: string;
  showReference: boolean;
}

const ProtocolReference: React.FC<ProtocolReferenceProps> = ({ protocolType, showReference }) => {
  if (!showReference) return null;

  const getProtocolUrl = () => {
    const baseUrl = 'http://www.bccancer.bc.ca/chemotherapy-protocols-site';
    
    switch (protocolType.toLowerCase()) {
      case 'breast':
        return `${baseUrl}/breast`;
      case 'lung':
        return `${baseUrl}/lung`;
      case 'lymphoma':
        return `${baseUrl}/lymphoma`;
      case 'gastrointestinal':
      case 'colorectal':
        return `${baseUrl}/gastrointestinal`;
      default:
        return baseUrl;
    }
  };

  return (
    <div className="mt-2 p-3 bg-blue-50 rounded-lg">
      <div className="flex items-start gap-2">
        <BookOpen className="w-5 h-5 text-blue-600 mt-0.5" />
        <div>
          <p className="text-sm text-blue-800">
            For detailed protocol information, please refer to the BC Cancer Protocol Database:
          </p>
          <a
            href={getProtocolUrl()}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-1 inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800"
          >
            View Official Protocols
            <ExternalLink className="w-4 h-4" />
          </a>
        </div>
      </div>
    </div>
  );
};

export default ProtocolReference;