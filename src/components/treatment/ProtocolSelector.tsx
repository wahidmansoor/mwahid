import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import { protocolDatabase } from './ProtocolDatabase';
import ProtocolDetails from './ProtocolDetails';

interface ProtocolSelectorProps {
  cancerType: string;
  onProtocolSelect: (protocol: any) => void;
}

const ProtocolSelector: React.FC<ProtocolSelectorProps> = ({ cancerType, onProtocolSelect }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIntent, setSelectedIntent] = useState<string>('');
  const protocols = protocolDatabase[cancerType]?.protocols || [];

  const filteredProtocols = protocols.filter(protocol => {
    const matchesSearch = protocol.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      protocol.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesIntent = !selectedIntent || protocol.intent === selectedIntent;
    return matchesSearch && matchesIntent;
  });

  return (
    <div className="space-y-4">
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            placeholder="Search protocols..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <select
          value={selectedIntent}
          onChange={(e) => setSelectedIntent(e.target.value)}
          className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">All Intents</option>
          <option value="Curative">Curative</option>
          <option value="Palliative">Palliative</option>
          <option value="Neoadjuvant">Neoadjuvant</option>
          <option value="Adjuvant">Adjuvant</option>
        </select>
      </div>

      <div className="space-y-3">
        {filteredProtocols.map((protocol, index) => (
          <button
            key={index}
            onClick={() => onProtocolSelect(protocol)}
            className="w-full text-left p-4 bg-white rounded-lg border border-gray-200 hover:border-blue-500 transition-colors"
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="font-medium text-gray-900">{protocol.name}</h3>
                <p className="text-sm text-gray-600 mt-1">{protocol.description}</p>
              </div>
              <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                {protocol.intent}
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default ProtocolSelector;