import React, { useState } from 'react';
import { Search, Filter, ChevronDown, ChevronUp } from 'lucide-react';
import { Protocol } from '../../database/types';
import { protocolDatabase, searchProtocols } from '../../database';

interface ProtocolViewerProps {
  category?: string;
}

const ProtocolViewer: React.FC<ProtocolViewerProps> = ({ category }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedIntent, setSelectedIntent] = useState<string>('');
  const [expandedProtocol, setExpandedProtocol] = useState<string | null>(null);

  const protocols = category 
    ? protocolDatabase[category as keyof typeof protocolDatabase] || []
    : searchProtocols(searchTerm);

  const filteredProtocols = protocols.filter(protocol => {
    const matchesSearch = !searchTerm || 
      protocol.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      protocol.regimens.some(r => r.name.toLowerCase().includes(searchTerm.toLowerCase()));
    const matchesIntent = !selectedIntent || protocol.intent === selectedIntent;
    return matchesSearch && matchesIntent;
  });

  const renderDrugInfo = (protocol: Protocol) => (
    <div className="mt-4 space-y-4">
      {protocol.regimens.map((regimen, idx) => (
        <div key={idx} className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-medium text-gray-900 mb-2">{regimen.name}</h4>
          <div className="space-y-2">
            {regimen.drugs.map((drug, drugIdx) => (
              <div key={drugIdx} className="flex flex-col sm:flex-row sm:items-center gap-2 text-sm">
                <span className="font-medium text-gray-700 min-w-[150px]">{drug.name}</span>
                <span className="text-gray-600">{drug.dose}</span>
                <span className="text-gray-500">({drug.route}, {drug.schedule})</span>
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-gray-200">
            <div className="text-sm text-gray-600">
              <span className="font-medium">Cycles:</span> {regimen.cycles},
              <span className="font-medium ml-2">Interval:</span> {regimen.interval} days
            </div>
          </div>
        </div>
      ))}
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row gap-4">
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

      <div className="space-y-4">
        {filteredProtocols.map((protocol) => (
          <div
            key={protocol.id}
            className="bg-white rounded-lg border border-gray-200 overflow-hidden"
          >
            <button
              onClick={() => setExpandedProtocol(
                expandedProtocol === protocol.id ? null : protocol.id
              )}
              className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-gray-50"
            >
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="font-medium text-gray-900">{protocol.name}</h3>
                  <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                    {protocol.intent}
                  </span>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  {protocol.regimens.map(r => r.name).join(' + ')}
                </p>
              </div>
              {expandedProtocol === protocol.id ? (
                <ChevronUp className="w-5 h-5 text-gray-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-400" />
              )}
            </button>

            {expandedProtocol === protocol.id && (
              <div className="px-4 py-3 border-t border-gray-200">
                <div className="space-y-4">
                  <div>
                    <h4 className="text-sm font-medium text-gray-700">Eligibility Criteria</h4>
                    <ul className="mt-2 space-y-1">
                      {protocol.eligibility.map((criterion, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                          {criterion}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {renderDrugInfo(protocol)}

                  <div>
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Monitoring</h4>
                    <ul className="space-y-1">
                      {protocol.monitoring.map((item, idx) => (
                        <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                          <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div className="pt-3 border-t border-gray-200">
                    <a
                      href={protocol.references[0]}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-blue-600 hover:text-blue-800"
                    >
                      View Full Protocol Document →
                    </a>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}

        {filteredProtocols.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            No protocols found matching your criteria.
          </div>
        )}
      </div>
    </div>
  );
};

export default ProtocolViewer;