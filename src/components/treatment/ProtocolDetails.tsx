import React from 'react';
import { Clock, Calendar, AlertCircle, CheckCircle2, Activity } from 'lucide-react';

interface ProtocolDetailsProps {
  protocol: any;
}

const ProtocolDetails: React.FC<ProtocolDetailsProps> = ({ protocol }) => {
  return (
    <div className="bg-white rounded-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200 bg-gray-50">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-lg font-medium text-gray-900">{protocol.name}</h2>
            <p className="text-sm text-gray-600 mt-1">{protocol.description}</p>
          </div>
          <span className="px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
            {protocol.intent}
          </span>
        </div>
      </div>

      <div className="p-4 space-y-6">
        <div>
          <h3 className="font-medium text-gray-900 mb-2">Eligibility Criteria</h3>
          <ul className="space-y-2">
            {protocol.eligibility.map((criterion: string, index: number) => (
              <li key={index} className="flex items-start gap-2 text-sm text-gray-600">
                <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5" />
                <span>{criterion}</span>
              </li>
            ))}
          </ul>
        </div>

        {protocol.protocols.map((drugProtocol: any, index: number) => (
          <div key={index} className="space-y-4">
            <h3 className="font-medium text-gray-900">{drugProtocol.name}</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Activity className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-gray-700">Regimen:</span>
                <span className="text-gray-600">{drugProtocol.regimen}</span>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-gray-700">Frequency:</span>
                <span className="text-gray-600">{drugProtocol.frequency}</span>
              </div>
              
              <div className="flex items-center gap-2 text-sm">
                <Calendar className="w-4 h-4 text-gray-500" />
                <span className="font-medium text-gray-700">Duration:</span>
                <span className="text-gray-600">{drugProtocol.duration}</span>
              </div>
            </div>

            {drugProtocol.premedications && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Premedications</h4>
                <ul className="space-y-1">
                  {drugProtocol.premedications.map((med: string, idx: number) => (
                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                      {med}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {drugProtocol.supportiveCare && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Supportive Care</h4>
                <ul className="space-y-1">
                  {drugProtocol.supportiveCare.map((care: string, idx: number) => (
                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                      {care}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {drugProtocol.monitoring && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Monitoring</h4>
                <ul className="space-y-1">
                  {drugProtocol.monitoring.map((item: string, idx: number) => (
                    <li key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}

        <div className="pt-4 border-t border-gray-200">
          <div className="flex items-start gap-2 text-sm text-gray-600">
            <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5" />
            <p>These protocols are guidelines only. Treatment should be individualized based on patient factors and institutional policies.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProtocolDetails;