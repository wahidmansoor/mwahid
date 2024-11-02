import React, { useState } from 'react';
import { cancerTypes } from './CancerTypes';
import StagingChecklist from './StagingChecklist';

interface StagingSystemProps {
  onStagingComplete: (data: {
    cancerType: string;
    stagingData: Record<string, string>;
  }) => void;
}

const StagingSystem: React.FC<StagingSystemProps> = ({ onStagingComplete }) => {
  const [selectedCancer, setSelectedCancer] = useState<string>('');
  const [stagingData, setStagingData] = useState<Record<string, string>>({});

  const handleCancerSelect = (cancer: string) => {
    setSelectedCancer(cancer);
    setStagingData({});
  };

  const handleStagingComplete = (isComplete: boolean, data: Record<string, string>) => {
    setStagingData(data);
    if (isComplete) {
      onStagingComplete({
        cancerType: selectedCancer,
        stagingData: data
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-2">
        <label className="block text-sm font-medium text-gray-700">
          Select Cancer Type
        </label>
        <select
          value={selectedCancer}
          onChange={(e) => handleCancerSelect(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select a cancer type...</option>
          {Object.entries(cancerTypes).map(([key, value]) => (
            <option key={key} value={key}>
              {value.name}
            </option>
          ))}
        </select>
      </div>

      {selectedCancer && (
        <StagingChecklist
          criteria={cancerTypes[selectedCancer]}
          onComplete={handleStagingComplete}
        />
      )}
    </div>
  );
};

export default StagingSystem;