import React, { useState } from 'react';
import { CheckCircle2, Circle, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { StagingCriteria } from './CancerTypes';

interface StagingChecklistProps {
  criteria: StagingCriteria;
  onComplete: (isComplete: boolean, data: Record<string, string>) => void;
}

const StagingChecklist: React.FC<StagingChecklistProps> = ({ criteria, onComplete }) => {
  const [checkedItems, setCheckedItems] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState(true);

  const handleInputChange = (item: string, value: string) => {
    const newCheckedItems = {
      ...checkedItems,
      [item]: value
    };
    setCheckedItems(newCheckedItems);

    // Check if all required items are completed
    const isComplete = criteria.required.every(req => newCheckedItems[req]?.trim());
    onComplete(isComplete, newCheckedItems);
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-3 flex items-center justify-between bg-gray-50 border-b border-gray-200"
      >
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-700">{criteria.name}</span>
          {criteria.required.every(req => checkedItems[req]?.trim()) ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : (
            <AlertCircle className="w-4 h-4 text-amber-500" />
          )}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        )}
      </button>

      {expanded && (
        <div className="p-4 space-y-4">
          <p className="text-sm text-gray-600 mb-4">{criteria.description}</p>
          
          <div className="space-y-3">
            <h3 className="font-medium text-gray-700">Required Criteria</h3>
            {criteria.required.map((item) => (
              <div key={item} className="flex items-start gap-3">
                {checkedItems[item]?.trim() ? (
                  <CheckCircle2 className="w-5 h-5 text-green-500 mt-2" />
                ) : (
                  <Circle className="w-5 h-5 text-gray-300 mt-2" />
                )}
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {item}
                  </label>
                  <input
                    type="text"
                    value={checkedItems[item] || ''}
                    onChange={(e) => handleInputChange(item, e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    placeholder={`Enter ${item.toLowerCase()}`}
                  />
                </div>
              </div>
            ))}
          </div>

          {criteria.optional && (
            <div className="space-y-3 mt-6">
              <h3 className="font-medium text-gray-700">Optional Criteria</h3>
              {criteria.optional.map((item) => (
                <div key={item} className="flex items-start gap-3">
                  {checkedItems[item]?.trim() ? (
                    <CheckCircle2 className="w-5 h-5 text-green-500 mt-2" />
                  ) : (
                    <Circle className="w-5 h-5 text-gray-300 mt-2" />
                  )}
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      {item}
                    </label>
                    <input
                      type="text"
                      value={checkedItems[item] || ''}
                      onChange={(e) => handleInputChange(item, e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                      placeholder={`Enter ${item.toLowerCase()} (optional)`}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default StagingChecklist;