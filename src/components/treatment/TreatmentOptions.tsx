import React from 'react';
import { ClipboardList, AlertCircle } from 'lucide-react';
import { getTreatmentPlan } from './treatmentPlans';

interface TreatmentOptionsProps {
  cancerType: string;
  stagingData: Record<string, string>;
}

const TreatmentOptions: React.FC<TreatmentOptionsProps> = ({ cancerType, stagingData }) => {
  const treatmentPlan = getTreatmentPlan(cancerType, stagingData);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-2">
          <ClipboardList className="w-5 h-5 text-blue-600" />
          <h3 className="font-medium text-gray-800">Recommended Treatment Plan</h3>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {treatmentPlan.disclaimer && (
          <div className="flex items-start gap-2 p-3 bg-amber-50 text-amber-800 rounded-lg text-sm">
            <AlertCircle className="w-5 h-5 mt-0.5 flex-shrink-0" />
            <p>{treatmentPlan.disclaimer}</p>
          </div>
        )}

        {treatmentPlan.sections.map((section, index) => (
          <div key={index} className="space-y-2">
            <h4 className="font-medium text-gray-700">{section.title}</h4>
            <div className="pl-4 border-l-2 border-gray-200">
              {section.items.map((item, itemIndex) => (
                <div key={itemIndex} className="py-2">
                  <div className="flex items-start gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-blue-600 mt-2" />
                    <div>
                      <p className="text-gray-800">{item.treatment}</p>
                      {item.details && (
                        <p className="text-sm text-gray-600 mt-1">{item.details}</p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}

        {treatmentPlan.additionalConsiderations && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <h4 className="font-medium text-gray-700 mb-2">Additional Considerations</h4>
            <ul className="space-y-2 text-sm text-gray-600">
              {treatmentPlan.additionalConsiderations.map((item, index) => (
                <li key={index} className="flex items-start gap-2">
                  <div className="w-1 h-1 rounded-full bg-gray-400 mt-2" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default TreatmentOptions;