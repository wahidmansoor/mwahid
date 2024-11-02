import React from 'react';
import { Users, Building2, Syringe, MessageSquare, Heart } from 'lucide-react';

interface SidebarProps {
  selectedUnit: string;
  setSelectedUnit: (unit: string) => void;
  isSidebarOpen: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ selectedUnit, setSelectedUnit, isSidebarOpen }) => {
  const units = [
    {
      id: 'outpatient',
      name: 'Outpatient Clinic',
      icon: Users,
      description: 'Manage outpatient consultations and follow-ups'
    },
    {
      id: 'chemotherapy',
      name: 'Chemotherapy Unit',
      icon: Syringe,
      description: 'Day unit treatment protocols and management'
    },
    {
      id: 'inpatient',
      name: 'Inpatient Care',
      icon: Building2,
      description: 'Inpatient care and monitoring'
    },
    {
      id: 'palliative',
      name: 'Palliative Care',
      icon: Heart,
      description: 'Supportive and end-of-life care management'
    }
  ];

  if (!isSidebarOpen) return null;

  return (
    <div className="h-full bg-white border-r border-gray-200 p-4">
      <div className="space-y-4">
        <div className="px-3 py-2">
          <div className="flex items-center gap-2 text-blue-600">
            <MessageSquare className="w-5 h-5" />
            <span className="font-semibold">Care Units</span>
          </div>
        </div>

        {units.map((unit) => {
          const Icon = unit.icon;
          return (
            <button
              key={unit.id}
              onClick={() => setSelectedUnit(unit.id)}
              className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                selectedUnit === unit.id
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              <div className="flex items-center gap-3">
                <Icon className="w-5 h-5" />
                <div>
                  <div className="font-medium">{unit.name}</div>
                  <div className="text-xs text-gray-500">{unit.description}</div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="absolute bottom-0 left-0 right-0 p-4 border-t">
        <div className="text-xs text-gray-500 text-center">
          Medical AI Assistant v1.0
        </div>
      </div>
    </div>
  );
}

export default Sidebar;