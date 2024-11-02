import { Protocol } from '../types';

export const giProtocols: Protocol[] = [
  {
    id: 'GIGAVPOX',
    name: 'Palliative FOLFOX',
    intent: 'Palliative',
    category: 'gastrointestinal',
    eligibility: [
      'Metastatic colorectal cancer',
      'ECOG 0-2',
      'Adequate organ function',
      'No contraindications to chemotherapy'
    ],
    regimens: [
      {
        name: 'FOLFOX',
        drugs: [
          {
            name: 'Oxaliplatin',
            dose: '85 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'Leucovorin',
            dose: '400 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: '5-Fluorouracil',
            dose: '400 mg/m² bolus then 2400 mg/m² over 46h',
            route: 'IV',
            schedule: 'Days 1-2'
          }
        ],
        cycles: 12,
        interval: 14,
        supportiveCare: [
          'Antiemetics for moderately emetogenic chemotherapy',
          'Cold sensitivity precautions'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each cycle',
      'Hepatic function',
      'Neuropathy assessment',
      'CEA levels'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Gastrointestinal/GIGAVPOX_Protocol.pdf'
    ]
  },
  {
    id: 'GIGAVIRI',
    name: 'Palliative FOLFIRI',
    intent: 'Palliative',
    category: 'gastrointestinal',
    eligibility: [
      'Metastatic colorectal cancer',
      'ECOG 0-2',
      'Adequate organ function',
      'No contraindications to chemotherapy'
    ],
    regimens: [
      {
        name: 'FOLFIRI',
        drugs: [
          {
            name: 'Irinotecan',
            dose: '180 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'Leucovorin',
            dose: '400 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: '5-Fluorouracil',
            dose: '400 mg/m² bolus then 2400 mg/m² over 46h',
            route: 'IV',
            schedule: 'Days 1-2'
          }
        ],
        cycles: 12,
        interval: 14,
        supportiveCare: [
          'Antiemetics for moderately emetogenic chemotherapy',
          'Atropine for cholinergic symptoms',
          'Loperamide for delayed diarrhea'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each cycle',
      'Hepatic function',
      'Diarrhea assessment',
      'CEA levels'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Gastrointestinal/GIGAVIRI_Protocol.pdf'
    ]
  }
];