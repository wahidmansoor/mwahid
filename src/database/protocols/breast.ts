import { Protocol } from '../types';

export const breastProtocols: Protocol[] = [
  {
    id: 'BRAJACT',
    name: 'Adjuvant AC-T',
    intent: 'Curative',
    category: 'breast',
    eligibility: [
      'Node positive or high-risk node negative',
      'HER2 negative',
      'ECOG 0-1',
      'Adequate organ function'
    ],
    regimens: [
      {
        name: 'AC',
        drugs: [
          {
            name: 'DOXOrubicin',
            dose: '60 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'Cyclophosphamide',
            dose: '600 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          }
        ],
        cycles: 4,
        interval: 21,
        supportiveCare: [
          'G-CSF support as per guidelines',
          'Antiemetics for highly emetogenic chemotherapy'
        ]
      },
      {
        name: 'T',
        drugs: [
          {
            name: 'PACLitaxel',
            dose: '80 mg/m²',
            route: 'IV',
            schedule: 'Days 1, 8, and 15'
          }
        ],
        cycles: 4,
        interval: 7,
        supportiveCare: [
          'Premedication for hypersensitivity',
          'Monitor for neuropathy'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each treatment',
      'Cardiac monitoring (LVEF)',
      'Hepatic function',
      'Neuropathy assessment'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Breast/BRAJACT_Protocol.pdf'
    ]
  },
  {
    id: 'BRAJACTT',
    name: 'Adjuvant AC-TH (Trastuzumab)',
    intent: 'Curative',
    category: 'breast',
    eligibility: [
      'Node positive or high-risk node negative',
      'HER2 positive',
      'LVEF ≥ 55%',
      'No significant cardiac history'
    ],
    regimens: [
      {
        name: 'AC',
        drugs: [
          {
            name: 'DOXOrubicin',
            dose: '60 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'Cyclophosphamide',
            dose: '600 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          }
        ],
        cycles: 4,
        interval: 21,
        supportiveCare: [
          'G-CSF support as per guidelines',
          'Antiemetics for highly emetogenic chemotherapy'
        ]
      },
      {
        name: 'TH',
        drugs: [
          {
            name: 'PACLitaxel',
            dose: '80 mg/m²',
            route: 'IV',
            schedule: 'Weekly'
          },
          {
            name: 'Trastuzumab',
            dose: '8 mg/kg loading, then 6 mg/kg',
            route: 'IV',
            schedule: 'Every 21 days'
          }
        ],
        cycles: 12,
        interval: 7,
        supportiveCare: [
          'Cardiac monitoring every 3 months',
          'Premedication for hypersensitivity'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each treatment',
      'Cardiac monitoring (LVEF) q3monthly',
      'Hepatic function',
      'Neuropathy assessment'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Breast/BRAJACTT_Protocol.pdf'
    ]
  }
];