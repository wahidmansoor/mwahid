import { Protocol } from '../types';

export const lungProtocols: Protocol[] = [
  {
    id: 'LUAVPC',
    name: 'Palliative Carboplatin and Paclitaxel',
    intent: 'Palliative',
    category: 'lung',
    eligibility: [
      'Stage IV NSCLC',
      'ECOG 0-2',
      'Adequate organ function',
      'No contraindications to chemotherapy'
    ],
    regimens: [
      {
        name: 'Carboplatin-Paclitaxel',
        drugs: [
          {
            name: 'Carboplatin',
            dose: 'AUC 5-6',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'PACLitaxel',
            dose: '200 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          }
        ],
        cycles: 4,
        interval: 21,
        supportiveCare: [
          'Antiemetics for moderately emetogenic chemotherapy',
          'Premedication for hypersensitivity'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each cycle',
      'Renal function',
      'Neuropathy assessment',
      'Radiologic assessment q2cycles'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Lung/LUAVPC_Protocol.pdf'
    ]
  },
  {
    id: 'LUAVPG',
    name: 'Palliative Pemetrexed-Platinum',
    intent: 'Palliative',
    category: 'lung',
    eligibility: [
      'Non-squamous NSCLC',
      'ECOG 0-2',
      'Adequate organ function',
      'No contraindications to chemotherapy'
    ],
    regimens: [
      {
        name: 'Pemetrexed-Platinum',
        drugs: [
          {
            name: 'Pemetrexed',
            dose: '500 mg/m²',
            route: 'IV',
            schedule: 'Day 1'
          },
          {
            name: 'CISplatin or CARBOplatin',
            dose: '75 mg/m² or AUC 5',
            route: 'IV',
            schedule: 'Day 1'
          }
        ],
        cycles: 4,
        interval: 21,
        supportiveCare: [
          'Folic acid supplementation',
          'Vitamin B12 supplementation',
          'Dexamethasone prophylaxis'
        ]
      }
    ],
    monitoring: [
      'CBC prior to each cycle',
      'Renal function',
      'Vitamin B12 levels',
      'Radiologic assessment q2cycles'
    ],
    references: [
      'http://www.bccancer.bc.ca/chemotherapy-protocols-site/Documents/Lung/LUAVPG_Protocol.pdf'
    ]
  }
];