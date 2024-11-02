interface DrugProtocol {
  name: string;
  regimen: string;
  frequency: string;
  duration: string;
  premedications?: string[];
  supportiveCare?: string[];
  monitoring?: string[];
}

interface Protocol {
  intent: 'Curative' | 'Palliative' | 'Neoadjuvant' | 'Adjuvant';
  name: string;
  description: string;
  eligibility: string[];
  protocols: DrugProtocol[];
  references: string[];
}

interface CancerProtocols {
  [key: string]: {
    name: string;
    protocols: Protocol[];
  };
}

export const protocolDatabase: CancerProtocols = {
  'breast': {
    name: 'Breast Cancer',
    protocols: [
      {
        intent: 'Curative',
        name: 'BRAJAC',
        description: 'Adjuvant therapy for high-risk breast cancer',
        eligibility: [
          'Node positive or high-risk node negative',
          'HER2 negative',
          'Adequate organ function'
        ],
        protocols: [
          {
            name: 'AC-T',
            regimen: 'DOXOrubicin 60 mg/m² + cyclophosphamide 600 mg/m² q3w × 4, followed by paclitaxel 80 mg/m² weekly × 12',
            frequency: 'Every 3 weeks',
            duration: '4 cycles AC, then 12 weekly cycles of T',
            premedications: [
              'dexamethasone 8mg PO',
              'ondansetron 8mg PO'
            ],
            supportiveCare: [
              'G-CSF support as per guidelines',
              'Antiemetic protocol for highly emetogenic chemotherapy'
            ],
            monitoring: [
              'CBC prior to each cycle',
              'LVEF monitoring',
              'Neuropathy assessment'
            ]
          }
        ],
        references: ['BC Cancer Protocol Summary for Breast Cancer']
      }
    ]
  },
  'lung': {
    name: 'Lung Cancer',
    protocols: [
      {
        intent: 'Palliative',
        name: 'LUAVPC',
        description: 'First-line treatment for advanced NSCLC',
        eligibility: [
          'Stage IV NSCLC',
          'No targetable mutations',
          'PD-L1 ≥ 50%'
        ],
        protocols: [
          {
            name: 'Pembrolizumab Monotherapy',
            regimen: 'Pembrolizumab 200mg IV',
            frequency: 'Every 3 weeks',
            duration: 'Up to 2 years',
            premedications: [
              'acetaminophen 650mg PO',
              'diphenhydramine 25mg PO'
            ],
            supportiveCare: [
              'Thyroid function monitoring',
              'Pneumonitis monitoring'
            ],
            monitoring: [
              'CT scan every 9 weeks',
              'Thyroid function tests q3 months',
              'Liver function monitoring'
            ]
          }
        ],
        references: ['BC Cancer Protocol Summary for Lung Cancer']
      }
    ]
  },
  'colorectal': {
    name: 'Colorectal Cancer',
    protocols: [
      {
        intent: 'Palliative',
        name: 'GIFOLFOX',
        description: 'First-line treatment for metastatic colorectal cancer',
        eligibility: [
          'Metastatic colorectal cancer',
          'Good performance status',
          'Adequate organ function'
        ],
        protocols: [
          {
            name: 'FOLFOX',
            regimen: 'Oxaliplatin 85 mg/m² + leucovorin 400 mg/m² + 5-FU bolus 400 mg/m² + 5-FU infusion 2400 mg/m²',
            frequency: 'Every 2 weeks',
            duration: '12 cycles',
            premedications: [
              'ondansetron 8mg PO',
              'dexamethasone 8mg PO'
            ],
            supportiveCare: [
              'Cold sensitivity precautions',
              'Neuropathy monitoring'
            ],
            monitoring: [
              'CBC prior to each cycle',
              'CEA every 6 weeks',
              'CT scan every 8 weeks'
            ]
          }
        ],
        references: ['BC Cancer Protocol Summary for Gastrointestinal Cancer']
      }
    ]
  }
};