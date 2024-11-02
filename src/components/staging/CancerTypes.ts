export interface StagingCriteria {
  name: string;
  description: string;
  required: string[];
  optional?: string[];
}

export const cancerTypes: Record<string, StagingCriteria> = {
  'breast': {
    name: 'Breast Cancer',
    description: 'TNM staging system for breast cancer',
    required: [
      'Tumor size (cm)',
      'Lymph node involvement',
      'Distant metastasis',
      'Histological grade',
      'ER/PR status',
      'HER2 status',
      'Ki-67 index'
    ],
    optional: [
      'Oncotype DX',
      'MammaPrint',
      'BRCA status'
    ]
  },
  'lung': {
    name: 'Lung Cancer',
    description: 'TNM staging system for lung cancer',
    required: [
      'Tumor size and invasion',
      'Lymph node status',
      'Distant metastasis',
      'Histological type',
      'PD-L1 expression',
      'EGFR mutation',
      'ALK rearrangement'
    ],
    optional: [
      'ROS1 fusion',
      'KRAS mutation',
      'BRAF mutation'
    ]
  },
  'colorectal': {
    name: 'Colorectal Cancer',
    description: 'TNM staging system for colorectal cancer',
    required: [
      'Tumor invasion depth',
      'Lymph node involvement',
      'Distant metastasis',
      'MSI status',
      'KRAS mutation',
      'BRAF mutation',
      'Tumor location'
    ],
    optional: [
      'CEA level',
      'MMR status',
      'Tumor budding'
    ]
  }
}