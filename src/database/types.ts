export interface Drug {
  name: string;
  dose: string;
  route: string;
  schedule: string;
}

export interface Regimen {
  name: string;
  drugs: Drug[];
  cycles: number;
  interval: number;
  supportiveCare: string[];
}

export interface Protocol {
  id: string;
  name: string;
  intent: 'Curative' | 'Palliative' | 'Neoadjuvant' | 'Adjuvant';
  category: string;
  eligibility: string[];
  regimens: Regimen[];
  monitoring: string[];
  references: string[];
}

export interface ProtocolDatabase {
  breast: Protocol[];
  lung: Protocol[];
  gastrointestinal: Protocol[];
}