import { breastProtocols } from './protocols/breast';
import { lungProtocols } from './protocols/lung';
import { giProtocols } from './protocols/gastrointestinal';
import type { ProtocolDatabase } from './types';

export const protocolDatabase: ProtocolDatabase = {
  breast: breastProtocols,
  lung: lungProtocols,
  gastrointestinal: giProtocols
};

export function getProtocolById(id: string): Protocol | undefined {
  const allProtocols = [
    ...breastProtocols,
    ...lungProtocols,
    ...giProtocols
  ];
  return allProtocols.find(protocol => protocol.id === id);
}

export function getProtocolsByCategory(category: string): Protocol[] {
  return protocolDatabase[category as keyof ProtocolDatabase] || [];
}

export function searchProtocols(query: string): Protocol[] {
  const allProtocols = [
    ...breastProtocols,
    ...lungProtocols,
    ...giProtocols
  ];
  
  const searchTerms = query.toLowerCase().split(' ');
  
  return allProtocols.filter(protocol => {
    const searchText = `
      ${protocol.name.toLowerCase()}
      ${protocol.intent.toLowerCase()}
      ${protocol.category.toLowerCase()}
      ${protocol.regimens.map(r => r.name.toLowerCase()).join(' ')}
      ${protocol.regimens.map(r => r.drugs.map(d => d.name.toLowerCase()).join(' ')).join(' ')}
    `;
    
    return searchTerms.every(term => searchText.includes(term));
  });
}