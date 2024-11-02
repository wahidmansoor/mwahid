interface TreatmentItem {
  treatment: string;
  details?: string;
}

interface TreatmentSection {
  title: string;
  items: TreatmentItem[];
}

interface TreatmentPlan {
  disclaimer?: string;
  sections: TreatmentSection[];
  additionalConsiderations?: string[];
}

const getBreastCancerTreatment = (stagingData: Record<string, string>): TreatmentPlan => {
  const hasMetastasis = stagingData['Distant metastasis']?.toLowerCase().includes('yes');
  const erPrStatus = stagingData['ER/PR status']?.toLowerCase() || '';
  const her2Status = stagingData['HER2 status']?.toLowerCase() || '';
  
  if (hasMetastasis) {
    return {
      disclaimer: "Treatment plan for metastatic breast cancer should be individualized based on multiple factors.",
      sections: [
        {
          title: "Systemic Therapy Options",
          items: [
            {
              treatment: "Hormone Therapy",
              details: "For HR-positive disease: Consider aromatase inhibitors, tamoxifen, or fulvestrant"
            },
            {
              treatment: "Targeted Therapy",
              details: "For HER2-positive: Trastuzumab + pertuzumab + chemotherapy"
            },
            {
              treatment: "Chemotherapy",
              details: "Consider single-agent or combination based on disease burden"
            }
          ]
        },
        {
          title: "Supportive Care",
          items: [
            {
              treatment: "Bone-Targeted Therapy",
              details: "Zoledronic acid or denosumab for bone metastases"
            },
            {
              treatment: "Pain Management",
              details: "Individualized pain management protocol"
            }
          ]
        }
      ],
      additionalConsiderations: [
        "Regular monitoring of treatment response",
        "Quality of life assessments",
        "Clinical trial opportunities",
        "Genetic counseling if indicated"
      ]
    };
  }

  // Early-stage breast cancer
  return {
    sections: [
      {
        title: "Primary Treatment",
        items: [
          {
            treatment: "Surgery",
            details: "Breast-conserving surgery or mastectomy based on tumor size and location"
          }
        ]
      },
      {
        title: "Adjuvant Therapy",
        items: [
          {
            treatment: "Radiation Therapy",
            details: "Post-lumpectomy or post-mastectomy as indicated"
          },
          {
            treatment: "Systemic Therapy",
            details: her2Status.includes('positive') 
              ? "HER2-targeted therapy + chemotherapy"
              : "Chemotherapy based on risk factors"
          }
        ]
      }
    ],
    additionalConsiderations: [
      "Consider genetic testing if indicated",
      "Discuss fertility preservation if applicable",
      "Regular follow-up schedule",
      "Lifestyle modifications and supportive care"
    ]
  };
};

const getLungCancerTreatment = (stagingData: Record<string, string>): TreatmentPlan => {
  const hasMetastasis = stagingData['Distant metastasis']?.toLowerCase().includes('yes');
  const pdl1Expression = stagingData['PD-L1 expression'] || '';
  
  return {
    disclaimer: "Treatment recommendations should be discussed in a multidisciplinary tumor board.",
    sections: [
      {
        title: "Primary Treatment",
        items: hasMetastasis ? [
          {
            treatment: "Systemic Therapy",
            details: "Based on molecular profile and PD-L1 status"
          }
        ] : [
          {
            treatment: "Surgery",
            details: "Lobectomy with lymph node dissection for early-stage disease"
          }
        ]
      },
      {
        title: "Additional Therapy",
        items: [
          {
            treatment: "Radiation Therapy",
            details: hasMetastasis 
              ? "Palliative radiation for symptomatic sites"
              : "Consider adjuvant radiation based on pathologic findings"
          },
          {
            treatment: "Immunotherapy",
            details: Number(pdl1Expression) >= 50 
              ? "Consider pembrolizumab as first-line therapy"
              : "Consider in combination with chemotherapy"
          }
        ]
      }
    ],
    additionalConsiderations: [
      "Smoking cessation support",
      "Pulmonary rehabilitation",
      "Regular imaging follow-up",
      "Supportive care integration"
    ]
  };
};

const getColorectalTreatment = (stagingData: Record<string, string>): TreatmentPlan => {
  const hasMetastasis = stagingData['Distant metastasis']?.toLowerCase().includes('yes');
  const msiStatus = stagingData['MSI status']?.toLowerCase() || '';
  
  return {
    sections: [
      {
        title: "Primary Treatment",
        items: [
          {
            treatment: "Surgery",
            details: hasMetastasis 
              ? "Consider surgical options for oligometastatic disease"
              : "Colectomy with lymph node dissection"
          }
        ]
      },
      {
        title: "Systemic Therapy",
        items: [
          {
            treatment: "Chemotherapy",
            details: hasMetastasis 
              ? "FOLFOX or FOLFIRI based regimens"
              : "Adjuvant chemotherapy based on risk factors"
          },
          {
            treatment: "Targeted Therapy",
            details: "Consider based on molecular profile (RAS/BRAF status)"
          }
        ]
      },
      {
        title: "Additional Considerations",
        items: [
          {
            treatment: "Immunotherapy",
            details: msiStatus.includes('high') 
              ? "Consider checkpoint inhibitors"
              : "Standard chemotherapy protocols"
          }
        ]
      }
    ],
    additionalConsiderations: [
      "Regular CEA monitoring",
      "Surveillance colonoscopy schedule",
      "Nutritional support",
      "Genetic counseling if indicated"
    ]
  };
};

export const getTreatmentPlan = (
  cancerType: string,
  stagingData: Record<string, string>
): TreatmentPlan => {
  switch (cancerType) {
    case 'breast':
      return getBreastCancerTreatment(stagingData);
    case 'lung':
      return getLungCancerTreatment(stagingData);
    case 'colorectal':
      return getColorectalTreatment(stagingData);
    default:
      return {
        disclaimer: "Please select a specific cancer type for treatment recommendations.",
        sections: []
      };
  }
};