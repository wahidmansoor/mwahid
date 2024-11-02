import React from 'react';
import { ChevronRight } from 'lucide-react';

interface Question {
  id: string;
  text: string;
  category: string;
}

interface LeadingQuestionsProps {
  unit: string;
  onQuestionSelect: (question: string) => void;
}

const LeadingQuestions: React.FC<LeadingQuestionsProps> = ({ unit, onQuestionSelect }) => {
  const getQuestionsByUnit = (unit: string): Question[] => {
    switch (unit) {
      case 'outpatient':
        return [
          { id: 'staging', text: 'Help me stage a new cancer patient', category: 'Diagnosis' },
          { id: 'followup', text: 'What follow-up schedule should I recommend?', category: 'Follow-up' },
          { id: 'symptoms', text: 'Which symptoms need urgent attention?', category: 'Monitoring' },
          { id: 'referral', text: 'When should I refer to a specialist?', category: 'Referral' }
        ];
      case 'chemotherapy':
        return [
          { id: 'protocol', text: 'Show me chemotherapy protocols', category: 'Treatment' },
          { id: 'side-effects', text: 'How to manage side effects?', category: 'Management' },
          { id: 'monitoring', text: 'What should we monitor during treatment?', category: 'Monitoring' },
          { id: 'emergency', text: 'What are the emergency protocols?', category: 'Emergency' }
        ];
      case 'inpatient':
        return [
          { id: 'admission', text: 'When should we admit a patient?', category: 'Admission' },
          { id: 'pain', text: 'How to manage severe pain?', category: 'Pain' },
          { id: 'complications', text: 'Managing common complications', category: 'Complications' },
          { id: 'discharge', text: 'When is it safe to discharge?', category: 'Discharge' }
        ];
      case 'palliative':
        return [
          { id: 'comfort', text: 'Best practices for comfort care', category: 'Care' },
          { id: 'pain-end', text: 'End-of-life pain management', category: 'Pain' },
          { id: 'family', text: 'Supporting family members', category: 'Support' },
          { id: 'symptoms-end', text: 'Managing end-of-life symptoms', category: 'Symptoms' }
        ];
      default:
        return [];
    }
  };

  const questions = getQuestionsByUnit(unit);
  const groupedQuestions = questions.reduce((acc, question) => {
    if (!acc[question.category]) {
      acc[question.category] = [];
    }
    acc[question.category].push(question);
    return acc;
  }, {} as Record<string, Question[]>);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {Object.entries(groupedQuestions).map(([category, questions]) => (
        <div key={category} className="space-y-2">
          <h3 className="text-sm font-medium text-gray-500">{category}</h3>
          <div className="space-y-2">
            {questions.map((question) => (
              <button
                key={question.id}
                onClick={() => onQuestionSelect(question.text)}
                className="w-full text-left p-3 bg-white rounded-lg border border-gray-200 hover:border-blue-500 hover:bg-blue-50 transition-colors flex items-center justify-between group"
              >
                <span className="text-sm text-gray-700 group-hover:text-blue-700">
                  {question.text}
                </span>
                <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-blue-500" />
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default LeadingQuestions;