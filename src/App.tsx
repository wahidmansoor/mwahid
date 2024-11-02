import React, { useState } from 'react';
import { MessageSquare, Users, Building2, Syringe, Menu, Send, Plus } from 'lucide-react';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';

function App() {
  const [selectedUnit, setSelectedUnit] = useState('outpatient');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className={`${isSidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 ease-in-out`}>
        <Sidebar 
          selectedUnit={selectedUnit} 
          setSelectedUnit={setSelectedUnit}
          isSidebarOpen={isSidebarOpen}
        />
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Navigation */}
        <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <Menu className="w-5 h-5 text-gray-600" />
            </button>
            <h1 className="text-xl font-semibold text-gray-800">
              OncologyGPT Assistant
            </h1>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-4 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2">
              <Plus className="w-4 h-4" />
              New Chat
            </button>
          </div>
        </header>

        {/* Chat Interface */}
        <ChatInterface unit={selectedUnit} />
      </div>
    </div>
  );
}

export default App;