import React, { useState } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('classification');
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: '',
    day: '',
    month: '',
    year: '',
    time: ''
  });
  const [result, setResult] = useState(null);

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (activeTab === 'classification') {
      const mockPrediction = Math.random() > 0.5 ? 'Fire Detected' : 'No Fire';
      const confidence = (Math.random() * 30 + 70).toFixed(2);
      setResult({
        type: 'classification',
        prediction: mockPrediction,
        confidence: confidence,
        riskLevel: mockPrediction === 'Fire Detected' ? 'High' : 'Low'
      });
    } else {
      const area = (Math.random() * 500 + 50).toFixed(2);
      const severity = area > 300 ? 'Severe' : area > 150 ? 'Moderate' : 'Low';
      setResult({
        type: 'regression',
        area: area,
        severity: severity,
        estimatedDamage: (area * 1.5).toFixed(2)
      });
    }
  };

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-700 via-green-800 to-emerald-900 p-4 md:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-2xl shadow-2xl p-6 mb-6 text-center">
          <div className="flex items-center justify-center gap-3 mb-2">
            <span className="text-5xl">🔥</span>
            <h1 className="text-3xl md:text-4xl font-bold text-gray-800">
              Forest Fire Trend Analysis
            </h1>
          </div>
          <p className="text-gray-600">ML-powered prediction and area estimation system</p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl mb-6 p-2 flex gap-2">
          <button
            onClick={() => {
              setActiveTab('classification');
              setResult(null);
            }}
            className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
              activeTab === 'classification'
                ? 'bg-gradient-to-r from-orange-500 to-red-500 text-white shadow-lg'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <span className="text-xl">📊</span>
            Classification
          </button>
          <button
            onClick={() => {
              setActiveTab('regression');
              setResult(null);
            }}
            className={`flex-1 py-3 px-6 rounded-xl font-semibold transition-all duration-300 flex items-center justify-center gap-2 ${
              activeTab === 'regression'
                ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            <span className="text-xl">📈</span>
            Regression
          </button>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-6 md:p-8">
          <div className={`mb-6 p-4 rounded-lg border-l-4 ${
            activeTab === 'classification'
              ? 'bg-orange-50 border-orange-500'
              : 'bg-blue-50 border-blue-500'
          }`}>
            <p className="text-sm font-medium text-gray-700">
              {activeTab === 'classification' 
                ? '🔍 Classification: Predict whether a fire will occur or not'
                : '📊 Regression: Estimate the area affected by the fire spread (in hectares)'}
            </p>
          </div>

          <div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <span className="text-xl">📍</span>
                Geographical Location
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Latitude
                  </label>
                  <input
                    type="number"
                    name="latitude"
                    value={formData.latitude}
                    onChange={handleInputChange}
                    step="0.000001"
                    placeholder="e.g., 37.7749"
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Longitude
                  </label>
                  <input
                    type="number"
                    name="longitude"
                    value={formData.longitude}
                    onChange={handleInputChange}
                    step="0.000001"
                    placeholder="e.g., -122.4194"
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                    required
                  />
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <span className="text-xl">📅</span>
                Date Information
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Day
                  </label>
                  <input
                    type="number"
                    name="day"
                    value={formData.day}
                    onChange={handleInputChange}
                    min="1"
                    max="31"
                    placeholder="1-31"
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Month
                  </label>
                  <select
                    name="month"
                    value={formData.month}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                    required
                  >
                    <option value="">Select Month</option>
                    {months.map((month, index) => (
                      <option key={month} value={index + 1}>
                        {month}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Year
                  </label>
                  <input
                    type="number"
                    name="year"
                    value={formData.year}
                    onChange={handleInputChange}
                    min="2000"
                    max="2100"
                    placeholder="e.g., 2025"
                    className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                    required
                  />
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <span className="text-xl">🕐</span>
                Time of Day
              </h3>
              <input
                type="time"
                name="time"
                value={formData.time}
                onChange={handleInputChange}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:outline-none transition-colors"
                required
              />
            </div>

            <button
              onClick={handleSubmit}
              className={`w-full py-4 rounded-lg font-semibold text-white shadow-lg transition-all duration-300 hover:shadow-xl hover:-translate-y-1 ${
                activeTab === 'classification'
                  ? 'bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600'
                  : 'bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600'
              }`}
            >
              {activeTab === 'classification' ? '🔍 Predict Fire Occurrence' : '📊 Estimate Fire Spread Area'}
            </button>
          </div>

          {result && (
            <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-gray-100 rounded-xl border-2 border-gray-200 animate-fadeIn">
              <h3 className="text-xl font-bold text-gray-800 mb-4">
                {result.type === 'classification' ? '🎯 Classification Result' : '📈 Regression Result'}
              </h3>
              
              {result.type === 'classification' ? (
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Prediction:</span>
                    <span className={`font-bold text-lg ${
                      result.prediction === 'Fire Detected' ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {result.prediction}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Confidence:</span>
                    <span className="font-bold text-lg text-blue-600">{result.confidence}%</span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Risk Level:</span>
                    <span className={`font-bold text-lg ${
                      result.riskLevel === 'High' ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {result.riskLevel}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Estimated Area:</span>
                    <span className="font-bold text-lg text-orange-600">{result.area} hectares</span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Severity:</span>
                    <span className={`font-bold text-lg ${
                      result.severity === 'Severe' ? 'text-red-600' : 
                      result.severity === 'Moderate' ? 'text-orange-600' : 'text-yellow-600'
                    }`}>
                      {result.severity}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-4 bg-white rounded-lg shadow-sm">
                    <span className="font-semibold text-gray-700">Estimated Damage:</span>
                    <span className="font-bold text-lg text-purple-600">${result.estimatedDamage}K</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="mt-6 text-center text-white">
          <p className="text-sm">🌲 Forest Fire Trend Analysis System | Powered by Machine Learning 🔥</p>
        </div>
      </div>
    </div>
  );
}