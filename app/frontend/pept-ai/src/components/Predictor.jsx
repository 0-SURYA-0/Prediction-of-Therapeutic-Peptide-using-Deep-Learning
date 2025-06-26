import React, { useState } from 'react';

function Predictor() {
  const [sequence, setSequence] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null); // Reset previous predictions

    // Validate input on frontend
    const validAminoAcids = /^[ACDEFGHIKLMNPQRSTVWY]+$/i;
    if (!validAminoAcids.test(sequence)) {
      setError('Please enter a valid peptide sequence using only standard amino acids (ACDEFGHIKLMNPQRSTVWY).');
      setIsLoading(false);
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sequence: sequence.trim().toUpperCase()
        }),
      });

      const result = await response.json();
      
      if (!response.ok) {
        throw new Error(result.error || 'Prediction failed');
      }

      setPrediction(result);
    } catch (err) {
      console.error('Error making prediction:', err);
      setError(err.message || 'Failed to get prediction. Please make sure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderModel2Results = (model2Results) => {
    if (!model2Results || Object.keys(model2Results).length === 0) return null;

    // Filter to only show positive categories
    const positiveCategories = Object.entries(model2Results).filter(([_, value]) => value === true);

    if (positiveCategories.length === 0) {
      return <p className="text-gray-600 italic">No specific therapeutic categories identified.</p>;
    }

    return (
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        {positiveCategories.map(([category, _]) => (
          <div key={category} className="bg-blue-100 text-blue-800 px-3 py-2 rounded-full text-sm font-medium text-center">
            {category.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
          </div>
        ))}
      </div>
    );
  };

  const renderModel3Results = (model3Result) => {
    if (!model3Result) return null;

    // Parse the features and score from model3_result string
    const lines = model3Result.split('\n');
    const features = {};
    let biologicalScore = 0;

    lines.forEach(line => {
      // Extract features
      if (line.includes(':')) {
        const parts = line.split(':');
        if (parts.length >= 2) {
          const key = parts[0].trim();
          const value = parseFloat(parts[1].trim());
          if (!isNaN(value)) {
            features[key] = value;
          }
        }
      }
      
      // Extract biological score
      const scoreMatch = line.match(/Predicted Biological Score: ([\d.]+)/);
      if (scoreMatch && scoreMatch[1]) {
        biologicalScore = parseFloat(scoreMatch[1]);
      }
    });

    return (
      <div className="space-y-4">
        <h4 className="text-lg font-semibold">Extracted Biological Features</h4>

        {Object.entries(features).map(([feature, value]) => (
          <div key={feature} className="mb-2">
            <div className="flex justify-between mb-1">
              <span className="text-sm font-medium text-gray-700">{feature}</span>
              <span className="text-sm font-medium text-gray-700">{value.toFixed(4)}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div 
                className="bg-blue-600 h-2.5 rounded-full" 
                style={{ width: `${Math.min(Math.abs(value) * 5, 100)}%` }}
              ></div>
            </div>
          </div>
        ))}

        <div className="mt-6">
          <div className="flex justify-between mb-1">
            <span className="text-lg font-semibold text-gray-700">Biological Score</span>
            <span className="text-lg font-semibold text-gray-700">{biologicalScore.toFixed(4)}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-4">
            <div 
              className={`h-4 rounded-full ${biologicalScore > 10 ? 'bg-green-500' : biologicalScore > 5 ? 'bg-yellow-500' : 'bg-red-500'}`}
              style={{ width: `${Math.min((biologicalScore / 20) * 100, 100)}%` }}
            ></div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <section id="predictor" className="py-12 bg-gray-50">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-8">Peptide Predictor</h2>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="max-w-lg mx-auto bg-white p-6 rounded-lg shadow-md">
          <div className="mb-4">
            <label htmlFor="input-sequence" className="block text-gray-700 font-medium mb-2">
              Enter Peptide Sequence
            </label>
            <textarea
              id="input-sequence"
              value={sequence}
              onChange={(e) => setSequence(e.target.value.toUpperCase())}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows="4"
              placeholder="Enter your peptide sequence here (e.g., ACDEFGHIK)"
              required
            ></textarea>
            <p className="text-xs text-gray-500 mt-1">Use only standard amino acids (ACDEFGHIKLMNPQRSTVWY)</p>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300"
          >
            {isLoading ? 'Processing...' : 'Predict'}
          </button>
        </form>

        {isLoading && (
          <div className="mt-8 text-center">
            <div className="flex justify-center items-center">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-700"></div>
            </div>
            <p className="mt-2 text-gray-600">Processing your sequence. This may take a moment...</p>
          </div>
        )}

        {prediction && (
          <div className="mt-8 max-w-lg mx-auto bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-xl font-bold mb-6 text-center">Prediction Results</h3>

            {/* Model 1 Result */}
            <div className="mb-8">
              <h4 className="text-lg font-semibold mb-3">Therapeutic Potential</h4>
              {prediction.therapeutic === false ? (
                <div className="p-4 bg-red-100 rounded-lg border-l-4 border-red-500 text-red-700 font-bold text-center text-lg">
                  Non-Therapeutic
                </div>
              ) : (
                <div className="p-4 bg-green-100 rounded-lg border-l-4 border-green-500 text-green-700 font-bold text-center text-lg">
                  Therapeutic
                </div>
              )}
            </div>

            {/* Model 2 Categories - only show if therapeutic */}
            {prediction.therapeutic && prediction.model2_result && (
              <div className="mb-8">
                <h4 className="text-lg font-semibold mb-3">Therapeutic Categories</h4>
                {renderModel2Results(prediction.model2_result)}
              </div>
            )}

            {/* Model 3 Biological Features - only show if therapeutic */}
            {prediction.therapeutic && prediction.model3_result && (
              <div className="mb-4">
                <h4 className="text-lg font-semibold mb-3">Biological Properties</h4>
                {renderModel3Results(prediction.model3_result)}
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

export default Predictor;