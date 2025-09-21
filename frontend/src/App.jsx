import React from 'react';
import Navbar from './components/Navbar';
import Home from './components/Home';
import Predictor from './components/Predictor';
import Footer from './components/Footer';

function App() {
  return (
    <div className="font-sans antialiased">
      <Navbar />
      <main>
        <Home />
        <Predictor />
      </main>
      <Footer />
    </div>
  );
}

export default App;