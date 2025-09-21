import React, { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  // Handle scroll event to add shadow to navbar when scrolled
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Smooth scroll to section
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setIsOpen(false);
    }
  };

  return (
    <nav className={`fixed w-full z-50 transition-all duration-500 bg-white ${
      isScrolled 
        ? 'shadow-lg py-3' 
        : 'py-5'
    }`}>
      <div className="container mx-auto px-4 md:px-8">
        <div className="flex justify-between items-center">
          <div className="flex items-center">
            <span className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">Pept-AI</span>
          </div>
          
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-10">
            <NavItem onClick={() => scrollToSection('home')} label="Home" />
            <NavItem onClick={() => scrollToSection('predictor')} label="Predictor" />
            <button className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium py-2 px-6 rounded-full transition-all duration-300 transform hover:scale-105 shadow-md">
              Get Started
            </button>
          </div>
          
          {/* Mobile Navigation Toggle */}
          <div className="md:hidden">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-700 hover:text-indigo-600 focus:outline-none transition-colors duration-300"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>
      
      {/* Mobile Navigation Menu */}
      {isOpen && (
        <div className="md:hidden bg-white shadow-lg absolute top-full left-0 right-0 py-4 px-6 transition-all duration-300 border-t border-gray-100">
          <div className="flex flex-col space-y-4">
            <NavItem onClick={() => scrollToSection('home')} label="Home" mobile />
            <NavItem onClick={() => scrollToSection('predictor')} label="Predictor" mobile />
          </div>
        </div>
      )}
    </nav>
  );
};

// Reusable NavItem component
const NavItem = ({ onClick, label, mobile = false }) => (
  <button 
    onClick={onClick}
    className={`relative group ${
      mobile 
        ? "text-gray-700 py-3 w-full text-left border-b border-gray-100 hover:text-indigo-600" 
        : "text-gray-700 hover:text-indigo-600"
    } transition-colors duration-300 font-medium`}
  >
    {label}
    {!mobile && (
      <span className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-indigo-500 to-purple-600 group-hover:w-full transition-all duration-300"></span>
    )}
  </button>
);

export default Navbar;