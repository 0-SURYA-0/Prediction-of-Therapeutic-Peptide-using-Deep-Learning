import React, { useState } from 'react';
import { Mail, Github, Twitter, Instagram, Linkedin, ExternalLink } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const [activeSection, setActiveSection] = useState(null);
  const [hoverIcon, setHoverIcon] = useState(null);
  
  // Links data
  const socialLinks = [
    { name: 'Twitter', icon: <Twitter size={20} />, color: 'bg-blue-500' },
    { name: 'Github', icon: <Github size={20} />, color: 'bg-gray-800' },
    { name: 'Instagram', icon: <Instagram size={20} />, color: 'bg-gradient-to-r from-purple-500 to-pink-500' },
    { name: 'LinkedIn', icon: <Linkedin size={20} />, color: 'bg-blue-700' },
    { name: 'Email', icon: <Mail size={20} />, color: 'bg-red-500' },
  ];
  
  // Quick links
  const quickLinks = [
    { name: 'About', path: '#about' },
    { name: 'Features', path: '#features' },
    { name: 'Docs', path: '#docs' },
    { name: 'Pricing', path: '#pricing' },
  ];
  
  return (
    <footer className="relative overflow-hidden bg-gradient-to-r from-gray-900 to-gray-800 text-white py-12">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden opacity-10">
        <div className="absolute -top-24 -left-24 w-64 h-64 rounded-full bg-blue-500 blur-3xl"></div>
        <div className="absolute top-36 -right-24 w-80 h-80 rounded-full bg-purple-600 blur-3xl"></div>
        <div className="absolute bottom-12 left-36 w-72 h-72 rounded-full bg-indigo-500 blur-3xl"></div>
      </div>
      
      <div className="container mx-auto px-6 relative z-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-12">
          {/* Brand Column */}
          <div 
            className={`transform transition-all duration-300 ${activeSection === 'brand' ? 'scale-105' : ''}`}
            onMouseEnter={() => setActiveSection('brand')}
            onMouseLeave={() => setActiveSection(null)}
          >
            <div className="flex items-center space-x-2 mb-6">
              <span className="text-2xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">Pept-AI</span>
              <div className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></div>
            </div>
            <p className="text-gray-300 mb-4">
              Revolutionizing therapeutic peptide prediction with cutting-edge artificial intelligence.
            </p>
            <div className="w-24 h-1 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"></div>
          </div>
          
          {/* Quick Links */}
          <div 
            className={`transform transition-all duration-300 ${activeSection === 'links' ? 'scale-105' : ''}`}
            onMouseEnter={() => setActiveSection('links')}
            onMouseLeave={() => setActiveSection(null)}
          >
            <h3 className="text-lg font-semibold mb-6 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">Quick Links</h3>
            <ul className="space-y-3">
              {quickLinks.map((link, index) => (
                <li key={index} className="transform transition-all duration-300 hover:translate-x-2">
                  <a 
                    href={link.path} 
                    className="text-gray-300 hover:text-white flex items-center group"
                  >
                    <span className="w-0 group-hover:w-3 transition-all duration-300 h-px bg-indigo-400 mr-0 group-hover:mr-2"></span>
                    {link.name}
                    <ExternalLink size={14} className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  </a>
                </li>
              ))}
            </ul>
          </div>
          
          {/* Newsletter */}
          <div 
            className={`transform transition-all duration-300 ${activeSection === 'newsletter' ? 'scale-105' : ''}`}
            onMouseEnter={() => setActiveSection('newsletter')}
            onMouseLeave={() => setActiveSection(null)}
          >
            <h3 className="text-lg font-semibold mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">Stay Updated</h3>
            <p className="text-gray-300 mb-4">Subscribe to our newsletter for the latest updates.</p>
            <div className="relative">
              <input 
                type="email" 
                placeholder="Enter your email" 
                className="w-full bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg py-3 px-4 text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-300"
              />
              <button className="absolute right-0 top-0 bottom-0 px-4 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 rounded-r-lg text-white font-medium transition-all duration-300">
                Subscribe
              </button>
            </div>
          </div>
        </div>
        
        {/* Social Media */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {socialLinks.map((link, index) => (
            <button
              key={index}
              className={`relative group p-3 rounded-full transition-all duration-300 transform hover:scale-110 ${link.color} hover:shadow-lg hover:shadow-${link.color.split('-')[1]}-500/40`}
              onMouseEnter={() => setHoverIcon(index)}
              onMouseLeave={() => setHoverIcon(null)}
            >
              {link.icon}
              <span className={`absolute -top-10 left-1/2 transform -translate-x-1/2 bg-white text-gray-800 px-3 py-1 rounded-lg text-sm font-medium opacity-0 transition-opacity duration-300 ${hoverIcon === index ? 'opacity-100' : ''}`}>
                {link.name}
              </span>
            </button>
          ))}
        </div>
        
        {/* Bottom Border with Gradient Animation */}
        <div className="w-full h-px bg-gradient-to-r from-transparent via-indigo-500 to-transparent relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-purple-500 to-transparent animate-gradient-x"></div>
        </div>
        
        {/* Copyright */}
        <div className="mt-8 text-center text-gray-400">
          <p className="group">
            © {currentYear} Pept-AI. All rights reserved.
            <span className="inline-block ml-1 w-2 h-2 bg-indigo-400 rounded-full group-hover:animate-ping"></span>
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;