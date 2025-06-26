import React, { useEffect, useRef, useState } from 'react';
import { ChevronRight } from 'lucide-react';

const Home = () => {
  const titleRef = useRef(null);
  const subtitleRef = useRef(null);
  const videoRef = useRef(null);
  const [currentInfo, setCurrentInfo] = useState(0);

  // Project information content that changes every 6 seconds
  const projectInfo = [
    {
      title: "AI-Driven Peptide Discovery",
      content: "Utilizing advanced machine learning to predict therapeutic peptides with high accuracy and efficiency."
    },
    {
      title: "Revolutionizing Drug Development",
      content: "Significantly reducing the time and cost of conventional peptide-based drug discovery through AI-powered predictions."
    },
    {
      title: "Cutting-Edge Technology",
      content: "Combining bioinformatics, deep learning, and molecular modeling to identify novel peptide candidates."
    },
    {
      title: "Data-Driven Insights",
      content: "Analyzing vast datasets of peptide structures and properties to uncover patterns invisible to traditional methods."
    }
  ];

  // Team members data
  const teamMembers = [
    { name: "Surya HA", rollNumber: "CB.SC.U4AIE23267"},
    { name: "Vishal Seshadri B", rollNumber: "CB.SC.U4AIE23260"},
    { name: "Venkatram KS", rollNumber: "CB.SC.U4AIE23236"},
    { name: "Sanggit Saaran KCS", rollNumber: "CB.SC.U4AIE23247"}
  ];

  useEffect(() => {
    // Fade-in animation for title and subtitle
    const title = titleRef.current;
    const subtitle = subtitleRef.current;
    
    if (title && subtitle) {
      title.style.opacity = 0;
      subtitle.style.opacity = 0;
      
      setTimeout(() => {
        title.style.transition = 'opacity 1s ease-in-out, transform 1s ease-out';
        title.style.transform = 'translateY(0)';
        title.style.opacity = 1;
      }, 300);
      
      setTimeout(() => {
        subtitle.style.transition = 'opacity 1s ease-in-out, transform 1s ease-out';
        subtitle.style.transform = 'translateY(0)';
        subtitle.style.opacity = 1;
      }, 800);
    }

    // Setting up video properties
    if (videoRef.current) {
      videoRef.current.playbackRate = 0.8;
    }

    // Rotating project info every 6 seconds
    const infoInterval = setInterval(() => {
      setCurrentInfo((prev) => (prev + 1) % projectInfo.length);
    }, 6000);

    return () => clearInterval(infoInterval);
  }, []);

  // Smooth scroll function
  const scrollToPredictor = () => {
    const predictorSection = document.getElementById('predictor');
    if (predictorSection) {
      // Add offset to account for fixed navbar
      const navbarHeight = 80; // Adjust this value to match your navbar height
      const predictorPosition = predictorSection.getBoundingClientRect().top + window.pageYOffset - navbarHeight;
      
      window.scrollTo({
        top: predictorPosition,
        behavior: 'smooth'
      });
    }
  };

  return (
    <section id="home" className="min-h-screen flex items-center justify-center relative overflow-hidden pt-24">
      {/* Background video */}
      <div className="absolute inset-0 z-0">
        <video 
          ref={videoRef}
          autoPlay 
          loop 
          muted 
          className="w-full h-full object-cover"
        >
          <source src="/bg video.mp4" type="video/mp4" />
          Your browser does not support the video tag.
        </video>
        {/* Overlay to ensure text readability */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/50 via-black/40 to-black/60"></div>
      </div>

      <div className="container mx-auto px-4 md:px-6 relative z-10 py-16 mt-10">
        <div className="text-center mb-16">
          <h1 
            ref={titleRef}
            className="text-4xl md:text-6xl font-bold text-white mb-6 opacity-0 transform translate-y-8"
          >
            Welcome to <span className="bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">Pept-AI</span>
          </h1>
          <p 
            ref={subtitleRef}
            className="text-xl md:text-2xl text-gray-200 max-w-2xl mx-auto opacity-0 transform translate-y-8"
          >
            Explore therapeutic peptide predictions with AI.
          </p>
          <div className="mt-12 flex justify-center">
            <button
              onClick={scrollToPredictor}
              className="bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-medium py-3 px-8 rounded-full transition-all duration-300 shadow-lg hover:shadow-indigo-500/30 flex items-center group"
            >
              Get Started
              <ChevronRight size={20} className="ml-2 group-hover:ml-3 transition-all" />
            </button>
          </div>
        </div>

        {/* Project Info Box with Smooth Transition - 75% white background */}
        <div className="max-w-4xl mx-auto mb-16 bg-white/75 backdrop-blur-md p-8 rounded-xl border border-white/20 shadow-lg transform hover:scale-105 transition-all duration-500">
          <div className="relative h-32"> {/* Fixed height container for smooth transitions */}
            {projectInfo.map((info, index) => (
              <div 
                key={index} 
                className={`absolute w-full transition-all duration-1000 ease-in-out ${
                  currentInfo === index 
                    ? 'opacity-100 translate-y-0' 
                    : index < currentInfo || (currentInfo === 0 && index === projectInfo.length - 1)
                      ? 'opacity-0 -translate-y-8' 
                      : 'opacity-0 translate-y-8'
                }`}
              >
                <h2 className="text-2xl md:text-3xl font-bold mb-4 bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
                  {info.title}
                </h2>
                <p className="text-lg text-gray-800">
                  {info.content}
                </p>
              </div>
            ))}
          </div>
          <div className="flex justify-center mt-4">
            {projectInfo.map((_, i) => (
              <div 
                key={i} 
                className={`w-2 h-2 rounded-full mx-1 ${
                  i === currentInfo ? 'bg-indigo-500' : 'bg-gray-400'
                }`}
              ></div>
            ))}
          </div>
        </div>

        {/* Team Members Section */}
        <div className="mt-16">
          <h2 className="text-3xl font-bold text-center text-white mb-12 bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
            Meet Our Development Team
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {teamMembers.map((member, index) => (
              <div 
                key={index} 
                className="bg-white/10 backdrop-blur-md p-6 rounded-xl border border-white/20 shadow-lg hover:shadow-indigo-500/20 transition-all duration-300 hover:-translate-y-2"
              >
                <div className="w-20 h-20 mx-auto bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full flex items-center justify-center mb-4">
                  <span className="text-white text-2xl font-bold">{member.name.charAt(0)}</span>
                </div>
                <h3 className="text-xl font-semibold text-white text-center mb-2">{member.name}</h3>
                <p className="text-gray-300 text-center mb-1">{member.role}</p>
                <p className="text-indigo-300 text-sm text-center">{member.rollNumber}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Home;