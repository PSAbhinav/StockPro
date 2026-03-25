'use client';

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

const BackgroundAnimation = () => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  // Generate random data points for multiple chart lines
  const generatePath = (points: number, height: number, width: number, volatility: number) => {
    let d = `M 0 ${height / 2}`;
    let y = height / 2;
    for (let i = 1; i <= points; i++) {
      const x = (i / points) * width;
      // Random walk
      y += (Math.random() - 0.5) * volatility;
      // Keep within bounds
      y = Math.max(0, Math.min(height, y));
      d += ` L ${x} ${y}`;
    }
    return d;
  };

  const lines = [
    { delay: 0, color: 'rgba(59, 130, 246, 0.15)', duration: 40, yOffset: 100 },
    { delay: 2, color: 'rgba(16, 185, 129, 0.1)', duration: 35, yOffset: 300 },
    { delay: 5, color: 'rgba(99, 102, 241, 0.12)', duration: 50, yOffset: 500 },
    { delay: 1, color: 'rgba(59, 130, 246, 0.08)', duration: 45, yOffset: 700 }
  ];

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none select-none bg-[#020203]">
      {/* Background base gradient - Deep Space */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,_#0a0f20_0%,_#020203_100%)] opacity-80" />
      
      {/* Soft radial glows to create depth */}
      <div className="absolute top-0 left-[20%] w-[40%] h-[40%] bg-blue-600/5 blur-[120px] rounded-full" />
      <div className="absolute bottom-[20%] right-[-10%] w-[50%] h-[50%] bg-emerald-500/5 blur-[150px] rounded-full" />

      {/* Animated Stock Chart Lines */}
      <svg className="absolute w-[200vw] h-full left-0 opacity-60" preserveAspectRatio="none">
        {lines.map((line, i) => (
          <motion.path
            key={i}
            d={generatePath(50, 1000, 4000, 150)}
            fill="none"
            stroke={line.color}
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ x: 0, y: line.yOffset - 500 }}
            animate={{ x: -2000 }}
            transition={{
              repeat: Infinity,
              ease: "linear",
              duration: line.duration,
              delay: line.delay
            }}
          />
        ))}
        {/* Fill gradients below the lines for an area chart effect */}
        {lines.map((line, i) => (
          <motion.path
            key={`area-${i}`}
            d={`${generatePath(50, 1000, 4000, 150)} L 4000 1000 L 0 1000 Z`}
            fill={`url(#gradient-${i})`}
            initial={{ x: 0, y: line.yOffset - 500 }}
            animate={{ x: -2000 }}
            transition={{
              repeat: Infinity,
              ease: "linear",
              duration: line.duration,
              delay: line.delay
            }}
          />
        ))}
        <defs>
          {lines.map((line, i) => (
            <linearGradient key={`gradient-${i}`} id={`gradient-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={line.color.replace('0.15', '0.05').replace('0.1', '0.03').replace('0.12', '0.04')} />
              <stop offset="100%" stopColor="transparent" />
            </linearGradient>
          ))}
        </defs>
      </svg>

      {/* Grid horizontal lines indicating price levels */}
      <div className="absolute inset-0 flex flex-col justify-evenly opacity-[0.03]">
        {[...Array(8)].map((_, i) => (
          <div key={`h-grid-${i}`} className="w-full h-px bg-white border-dashed" />
        ))}
      </div>
      
      {/* Grid vertical lines indicating timeframes */}
      <div className="absolute inset-0 flex justify-evenly opacity-[0.02]">
        {[...Array(12)].map((_, i) => (
          <div key={`v-grid-${i}`} className="h-full w-px bg-white border-dashed" />
        ))}
      </div>

    </div>
  );
};

export default BackgroundAnimation;
