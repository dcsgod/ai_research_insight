/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ['class'],
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // Dark background palette
        background: {
          DEFAULT: '#080c14',
          card: '#0d1421',
          elevated: '#111827',
          border: '#1e2d45',
        },
        // Accent — electric cyan/blue
        accent: {
          DEFAULT: '#00d4ff',
          dim: '#0099bb',
          glow: 'rgba(0,212,255,0.15)',
        },
        // Secondary accent — violet
        violet: {
          DEFAULT: '#7c3aed',
          light: '#a855f7',
          glow: 'rgba(124,58,237,0.2)',
        },
        // Semantic colors
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        muted: '#64748b',
        // Text
        foreground: {
          DEFAULT: '#e2e8f0',
          muted: '#94a3b8',
          dim: '#64748b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'hero-gradient': 'linear-gradient(135deg, #080c14 0%, #0d1a2e 50%, #080c14 100%)',
        'card-gradient': 'linear-gradient(135deg, rgba(13,20,33,0.9), rgba(17,24,39,0.6))',
        'accent-gradient': 'linear-gradient(135deg, #00d4ff, #7c3aed)',
        'glow-cyan': 'radial-gradient(circle at center, rgba(0,212,255,0.15) 0%, transparent 70%)',
      },
      boxShadow: {
        'card': '0 4px 24px rgba(0,0,0,0.4), 0 1px 0 rgba(255,255,255,0.05) inset',
        'card-hover': '0 8px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(0,212,255,0.2)',
        'accent-glow': '0 0 20px rgba(0,212,255,0.3)',
        'violet-glow': '0 0 20px rgba(124,58,237,0.3)',
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'shimmer': 'shimmer 1.5s infinite',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        slideUp: {
          from: { opacity: '0', transform: 'translateY(16px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-6px)' },
        },
      },
    },
  },
  plugins: [],
};
