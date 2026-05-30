import type { Metadata } from 'next';
import './globals.css';
import { Sidebar } from '@/components/Sidebar';

export const metadata: Metadata = {
  title: 'AI Research Intelligence Platform',
  description: 'Bloomberg Terminal for AI Research Trends — Track papers, repos, and emerging AI topics in real-time with forecasting and ranking.',
  keywords: ['AI', 'research', 'trends', 'papers', 'GitHub', 'machine learning', 'forecasting'],
  authors: [{ name: 'AI Research Platform' }],
  openGraph: {
    title: 'AI Research Intelligence Platform',
    description: 'Bloomberg Terminal for AI Research Trends',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[#080c14] text-slate-100 antialiased">
        <div className="flex h-screen overflow-hidden">
          {/* Sidebar navigation */}
          <Sidebar />

          {/* Main content area */}
          <main className="flex-1 overflow-y-auto">
            {/* Subtle grid background */}
            <div className="fixed inset-0 grid-bg pointer-events-none opacity-40" />
            {/* Top ambient glow */}
            <div className="fixed top-0 left-64 right-0 h-px bg-gradient-to-r from-transparent via-cyan-400/20 to-transparent pointer-events-none" />
            <div className="relative z-10">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
