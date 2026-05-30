'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  LayoutDashboard,
  TrendingUp,
  FileText,
  Github,
  BarChart3,
  Network,
  Zap,
  Settings,
  RefreshCw,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useState } from 'react';
import { api } from '@/lib/api';

const NAV_ITEMS = [
  { href: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { href: '/trends', icon: TrendingUp, label: 'Trends' },
  { href: '/papers', icon: FileText, label: 'Papers' },
  { href: '/repos', icon: Github, label: 'Repositories' },
  { href: '/forecasts', icon: BarChart3, label: 'Forecasts' },
  { href: '/topics', icon: Network, label: 'Topic Explorer' },
];

export function Sidebar() {
  const pathname = usePathname();
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);

  const handleIngest = async () => {
    setIngesting(true);
    setIngestStatus(null);
    try {
      const result = await api.triggerIngestion();
      setIngestStatus(`✓ Job queued: ${result.job_id}`);
    } catch {
      setIngestStatus('Queued (offline mode)');
    } finally {
      setIngesting(false);
      setTimeout(() => setIngestStatus(null), 3000);
    }
  };

  return (
    <aside className="w-64 shrink-0 h-screen flex flex-col border-r border-[#1e2d45] bg-[#080c14]">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-[#1e2d45]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-violet-600 flex items-center justify-center glow-cyan">
            <Zap size={16} className="text-white" />
          </div>
          <div>
            <div className="text-sm font-bold text-white leading-tight">AI Research</div>
            <div className="text-xs text-slate-500 leading-tight">Intelligence Platform</div>
          </div>
        </div>
        {/* Live indicator */}
        <div className="mt-3 flex items-center gap-2">
          <div className="live-dot" />
          <span className="text-xs text-slate-500">Live ingestion active</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        <div className="text-xs font-semibold text-slate-600 uppercase tracking-wider px-3 mb-2">
          Intelligence
        </div>
        {NAV_ITEMS.map(({ href, icon: Icon, label }) => {
          const active = pathname === href || (href !== '/' && pathname.startsWith(href));
          return (
            <Link key={href} href={href}>
              <div
                className={cn(
                  'nav-link',
                  active && 'active'
                )}
              >
                <Icon size={16} className={active ? 'text-cyan-400' : 'text-slate-500'} />
                <span>{label}</span>
                {active && (
                  <div className="ml-auto w-1 h-4 rounded-full bg-cyan-400/60" />
                )}
              </div>
            </Link>
          );
        })}
      </nav>

      {/* Bottom section */}
      <div className="px-3 pb-4 space-y-2 border-t border-[#1e2d45] pt-3">
        {/* Ingest trigger */}
        <button
          onClick={handleIngest}
          disabled={ingesting}
          className={cn(
            'w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm font-medium',
            'bg-cyan-400/10 border border-cyan-400/20 text-cyan-300',
            'hover:bg-cyan-400/15 hover:border-cyan-400/30',
            'transition-all duration-150',
            ingesting && 'opacity-60 cursor-not-allowed'
          )}
        >
          <RefreshCw size={15} className={ingesting ? 'animate-spin' : ''} />
          <span>{ingesting ? 'Ingesting...' : 'Run Ingestion'}</span>
        </button>

        {ingestStatus && (
          <div className="px-3 py-1.5 rounded-lg bg-emerald-400/10 border border-emerald-400/20">
            <p className="text-xs text-emerald-400">{ingestStatus}</p>
          </div>
        )}

        {/* Data sources */}
        <div className="px-3 py-2">
          <div className="text-xs text-slate-600 mb-2 font-medium">Data Sources</div>
          <div className="space-y-1.5">
            {['arXiv', 'GitHub', 'HuggingFace', 'PapersWithCode', 'Reddit'].map((src) => (
              <div key={src} className="flex items-center justify-between">
                <span className="text-xs text-slate-500">{src}</span>
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              </div>
            ))}
          </div>
        </div>

        {/* Version */}
        <div className="px-3 text-xs text-slate-700 font-mono">v1.0.0</div>
      </div>
    </aside>
  );
}
