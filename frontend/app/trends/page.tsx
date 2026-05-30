'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_TRENDING } from '@/lib/api';
import { TrendingItem, Timeframe } from '@/lib/types';
import { TrendCard } from '@/components/TrendCard';
import { TrendingUp, Filter, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

const TIMEFRAMES: { value: Timeframe; label: string }[] = [
  { value: '24h', label: '24H' },
  { value: '7d', label: '7D' },
  { value: '30d', label: '30D' },
];

const ENTITY_TYPES = [
  { value: '', label: 'All' },
  { value: 'paper', label: 'Papers' },
  { value: 'repo', label: 'Repos' },
  { value: 'topic', label: 'Topics' },
];

export default function TrendsPage() {
  const [items, setItems] = useState<TrendingItem[]>(MOCK_TRENDING);
  const [timeframe, setTimeframe] = useState<Timeframe>('7d');
  const [entityType, setEntityType] = useState('');
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    const data = await api.getTrending(timeframe, 30, entityType || undefined);
    setItems(data);
    setLoading(false);
  };

  useEffect(() => { load(); }, [timeframe, entityType]);

  const filtered = entityType ? items.filter(i => i.entity_type === entityType) : items;

  return (
    <div className="p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <TrendingUp className="text-cyan-400" size={22} />
            AI <span className="gradient-text">Trends</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            X-algorithm ranked AI research signals
          </p>
        </div>
        <button onClick={load} disabled={loading} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 border border-[#1e2d45] text-sm text-slate-300 hover:bg-white/8 transition-all">
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-5 flex-wrap">
        <div className="flex items-center gap-1 bg-white/3 border border-[#1e2d45] rounded-lg p-0.5">
          {TIMEFRAMES.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setTimeframe(value)}
              className={cn(
                'px-3 py-1.5 rounded-md text-xs font-semibold transition-all',
                timeframe === value
                  ? 'bg-cyan-400/15 text-cyan-300 border border-cyan-400/25'
                  : 'text-slate-500 hover:text-slate-300'
              )}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 bg-white/3 border border-[#1e2d45] rounded-lg p-0.5">
          {ENTITY_TYPES.map(({ value, label }) => (
            <button
              key={value}
              onClick={() => setEntityType(value)}
              className={cn(
                'px-3 py-1.5 rounded-md text-xs font-semibold transition-all',
                entityType === value
                  ? 'bg-violet-500/15 text-violet-300 border border-violet-400/25'
                  : 'text-slate-500 hover:text-slate-300'
              )}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="ml-auto text-xs text-slate-600">
          {filtered.length} results
        </div>
      </div>

      {/* Trend List */}
      <div className="space-y-2">
        {loading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="card p-4 h-24 shimmer" />
          ))
        ) : (
          filtered.map((item, i) => (
            <TrendCard key={item.entity_id} item={item} rank={i + 1} showInsight={i < 4} />
          ))
        )}
      </div>
    </div>
  );
}
