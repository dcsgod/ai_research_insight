'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_DASHBOARD, MOCK_TRENDING } from '@/lib/api';
import { DashboardSummary, TrendingItem } from '@/lib/types';
import { TrendCard } from '@/components/TrendCard';
import { LiveFeed } from '@/components/LiveFeed';
import { ForecastChart } from '@/components/ForecastChart';
import {
  FileText, Github, Network, TrendingUp,
  Zap, AlertTriangle, BarChart3, Clock
} from 'lucide-react';
import { cn, formatNumber, formatRelativeTime } from '@/lib/utils';

// ── Stat Card ─────────────────────────────────────────────────────────────────
function StatCard({
  icon: Icon,
  label,
  value,
  subtext,
  accentColor = 'text-cyan-400',
  glowColor = 'bg-cyan-400/10',
}: {
  icon: any;
  label: string;
  value: string | number;
  subtext?: string;
  accentColor?: string;
  glowColor?: string;
}) {
  return (
    <div className="card card-hover p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">{label}</p>
          <p className={cn('text-2xl font-bold mt-1 mono', accentColor)}>{value}</p>
          {subtext && <p className="text-xs text-slate-500 mt-0.5">{subtext}</p>}
        </div>
        <div className={cn('w-9 h-9 rounded-lg flex items-center justify-center', glowColor)}>
          <Icon size={18} className={accentColor} />
        </div>
      </div>
    </div>
  );
}

// ── Score Breakdown Bar ───────────────────────────────────────────────────────
function ScoreBreakdown({ items }: { items: { label: string; score: number; color: string }[] }) {
  return (
    <div className="space-y-2">
      {items.map(({ label, score, color }) => (
        <div key={label} className="flex items-center gap-3">
          <span className="text-xs text-slate-500 w-32 shrink-0">{label}</span>
          <div className="flex-1 score-bar">
            <div
              className="score-bar-fill transition-all duration-700"
              style={{ width: `${score * 100}%`, background: color }}
            />
          </div>
          <span className="text-xs font-semibold mono text-slate-400 w-8 text-right">
            {(score * 100).toFixed(0)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [summary, setSummary] = useState<DashboardSummary>(MOCK_DASHBOARD);
  const [trending, setTrending] = useState<TrendingItem[]>(MOCK_TRENDING);
  const [forecast, setForecast] = useState<any>(null);
  const [dailyInsight, setDailyInsight] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      const [sum, trends, fc, insight] = await Promise.all([
        api.getDashboard(),
        api.getTrending('7d', 8),
        api.getForecast('global', 'topic'),
        api.getDailyInsight(),
      ]);
      setSummary(sum);
      setTrending(trends);
      setForecast(fc);
      setDailyInsight(insight);
      setLoading(false);
    };
    load();
  }, []);

  return (
    <div className="p-6 max-w-[1400px] mx-auto">
      {/* ── Page Header ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white">
            AI Research <span className="gradient-text">Intelligence</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Bloomberg Terminal for AI Research Trends
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/3 border border-[#1e2d45]">
          <Clock size={13} className="text-slate-500" />
          <span className="text-xs text-slate-500 mono">
            Updated {formatRelativeTime(summary.last_ingestion)}
          </span>
        </div>
      </div>

      {/* ── Daily AI Insight ─────────────────────────────────────────────── */}
      <div className="card p-4 mb-6 border-l-2 border-l-violet-500">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-violet-500/15 flex items-center justify-center shrink-0">
            <Zap size={16} className="text-violet-400" />
          </div>
          <div>
            <p className="text-xs font-semibold text-violet-400 mb-1 uppercase tracking-wider">
              Daily AI Market Briefing
            </p>
            <p className="text-sm text-slate-300 leading-relaxed italic">
              {dailyInsight || 'Loading daily briefing...'}
            </p>
          </div>
        </div>
      </div>

      {/* ── Stats Grid ──────────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <StatCard
          icon={FileText}
          label="Total Papers"
          value={formatNumber(summary.total_papers)}
          subtext={`+${summary.papers_today} today`}
          accentColor="text-cyan-400"
          glowColor="bg-cyan-400/10"
        />
        <StatCard
          icon={Github}
          label="Tracked Repos"
          value={formatNumber(summary.total_repos)}
          subtext={`${summary.repos_trending} trending`}
          accentColor="text-violet-400"
          glowColor="bg-violet-500/10"
        />
        <StatCard
          icon={Network}
          label="Topics Found"
          value={summary.total_topics}
          subtext={`${summary.breakout_topics} breakouts`}
          accentColor="text-emerald-400"
          glowColor="bg-emerald-400/10"
        />
        <StatCard
          icon={AlertTriangle}
          label="Breakout Topics"
          value={summary.breakout_topics}
          subtext="Anomaly detected"
          accentColor="text-amber-400"
          glowColor="bg-amber-400/10"
        />
      </div>

      {/* ── Main Layout ─────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">

        {/* Left: Trending + Ranking ─────────────────────────────────────── */}
        <div className="xl:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-semibold text-white flex items-center gap-2">
              <TrendingUp size={16} className="text-cyan-400" />
              Top Ranked This Week
            </h2>
            <a href="/trends" className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors">
              View all →
            </a>
          </div>

          <div className="space-y-2">
            {trending.slice(0, 6).map((item, i) => (
              <TrendCard key={item.entity_id} item={item} rank={i + 1} showInsight={i < 2} />
            ))}
          </div>
        </div>

        {/* Right: Live feed + Forecast + Ranking formula ───────────────── */}
        <div className="space-y-4">

          {/* Live Feed */}
          <div className="card p-4">
            <LiveFeed />
          </div>

          {/* Mini Forecast */}
          {forecast && (
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 size={14} className="text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">Ecosystem Forecast</h3>
              </div>
              <ForecastChart data={forecast} height={180} showConfidence />
            </div>
          )}

          {/* Ranking Formula */}
          <div className="card p-4">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Ranking Formula
            </h3>
            <ScoreBreakdown
              items={[
                { label: 'Growth Velocity', score: 0.35, color: '#00d4ff' },
                { label: 'GitHub Activity', score: 0.25, color: '#7c3aed' },
                { label: 'Citation Accel.', score: 0.20, color: '#10b981' },
                { label: 'Engagement', score: 0.10, color: '#f59e0b' },
                { label: 'Novelty Score', score: 0.10, color: '#6366f1' },
              ]}
            />
            <div className="mt-3 px-2.5 py-2 rounded-lg bg-white/3 border border-[#1e2d45]">
              <p className="text-[10px] text-slate-600 mono leading-relaxed">
                score = 0.35×velocity + 0.25×github<br />
                + 0.20×citation + 0.10×engage<br />
                + 0.10×novelty
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
