import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function formatScore(score: number): string {
  return (score * 100).toFixed(0);
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function formatRelativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function scoreColor(score: number): string {
  if (score >= 0.85) return 'text-emerald-400';
  if (score >= 0.70) return 'text-cyan-400';
  if (score >= 0.50) return 'text-amber-400';
  return 'text-slate-400';
}

export function scoreBg(score: number): string {
  if (score >= 0.85) return 'bg-emerald-400/10 border-emerald-400/30';
  if (score >= 0.70) return 'bg-cyan-400/10 border-cyan-400/30';
  if (score >= 0.50) return 'bg-amber-400/10 border-amber-400/30';
  return 'bg-slate-400/10 border-slate-400/30';
}

export function rankDeltaColor(delta: number): string {
  if (delta > 0) return 'text-emerald-400';
  if (delta < 0) return 'text-red-400';
  return 'text-slate-400';
}

export function rankDeltaLabel(delta: number): string {
  if (delta > 0) return `▲ ${delta}`;
  if (delta < 0) return `▼ ${Math.abs(delta)}`;
  return '–';
}

export function entityTypeIcon(type: string): string {
  switch (type) {
    case 'paper': return '📄';
    case 'repo': return '⭐';
    case 'topic': return '🔮';
    default: return '•';
  }
}

export function entityTypeLabel(type: string): string {
  switch (type) {
    case 'paper': return 'Paper';
    case 'repo': return 'Repository';
    case 'topic': return 'Topic';
    default: return type;
  }
}

export function generateChartColors(n: number): string[] {
  const palette = [
    '#00d4ff', '#7c3aed', '#10b981', '#f59e0b',
    '#ef4444', '#6366f1', '#ec4899', '#14b8a6',
  ];
  return Array.from({ length: n }, (_, i) => palette[i % palette.length]);
}
