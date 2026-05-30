'use client';

import { TrendingItem } from '@/lib/types';
import {
  cn,
  formatScore,
  scoreColor,
  scoreBg,
  rankDeltaColor,
  rankDeltaLabel,
  entityTypeIcon,
  entityTypeLabel,
} from '@/lib/utils';
import { ExternalLink, TrendingUp, Zap } from 'lucide-react';

interface TrendCardProps {
  item: TrendingItem;
  rank: number;
  showInsight?: boolean;
  className?: string;
}

export function TrendCard({ item, rank, showInsight = true, className }: TrendCardProps) {
  const scoreNum = parseFloat(formatScore(item.trend_score));
  const velocityPct = Math.round(item.growth_velocity * 100);

  return (
    <div
      className={cn(
        'card card-hover p-4 group cursor-pointer animate-slide-up',
        className
      )}
      style={{ animationDelay: `${rank * 40}ms` }}
    >
      <div className="flex items-start gap-3">
        {/* Rank */}
        <div className="shrink-0 pt-0.5">
          <span className="rank-number">#{rank}</span>
        </div>

        {/* Entity icon */}
        <div className="shrink-0 w-8 h-8 rounded-lg bg-white/5 border border-white/8 flex items-center justify-center text-base">
          {entityTypeIcon(item.entity_type)}
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              {/* Type badge */}
              <span className={cn('badge text-[10px] mr-2 mb-1',
                item.entity_type === 'paper' ? 'badge-violet' :
                item.entity_type === 'repo' ? 'badge-cyan' : 'badge-green'
              )}>
                {entityTypeLabel(item.entity_type)}
              </span>

              {/* Title */}
              <h3 className="text-sm font-semibold text-slate-100 leading-snug group-hover:text-cyan-300 transition-colors line-clamp-2">
                {item.title}
              </h3>
            </div>

            {/* Score */}
            <div className={cn('shrink-0 flex items-center gap-1 px-2 py-1 rounded-lg border text-xs font-bold mono', scoreBg(item.trend_score))}>
              <Zap size={10} />
              <span className={scoreColor(item.trend_score)}>{scoreNum}</span>
            </div>
          </div>

          {/* Description */}
          {item.description && (
            <p className="text-xs text-slate-500 mt-1 line-clamp-2">{item.description}</p>
          )}

          {/* Tags */}
          {item.tags && item.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {item.tags.slice(0, 4).map((tag) => (
                <span key={tag} className="badge-gray text-[10px]">{tag}</span>
              ))}
            </div>
          )}

          {/* AI Insight */}
          {showInsight && item.insight && (
            <div className="mt-2 px-2.5 py-2 rounded-lg bg-violet-500/8 border border-violet-400/15">
              <p className="text-xs text-violet-300/80 italic leading-relaxed">
                &ldquo;{item.insight}&rdquo;
              </p>
            </div>
          )}

          {/* Metrics row */}
          <div className="flex items-center gap-4 mt-3">
            {/* Velocity */}
            <div className="flex items-center gap-1.5">
              <TrendingUp size={12} className="text-cyan-400" />
              <span className="text-xs text-slate-400">Velocity</span>
              <span className="text-xs font-semibold text-cyan-300">{velocityPct}%</span>
            </div>

            {/* Rank delta */}
            <div className={cn('flex items-center gap-1 text-xs font-semibold', rankDeltaColor(item.delta_rank))}>
              {rankDeltaLabel(item.delta_rank)}
            </div>

            {/* Score bar */}
            <div className="flex-1">
              <div className="score-bar">
                <div
                  className="score-bar-fill bg-gradient-to-r from-cyan-400 to-violet-500"
                  style={{ width: `${scoreNum}%` }}
                />
              </div>
            </div>

            {/* Link */}
            {item.url && item.url !== '#' && (
              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="text-slate-600 hover:text-cyan-400 transition-colors"
              >
                <ExternalLink size={12} />
              </a>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
