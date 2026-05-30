'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_TOPICS } from '@/lib/api';
import { Topic, TopicGraphData } from '@/lib/types';
import { TopicGraph } from '@/components/TopicGraph';
import { Network, TrendingUp, FileText, Github, Zap } from 'lucide-react';
import { cn, scoreColor, scoreBg } from '@/lib/utils';

function TopicCard({ topic, rank, onClick, selected }: {
  topic: Topic;
  rank: number;
  onClick: () => void;
  selected: boolean;
}) {
  const velocityPct = Math.round(topic.velocity * 100);
  const accelSign = topic.acceleration >= 0 ? '+' : '';

  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full text-left p-3.5 rounded-xl border transition-all',
        selected
          ? 'border-cyan-400/30 bg-cyan-400/8'
          : 'card card-hover'
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-600 mono">#{rank}</span>
            <h3 className="text-sm font-semibold text-slate-200 line-clamp-1">{topic.name}</h3>
          </div>

          {/* Keywords */}
          <div className="flex flex-wrap gap-1 mt-1.5">
            {topic.keywords.slice(0, 3).map((kw) => (
              <span key={kw} className="badge-gray text-[10px]">{kw}</span>
            ))}
          </div>

          {/* Metrics */}
          <div className="flex items-center gap-3 mt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <FileText size={10} />{topic.paper_count}
            </span>
            <span className="flex items-center gap-1">
              <Github size={10} />{topic.repo_count}
            </span>
            <span className={cn('font-semibold', topic.acceleration >= 0 ? 'text-emerald-400' : 'text-red-400')}>
              {accelSign}{(topic.acceleration * 100).toFixed(1)}% accel
            </span>
          </div>
        </div>

        <div className={cn('shrink-0 px-2 py-1 rounded-lg border text-xs font-bold mono', scoreBg(topic.trend_score))}>
          <span className={scoreColor(topic.trend_score)}>
            {(topic.trend_score * 100).toFixed(0)}
          </span>
        </div>
      </div>

      {/* Momentum bar */}
      <div className="mt-2 score-bar">
        <div
          className="score-bar-fill bg-gradient-to-r from-violet-500 to-cyan-400"
          style={{ width: `${topic.momentum * 100}%` }}
        />
      </div>
    </button>
  );
}

export default function TopicsPage() {
  const [topics, setTopics] = useState<Topic[]>(MOCK_TOPICS);
  const [graphData, setGraphData] = useState<TopicGraphData | null>(null);
  const [selected, setSelected] = useState<Topic | null>(null);
  const [view, setView] = useState<'list' | 'graph'>('list');

  useEffect(() => {
    api.getTopics(30).then(setTopics);
    api.getTopicGraph().then(setGraphData);
  }, []);

  useEffect(() => {
    if (topics.length > 0 && !selected) setSelected(topics[0]);
  }, [topics]);

  return (
    <div className="p-6 max-w-[1300px] mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Network className="text-emerald-400" size={22} />
            Topic <span className="gradient-text">Explorer</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            BERTopic clusters · Momentum · Relationships
          </p>
        </div>

        {/* View toggle */}
        <div className="flex items-center gap-1 bg-white/3 border border-[#1e2d45] rounded-lg p-0.5">
          {(['list', 'graph'] as const).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={cn(
                'px-4 py-1.5 rounded-md text-xs font-semibold capitalize transition-all',
                view === v
                  ? 'bg-emerald-400/15 text-emerald-300 border border-emerald-400/25'
                  : 'text-slate-500 hover:text-slate-300'
              )}
            >
              {v === 'list' ? '☰ List' : '⬡ Graph'}
            </button>
          ))}
        </div>
      </div>

      {view === 'graph' ? (
        /* ── Graph View ── */
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <h2 className="text-sm font-semibold text-white">Topic Relationship Graph</h2>
            <span className="badge-green text-[10px]">{topics.length} topics</span>
          </div>
          {graphData ? (
            <TopicGraph
              data={graphData}
              height={520}
              onNodeClick={(id) => {
                const t = topics.find((t) => Number(t.id) === id);
                if (t) setSelected(t);
              }}
            />
          ) : (
            <div className="h-96 shimmer rounded-xl" />
          )}
        </div>
      ) : (
        /* ── List View ── */
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* Topic list */}
          <div className="xl:col-span-1 space-y-2 max-h-[700px] overflow-y-auto pr-1">
            {topics.map((topic, i) => (
              <TopicCard
                key={topic.id}
                topic={topic}
                rank={i + 1}
                selected={selected?.id === topic.id}
                onClick={() => setSelected(topic)}
              />
            ))}
          </div>

          {/* Topic detail */}
          <div className="xl:col-span-2">
            {selected ? (
              <div className="card p-6 animate-fade-in">
                <div className="flex items-start justify-between mb-5">
                  <div>
                    <h2 className="text-xl font-bold text-white">{selected.name}</h2>
                    <div className="flex items-center gap-3 mt-1">
                      <span className={cn('text-sm font-semibold mono', scoreColor(selected.trend_score))}>
                        Score: {(selected.trend_score * 100).toFixed(0)}
                      </span>
                      <span className="text-sm text-slate-500">
                        {selected.paper_count} papers · {selected.repo_count} repos
                      </span>
                    </div>
                  </div>
                  {selected.acceleration > 0.1 && (
                    <span className="badge-green animate-pulse">🚀 Breakout</span>
                  )}
                </div>

                {/* Metrics grid */}
                <div className="grid grid-cols-3 gap-4 mb-5">
                  {[
                    { label: 'Momentum', value: `${(selected.momentum * 100).toFixed(0)}%`, color: 'text-violet-400' },
                    { label: 'Velocity', value: `${(selected.velocity * 100).toFixed(0)}%`, color: 'text-cyan-400' },
                    { label: 'Acceleration', value: `${selected.acceleration >= 0 ? '+' : ''}${(selected.acceleration * 100).toFixed(1)}%`, color: selected.acceleration >= 0 ? 'text-emerald-400' : 'text-red-400' },
                  ].map(({ label, value, color }) => (
                    <div key={label} className="p-3 rounded-lg bg-white/3 border border-[#1e2d45] text-center">
                      <div className={cn('text-xl font-bold mono', color)}>{value}</div>
                      <div className="text-xs text-slate-500 mt-0.5">{label}</div>
                    </div>
                  ))}
                </div>

                {/* Keywords */}
                <div className="mb-4">
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Keywords</p>
                  <div className="flex flex-wrap gap-2">
                    {selected.keywords.map((kw, i) => (
                      <span
                        key={kw}
                        className={cn('badge',
                          i < 3 ? 'badge-cyan' : 'badge-gray'
                        )}
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Score breakdown bar */}
                <div>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Trend Score Breakdown</p>
                  <div className="space-y-2">
                    {[
                      { label: 'Growth Velocity', score: selected.velocity, color: '#00d4ff' },
                      { label: 'Community Momentum', score: selected.momentum, color: '#7c3aed' },
                      { label: 'Novelty', score: Math.min(selected.trend_score * 0.9, 1), color: '#10b981' },
                    ].map(({ label, score, color }) => (
                      <div key={label} className="flex items-center gap-3">
                        <span className="text-xs text-slate-500 w-40 shrink-0">{label}</span>
                        <div className="flex-1 score-bar">
                          <div className="score-bar-fill" style={{ width: `${score * 100}%`, background: color }} />
                        </div>
                        <span className="text-xs text-slate-400 mono w-8 text-right">
                          {(score * 100).toFixed(0)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="card p-12 flex flex-col items-center justify-center text-slate-600">
                <Network size={40} className="opacity-30 mb-3" />
                <p>Select a topic to explore</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
