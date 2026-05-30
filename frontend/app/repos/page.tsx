'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_REPOS } from '@/lib/api';
import { Repository } from '@/lib/types';
import { Github, Star, GitFork, TrendingUp, ExternalLink, Code2 } from 'lucide-react';
import { cn, formatNumber, scoreColor, scoreBg, formatRelativeTime } from '@/lib/utils';

const LANGUAGES = ['', 'Python', 'TypeScript', 'Rust', 'Go', 'C++', 'Julia'];

function RepoCard({ repo, rank }: { repo: Repository; rank: number }) {
  return (
    <div className="card card-hover p-4 group animate-slide-up" style={{ animationDelay: `${rank * 30}ms` }}>
      <div className="flex items-start gap-3">
        <span className="rank-number pt-1">#{rank}</span>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <div className="flex-1 min-w-0">
              {/* Language badge */}
              {repo.language && (
                <span className="badge-gray text-[10px] mb-1.5 mr-2">{repo.language}</span>
              )}
              {repo.stars_today > 100 && (
                <span className="badge-amber text-[10px] mb-1.5">🔥 Trending</span>
              )}

              {/* Name */}
              <h3 className="text-sm font-semibold text-slate-100 group-hover:text-cyan-300 transition-colors font-mono">
                {repo.full_name}
              </h3>

              {/* Description */}
              {repo.description && (
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">{repo.description}</p>
              )}

              {/* Topics */}
              {repo.topics.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {repo.topics.slice(0, 5).map((t) => (
                    <span key={t} className="badge-cyan text-[10px]">{t}</span>
                  ))}
                </div>
              )}
            </div>

            {/* Score */}
            <div className={cn('shrink-0 px-2 py-1 rounded-lg border text-xs font-bold mono', scoreBg(repo.trend_score))}>
              <span className={scoreColor(repo.trend_score)}>
                {(repo.trend_score * 100).toFixed(0)}
              </span>
            </div>
          </div>

          {/* Stats */}
          <div className="flex items-center gap-4 mt-3 flex-wrap">
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Star size={12} className="text-amber-400" />
              <span className="font-semibold text-slate-300">{formatNumber(repo.stars)}</span>
              {repo.stars_today > 0 && (
                <span className="text-emerald-400">+{repo.stars_today} today</span>
              )}
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <GitFork size={12} />
              <span>{formatNumber(repo.forks)}</span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-500">
              <TrendingUp size={12} className="text-cyan-400" />
              <span className="text-cyan-400">{(repo.growth_velocity * 100).toFixed(0)}% velocity</span>
            </div>
            {repo.repo_updated_at && (
              <span className="text-xs text-slate-600 ml-auto">
                {formatRelativeTime(repo.repo_updated_at)}
              </span>
            )}
            <a href={repo.url} target="_blank" rel="noopener noreferrer"
              className="text-slate-600 hover:text-cyan-400 transition-colors">
              <ExternalLink size={12} />
            </a>
          </div>

          {/* Star velocity bar */}
          <div className="mt-2.5">
            <div className="score-bar">
              <div
                className="score-bar-fill bg-gradient-to-r from-amber-400 to-orange-500"
                style={{ width: `${Math.min(repo.growth_velocity * 100, 100)}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ReposPage() {
  const [repos, setRepos] = useState<Repository[]>(MOCK_REPOS);
  const [language, setLanguage] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const res = await api.getRepos({ language: language || undefined, limit: 20 });
      setRepos(res.items);
      setLoading(false);
    };
    load();
  }, [language]);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Github className="text-slate-300" size={22} />
            GitHub <span className="gradient-text">Repositories</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Trending AI repositories ranked by momentum
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2 mb-5 flex-wrap">
        <Code2 size={14} className="text-slate-500" />
        <span className="text-xs text-slate-500">Language:</span>
        <div className="flex gap-1">
          {LANGUAGES.map((lang) => (
            <button
              key={lang || 'all'}
              onClick={() => setLanguage(lang)}
              className={cn(
                'px-2.5 py-1.5 rounded-md text-xs font-medium transition-all',
                language === lang
                  ? 'bg-cyan-400/15 text-cyan-300 border border-cyan-400/25'
                  : 'text-slate-500 hover:text-slate-300 border border-transparent'
              )}
            >
              {lang || 'All'}
            </button>
          ))}
        </div>
        <span className="ml-auto text-xs text-slate-600">{repos.length} repos</span>
      </div>

      {/* Repo List */}
      <div className="space-y-2">
        {loading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="card p-4 h-24 shimmer" />
          ))
        ) : (
          repos.map((repo, i) => <RepoCard key={repo.id} repo={repo} rank={i + 1} />)
        )}
      </div>
    </div>
  );
}
