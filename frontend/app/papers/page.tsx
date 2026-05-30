'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_PAPERS } from '@/lib/api';
import { Paper } from '@/lib/types';
import { FileText, Search, ExternalLink, Github, Star } from 'lucide-react';
import { cn, formatDate, formatNumber, scoreColor, scoreBg } from '@/lib/utils';

function PaperCard({ paper }: { paper: Paper }) {
  const [insight, setInsight] = useState<string | null>(null);
  const [loadingInsight, setLoadingInsight] = useState(false);

  const loadInsight = async () => {
    if (insight) return;
    setLoadingInsight(true);
    const text = await api.getPaperInsight(paper.id);
    setInsight(text);
    setLoadingInsight(false);
  };

  return (
    <div className="card card-hover p-5 group animate-fade-in">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          {/* Badges */}
          <div className="flex items-center gap-2 flex-wrap mb-2">
            <span className={cn('badge text-[10px]',
              paper.source === 'arxiv' ? 'badge-cyan' :
              paper.source === 'huggingface' ? 'badge-amber' : 'badge-violet'
            )}>
              {paper.source}
            </span>
            {paper.primary_category && (
              <span className="badge-gray text-[10px]">{paper.primary_category}</span>
            )}
            {paper.has_implementation && (
              <span className="badge-green text-[10px]">💻 Code Available</span>
            )}
          </div>

          {/* Title */}
          <h3 className="text-sm font-semibold text-slate-100 leading-snug group-hover:text-cyan-300 transition-colors">
            {paper.title}
          </h3>

          {/* Authors */}
          <p className="text-xs text-slate-500 mt-1">
            {paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}
          </p>

          {/* Abstract */}
          {paper.abstract && (
            <p className="text-xs text-slate-400 mt-2 line-clamp-3 leading-relaxed">
              {paper.abstract}
            </p>
          )}

          {/* AI Insight (expandable) */}
          {insight ? (
            <div className="mt-2 px-3 py-2 rounded-lg bg-violet-500/8 border border-violet-400/15">
              <p className="text-xs text-violet-300/80 italic">&ldquo;{insight}&rdquo;</p>
            </div>
          ) : (
            <button
              onClick={loadInsight}
              disabled={loadingInsight}
              className="mt-2 text-xs text-slate-500 hover:text-violet-400 transition-colors"
            >
              {loadingInsight ? '⚡ Generating insight...' : '⚡ Generate AI insight'}
            </button>
          )}

          {/* Metrics */}
          <div className="flex items-center gap-4 mt-3 flex-wrap">
            {paper.published_date && (
              <span className="text-xs text-slate-500">{formatDate(paper.published_date)}</span>
            )}
            <span className="text-xs text-slate-500">
              📎 {formatNumber(paper.citation_count)} citations
            </span>
            {paper.url && (
              <a href={paper.url} target="_blank" rel="noopener noreferrer"
                className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1">
                <ExternalLink size={11} />
                arXiv
              </a>
            )}
            {paper.github_url && (
              <a href={paper.github_url} target="_blank" rel="noopener noreferrer"
                className="text-xs text-slate-400 hover:text-white flex items-center gap-1">
                <Github size={11} />
                Code
              </a>
            )}
          </div>
        </div>

        {/* Score */}
        <div className="shrink-0 text-right">
          <div className={cn('inline-flex items-center gap-1 px-2 py-1 rounded-lg border text-xs font-bold mono', scoreBg(paper.trend_score))}>
            <span className={scoreColor(paper.trend_score)}>
              {(paper.trend_score * 100).toFixed(0)}
            </span>
          </div>
          <div className="mt-1 text-[10px] text-slate-600">trend score</div>
        </div>
      </div>
    </div>
  );
}

export default function PapersPage() {
  const [papers, setPapers] = useState<Paper[]>(MOCK_PAPERS);
  const [query, setQuery] = useState('');
  const [source, setSource] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      if (query.length > 2) {
        const results = await api.searchPapers(query);
        setPapers(results);
      } else {
        const res = await api.getPapers({ source: source || undefined, limit: 20 });
        setPapers(res.items);
      }
      setLoading(false);
    };
    const timer = setTimeout(load, 400);
    return () => clearTimeout(timer);
  }, [query, source]);

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <FileText className="text-violet-400" size={22} />
            Research <span className="gradient-text">Papers</span>
          </h1>
          <p className="text-sm text-slate-500 mt-0.5">arXiv · HuggingFace · PapersWithCode</p>
        </div>
      </div>

      {/* Search + Filter */}
      <div className="flex gap-3 mb-5">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            placeholder="Semantic search papers..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2.5 bg-white/5 border border-[#1e2d45] rounded-lg text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-400/40 focus:bg-white/7 transition-all"
          />
        </div>
        <select
          value={source}
          onChange={(e) => setSource(e.target.value)}
          className="px-3 py-2.5 bg-white/5 border border-[#1e2d45] rounded-lg text-sm text-slate-300 focus:outline-none focus:border-cyan-400/40 appearance-none cursor-pointer"
        >
          <option value="">All Sources</option>
          <option value="arxiv">arXiv</option>
          <option value="huggingface">HuggingFace</option>
          <option value="paperswithcode">PapersWithCode</option>
        </select>
      </div>

      {/* Paper List */}
      <div className="space-y-3">
        {loading ? (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="card p-5 h-36 shimmer" />
          ))
        ) : papers.length === 0 ? (
          <div className="card p-12 text-center text-slate-600">
            <FileText size={32} className="mx-auto mb-3 opacity-30" />
            <p>No papers found</p>
          </div>
        ) : (
          papers.map((paper) => <PaperCard key={paper.id} paper={paper} />)
        )}
      </div>
    </div>
  );
}
