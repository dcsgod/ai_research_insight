'use client';

import { useEffect, useState } from 'react';
import { WSMessage } from '@/lib/types';
import { createWSConnection } from '@/lib/api';
import { cn, entityTypeIcon } from '@/lib/utils';
import { Wifi, WifiOff, Radio } from 'lucide-react';

interface LiveFeedProps {
  className?: string;
}

interface FeedItem {
  id: string;
  type: string;
  title: string;
  score?: number;
  timestamp: string;
  entity_type?: string;
}

export function LiveFeed({ className }: LiveFeedProps) {
  const [connected, setConnected] = useState(false);
  const [feedItems, setFeedItems] = useState<FeedItem[]>([]);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  useEffect(() => {
    const ws = createWSConnection(
      (msg: WSMessage) => {
        setLastUpdate(new Date().toLocaleTimeString());

        if (msg.type === 'trend_update' && msg.data?.top_topics) {
          const items: FeedItem[] = msg.data.top_topics.map((t: any, i: number) => ({
            id: `${Date.now()}-${i}`,
            type: 'trend_update',
            title: `${t.name} — Score ${(t.score * 100).toFixed(0)}`,
            score: t.score,
            timestamp: msg.timestamp,
            entity_type: 'topic',
          }));
          setFeedItems((prev) => [...items, ...prev].slice(0, 20));
        } else if (msg.type === 'new_paper' && msg.data) {
          const item: FeedItem = {
            id: String(Date.now()),
            type: 'new_paper',
            title: msg.data.title || 'New paper ingested',
            score: msg.data.trend_score,
            timestamp: msg.timestamp,
            entity_type: 'paper',
          };
          setFeedItems((prev) => [item, ...prev].slice(0, 20));
        }
      },
      () => setConnected(true),
      () => setConnected(false)
    );

    // Seed with mock items while disconnected
    setFeedItems([
      { id: '1', type: 'trend_update', title: 'GraphRAG — Score 94', score: 0.94, timestamp: new Date().toISOString(), entity_type: 'topic' },
      { id: '2', type: 'new_paper', title: 'Mixture of Experts at Scale', score: 0.87, timestamp: new Date(Date.now() - 120000).toISOString(), entity_type: 'paper' },
      { id: '3', type: 'trend_update', title: 'vllm-project/vllm — Score 83', score: 0.83, timestamp: new Date(Date.now() - 300000).toISOString(), entity_type: 'repo' },
      { id: '4', type: 'new_paper', title: 'Flash Attention 3 on H100', score: 0.71, timestamp: new Date(Date.now() - 600000).toISOString(), entity_type: 'paper' },
      { id: '5', type: 'trend_update', title: 'Agent Frameworks — Score 74', score: 0.74, timestamp: new Date(Date.now() - 900000).toISOString(), entity_type: 'topic' },
    ]);

    return () => ws?.close();
  }, []);

  const typeLabel = (type: string) => {
    if (type === 'new_paper') return 'New Paper';
    if (type === 'trend_update') return 'Trend Update';
    return 'Update';
  };

  const typeBadge = (type: string) => {
    if (type === 'new_paper') return 'badge-violet';
    if (type === 'trend_update') return 'badge-cyan';
    return 'badge-gray';
  };

  return (
    <div className={cn('flex flex-col', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Radio size={14} className="text-cyan-400" />
          <span className="text-sm font-semibold text-slate-200">Live Feed</span>
        </div>
        <div className="flex items-center gap-2">
          {connected ? (
            <div className="flex items-center gap-1.5">
              <div className="live-dot w-1.5 h-1.5" />
              <span className="text-xs text-emerald-400">Live</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <WifiOff size={12} className="text-slate-500" />
              <span className="text-xs text-slate-500">Offline</span>
            </div>
          )}
          {lastUpdate && (
            <span className="text-xs text-slate-600 mono">{lastUpdate}</span>
          )}
        </div>
      </div>

      {/* Feed items */}
      <div className="space-y-2 overflow-y-auto max-h-[320px] pr-1">
        {feedItems.map((item, idx) => (
          <div
            key={item.id}
            className={cn(
              'flex items-start gap-2.5 p-2.5 rounded-lg',
              'bg-white/2 border border-[#1e2d45]/60',
              'hover:border-[#1e2d45] hover:bg-white/4 transition-all',
              idx === 0 && 'border-cyan-400/20 bg-cyan-400/3'
            )}
          >
            <span className="text-base shrink-0 mt-0.5">
              {entityTypeIcon(item.entity_type || 'topic')}
            </span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className={cn('badge text-[10px]', typeBadge(item.type))}>
                  {typeLabel(item.type)}
                </span>
                {item.score !== undefined && (
                  <span className="text-xs text-slate-500 mono">
                    {(item.score * 100).toFixed(0)}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-300 mt-0.5 line-clamp-1">{item.title}</p>
              <p className="text-[10px] text-slate-600 mt-0.5">
                {new Date(item.timestamp).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        {feedItems.length === 0 && (
          <div className="flex flex-col items-center justify-center py-8 text-slate-600">
            <Radio size={24} className="mb-2 opacity-40" />
            <span className="text-xs">Waiting for updates...</span>
          </div>
        )}
      </div>
    </div>
  );
}
