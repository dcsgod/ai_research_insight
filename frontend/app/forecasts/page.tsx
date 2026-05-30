'use client';

import { useEffect, useState } from 'react';
import { api, MOCK_TRENDING } from '@/lib/api';
import { ForecastSeries, TrendingItem } from '@/lib/types';
import { ForecastChart } from '@/components/ForecastChart';
import { BarChart3, TrendingUp, Zap, RefreshCw } from 'lucide-react';
import { cn, scoreColor } from '@/lib/utils';

function MiniMetric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="text-center">
      <div className={cn('text-lg font-bold mono', color)}>{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}

export default function ForecastsPage() {
  const [selectedItem, setSelectedItem] = useState<TrendingItem | null>(null);
  const [forecast, setForecast] = useState<ForecastSeries | null>(null);
  const [items, setItems] = useState<TrendingItem[]>(MOCK_TRENDING);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getTrending('30d', 15).then(setItems);
  }, []);

  const loadForecast = async (item: TrendingItem) => {
    setSelectedItem(item);
    setLoading(true);
    const fc = await api.getForecast(item.entity_id, item.entity_type);
    setForecast(fc);
    setLoading(false);
  };

  // Auto-load first item
  useEffect(() => {
    if (items.length > 0 && !selectedItem) {
      loadForecast(items[0]);
    }
  }, [items]);

  return (
    <div className="p-6 max-w-[1200px] mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <BarChart3 className="text-cyan-400" size={22} />
          Trend <span className="gradient-text">Forecasts</span>
        </h1>
        <p className="text-sm text-slate-500 mt-0.5">
          Prophet + XGBoost time-series momentum forecasting
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: Item selector */}
        <div className="space-y-2">
          <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Select Entity to Forecast
          </h2>
          {items.map((item) => (
            <button
              key={item.entity_id}
              onClick={() => loadForecast(item)}
              className={cn(
                'w-full text-left p-3 rounded-lg border transition-all',
                selectedItem?.entity_id === item.entity_id
                  ? 'border-cyan-400/30 bg-cyan-400/8'
                  : 'border-[#1e2d45] bg-white/2 hover:bg-white/4 hover:border-[#2a3f5f]'
              )}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-200 line-clamp-1">
                  {item.title}
                </span>
                <span className={cn('text-xs font-bold mono ml-2 shrink-0', scoreColor(item.trend_score))}>
                  {(item.trend_score * 100).toFixed(0)}
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span className={cn('badge text-[10px]',
                  item.entity_type === 'paper' ? 'badge-violet' :
                  item.entity_type === 'repo' ? 'badge-cyan' : 'badge-green'
                )}>
                  {item.entity_type}
                </span>
                <span className="text-xs text-slate-600">
                  v:{(item.growth_velocity * 100).toFixed(0)}%
                </span>
              </div>
            </button>
          ))}
        </div>

        {/* Right: Forecast chart */}
        <div className="xl:col-span-2">
          {loading ? (
            <div className="card p-6 h-[450px] shimmer flex items-center justify-center">
              <RefreshCw size={24} className="animate-spin text-slate-600" />
            </div>
          ) : forecast ? (
            <div className="card p-6">
              {/* Title */}
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-base font-semibold text-white">{forecast.title || selectedItem?.title}</h2>
                  <p className="text-xs text-slate-500 mt-0.5">
                    30-day historical + 14-day forecast with confidence bands
                  </p>
                </div>
                {forecast.breakout_detected && (
                  <span className="badge-green animate-pulse">🚀 Breakout</span>
                )}
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-3 gap-4 mb-5 p-3 rounded-lg bg-white/3 border border-[#1e2d45]">
                <MiniMetric
                  label="Momentum"
                  value={`${(forecast.momentum * 100).toFixed(0)}%`}
                  color={forecast.momentum > 0.5 ? 'text-emerald-400' : 'text-amber-400'}
                />
                <MiniMetric
                  label="Velocity"
                  value={`${(forecast.velocity * 100).toFixed(0)}%`}
                  color="text-cyan-400"
                />
                <MiniMetric
                  label="Acceleration"
                  value={`${forecast.acceleration >= 0 ? '+' : ''}${(forecast.acceleration * 100).toFixed(1)}%`}
                  color={forecast.acceleration >= 0 ? 'text-emerald-400' : 'text-red-400'}
                />
              </div>

              {/* Chart */}
              <ForecastChart data={forecast} height={320} showConfidence />
            </div>
          ) : (
            <div className="card p-12 flex flex-col items-center justify-center text-slate-600">
              <BarChart3 size={40} className="opacity-30 mb-3" />
              <p>Select an item to view its forecast</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
