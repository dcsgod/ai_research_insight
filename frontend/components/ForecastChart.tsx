'use client';

import { ForecastSeries } from '@/lib/types';
import { useEffect, useRef } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ForecastChartProps {
  data: ForecastSeries;
  height?: number;
  showConfidence?: boolean;
  className?: string;
}

export function ForecastChart({
  data,
  height = 280,
  showConfidence = true,
  className,
}: ForecastChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<any>(null);

  const momentum = data.momentum;
  const MomentumIcon = momentum > 0.1 ? TrendingUp : momentum < -0.1 ? TrendingDown : Minus;
  const momentumColor = momentum > 0.1 ? '#10b981' : momentum < -0.1 ? '#ef4444' : '#94a3b8';

  useEffect(() => {
    let echarts: any;

    const initChart = async () => {
      try {
        echarts = await import('echarts');
      } catch {
        return;
      }

      if (!chartRef.current) return;

      if (chartInstance.current) {
        chartInstance.current.dispose();
      }

      const chart = echarts.init(chartRef.current, null, {
        renderer: 'canvas',
        width: 'auto',
        height,
      });
      chartInstance.current = chart;

      const histDates = data.historical.map((p) => p.date);
      const histValues = data.historical.map((p) => p.value);
      const fcDates = data.forecast.map((p) => p.date);
      const fcValues = data.forecast.map((p) => p.value);
      const fcLower = data.forecast.map((p) => p.lower);
      const fcUpper = data.forecast.map((p) => p.upper);

      const allDates = [...histDates, ...fcDates];
      const splitIdx = histDates.length - 1;

      const option = {
        backgroundColor: 'transparent',
        grid: { top: 16, right: 16, bottom: 32, left: 48, containLabel: false },
        tooltip: {
          trigger: 'axis',
          backgroundColor: '#0d1421',
          borderColor: '#1e2d45',
          textStyle: { color: '#e2e8f0', fontSize: 12 },
          formatter: (params: any[]) => {
            const date = params[0]?.axisValue || '';
            let html = `<div style="font-weight:600;margin-bottom:4px;color:#94a3b8">${date}</div>`;
            params.forEach((p: any) => {
              if (p.value !== undefined && p.seriesName !== 'confidence') {
                const color = p.color?.colorStops?.[0]?.color || p.color || '#00d4ff';
                html += `<div style="display:flex;align-items:center;gap:6px">
                  <span style="width:8px;height:8px;border-radius:50%;background:${color};display:inline-block"></span>
                  <span style="color:#94a3b8">${p.seriesName}:</span>
                  <span style="font-weight:600;color:#e2e8f0">${typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</span>
                </div>`;
              }
            });
            return html;
          },
          axisPointer: { lineStyle: { color: '#1e2d45' } },
        },
        xAxis: {
          type: 'category',
          data: allDates,
          axisLine: { lineStyle: { color: '#1e2d45' } },
          axisTick: { show: false },
          axisLabel: {
            color: '#64748b',
            fontSize: 11,
            interval: Math.floor(allDates.length / 6),
            formatter: (val: string) => val.slice(5),
          },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          axisLine: { show: false },
          axisTick: { show: false },
          axisLabel: { color: '#64748b', fontSize: 11 },
          splitLine: { lineStyle: { color: '#1e2d45', type: 'dashed' } },
        },
        series: [
          // Historical line
          {
            name: 'Historical',
            type: 'line',
            data: [...histValues, ...new Array(fcDates.length).fill(null)],
            smooth: true,
            symbol: 'none',
            lineStyle: {
              width: 2.5,
              color: {
                type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
                colorStops: [
                  { offset: 0, color: '#0099bb' },
                  { offset: 1, color: '#00d4ff' },
                ],
              },
            },
            areaStyle: {
              opacity: 0.15,
              color: {
                type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                colorStops: [
                  { offset: 0, color: '#00d4ff' },
                  { offset: 1, color: 'transparent' },
                ],
              },
            },
          },
          // Forecast line
          {
            name: 'Forecast',
            type: 'line',
            data: [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcValues],
            smooth: true,
            symbol: 'none',
            lineStyle: {
              width: 2,
              type: 'dashed',
              color: '#7c3aed',
            },
          },
          // Confidence band (upper)
          ...(showConfidence ? [
            {
              name: 'confidence',
              type: 'line',
              data: [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcUpper],
              lineStyle: { opacity: 0 },
              stack: 'confidence',
              symbol: 'none',
              areaStyle: { opacity: 0 },
            },
            {
              name: 'confidence',
              type: 'line',
              data: [...new Array(histDates.length - 1).fill(null), histValues[histValues.length - 1], ...fcLower],
              lineStyle: { opacity: 0 },
              stack: 'confidence',
              symbol: 'none',
              areaStyle: {
                opacity: 0.08,
                color: '#7c3aed',
              },
            },
          ] : []),
          // Forecast start marker
          {
            name: 'Now',
            type: 'line',
            markLine: {
              silent: true,
              data: [{ xAxis: splitIdx }],
              lineStyle: { color: '#1e2d45', type: 'solid', width: 1 },
              label: {
                formatter: 'Now',
                color: '#64748b',
                fontSize: 10,
                position: 'insideEndTop',
              },
              symbol: 'none',
            },
          },
        ],
      };

      chart.setOption(option);

      const handleResize = () => chart.resize();
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    };

    initChart();

    return () => {
      chartInstance.current?.dispose();
    };
  }, [data, height, showConfidence]);

  return (
    <div className={cn('w-full', className)}>
      {/* Momentum metrics */}
      <div className="flex items-center gap-4 mb-3">
        <div className="flex items-center gap-1.5">
          <MomentumIcon size={14} style={{ color: momentumColor }} />
          <span className="text-xs text-slate-500">Momentum</span>
          <span className="text-xs font-semibold" style={{ color: momentumColor }}>
            {(momentum * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-slate-500">Velocity</span>
          <span className="text-xs font-semibold text-cyan-300">{(data.velocity * 100).toFixed(0)}%</span>
        </div>
        {data.breakout_detected && (
          <span className="badge-green text-[10px] animate-pulse">🚀 Breakout</span>
        )}
      </div>

      {/* ECharts container */}
      <div ref={chartRef} style={{ height, width: '100%' }} />

      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 justify-end">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 bg-cyan-400" />
          <span className="text-xs text-slate-500">Historical</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5 bg-violet-500 border-dashed" style={{ borderTop: '2px dashed #7c3aed', height: 0 }} />
          <span className="text-xs text-slate-500">Forecast</span>
        </div>
        {showConfidence && (
          <div className="flex items-center gap-1.5">
            <div className="w-5 h-3 bg-violet-500/15 rounded-sm" />
            <span className="text-xs text-slate-500">Confidence</span>
          </div>
        )}
      </div>
    </div>
  );
}
