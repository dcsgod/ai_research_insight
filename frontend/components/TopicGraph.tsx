'use client';

import { useEffect, useRef } from 'react';
import { TopicGraphData } from '@/lib/types';
import { cn } from '@/lib/utils';

interface TopicGraphProps {
  data: TopicGraphData;
  height?: number;
  className?: string;
  onNodeClick?: (nodeId: number) => void;
}

export function TopicGraph({ data, height = 400, className, onNodeClick }: TopicGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || data.nodes.length === 0) return;

    const initD3 = async () => {
      const d3 = await import('d3').catch(() => null);
      if (!d3) return;

      const container = containerRef.current!;
      const width = container.clientWidth || 600;
      const h = height;

      const svg = d3.select(svgRef.current!);
      svg.selectAll('*').remove();

      svg.attr('width', width).attr('height', h)
        .attr('viewBox', `0 0 ${width} ${h}`);

      // Background
      svg.append('rect')
        .attr('width', width)
        .attr('height', h)
        .attr('fill', 'transparent');

      const g = svg.append('g');

      // Zoom behavior
      const zoom = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 3])
        .on('zoom', (event) => g.attr('transform', event.transform));
      svg.call(zoom);

      // Scale node size
      const maxSize = Math.max(...data.nodes.map((n) => n.size));
      const sizeScale = d3.scaleSqrt().domain([0, maxSize]).range([12, 40]);
      const scoreScale = d3.scaleLinear().domain([0, 1]).range([0, 1]);

      // Color scale
      const colors = ['#00d4ff', '#7c3aed', '#10b981', '#f59e0b', '#ef4444', '#6366f1'];
      const colorScale = (idx: number) => colors[idx % colors.length];

      // Simulation
      const nodes = data.nodes.map((n) => ({ ...n, r: sizeScale(n.size) }));
      const links = data.edges.map((e) => ({ ...e }));

      const simulation = d3.forceSimulation(nodes as any)
        .force('link', d3.forceLink(links).id((d: any) => d.id).distance((d: any) => 120 - d.weight * 60))
        .force('charge', d3.forceManyBody().strength(-200))
        .force('center', d3.forceCenter(width / 2, h / 2))
        .force('collision', d3.forceCollide().radius((d: any) => d.r + 8));

      // Render links
      const link = g.append('g')
        .selectAll('line')
        .data(links)
        .join('line')
        .attr('stroke', '#1e2d45')
        .attr('stroke-width', (d: any) => d.weight * 2)
        .attr('stroke-opacity', 0.6);

      // Render nodes
      const node = g.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('cursor', 'pointer')
        .on('click', (_event: any, d: any) => onNodeClick?.(d.id))
        .call(
          d3.drag<any, any>()
            .on('start', (event, d) => {
              if (!event.active) simulation.alphaTarget(0.3).restart();
              d.fx = d.x; d.fy = d.y;
            })
            .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y; })
            .on('end', (event, d) => {
              if (!event.active) simulation.alphaTarget(0);
              d.fx = null; d.fy = null;
            })
        );

      // Node glow
      node.append('circle')
        .attr('r', (d: any) => d.r + 4)
        .attr('fill', (_d: any, i: number) => colorScale(i))
        .attr('opacity', 0.1);

      // Node circle
      node.append('circle')
        .attr('r', (d: any) => d.r)
        .attr('fill', (_d: any, i: number) => colorScale(i))
        .attr('fill-opacity', 0.2)
        .attr('stroke', (_d: any, i: number) => colorScale(i))
        .attr('stroke-width', 1.5)
        .attr('stroke-opacity', 0.7)
        .on('mouseover', function(_event: any, _d: any) {
          d3.select(this).attr('fill-opacity', 0.35).attr('stroke-opacity', 1);
        })
        .on('mouseout', function() {
          d3.select(this).attr('fill-opacity', 0.2).attr('stroke-opacity', 0.7);
        });

      // Node label
      node.append('text')
        .text((d: any) => d.name.split(' ')[0])
        .attr('text-anchor', 'middle')
        .attr('dy', '0.35em')
        .attr('font-size', (d: any) => Math.max(9, Math.min(13, d.r / 2.5)))
        .attr('fill', '#e2e8f0')
        .attr('font-family', 'Inter, sans-serif')
        .attr('font-weight', '500')
        .attr('pointer-events', 'none');

      // Tooltip
      const tooltip = d3.select(container)
        .append('div')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style('background', '#0d1421')
        .style('border', '1px solid #1e2d45')
        .style('border-radius', '8px')
        .style('padding', '8px 12px')
        .style('font-size', '12px')
        .style('color', '#e2e8f0')
        .style('pointer-events', 'none')
        .style('opacity', 0)
        .style('z-index', 10);

      node
        .on('mouseover.tip', (_event: any, d: any) => {
          tooltip.style('opacity', 1).html(
            `<div style="font-weight:600;color:#00d4ff;margin-bottom:4px">${d.name}</div>
             <div style="color:#64748b">Papers+Repos: ${d.size}</div>
             ${d.keywords ? `<div style="color:#94a3b8;margin-top:4px">${d.keywords.slice(0,3).join(' · ')}</div>` : ''}`
          );
        })
        .on('mousemove.tip', (event: any) => {
          const rect = container.getBoundingClientRect();
          tooltip
            .style('left', (event.clientX - rect.left + 12) + 'px')
            .style('top', (event.clientY - rect.top - 40) + 'px');
        })
        .on('mouseout.tip', () => tooltip.style('opacity', 0));

      // Tick
      simulation.on('tick', () => {
        link
          .attr('x1', (d: any) => d.source.x)
          .attr('y1', (d: any) => d.source.y)
          .attr('x2', (d: any) => d.target.x)
          .attr('y2', (d: any) => d.target.y);

        node.attr('transform', (d: any) => `translate(${d.x},${d.y})`);
      });

      return () => {
        simulation.stop();
        tooltip.remove();
      };
    };

    const cleanup = initD3();
    return () => {
      cleanup.then((fn) => fn?.());
    };
  }, [data, height, onNodeClick]);

  return (
    <div ref={containerRef} className={cn('relative w-full', className)} style={{ height }}>
      <svg ref={svgRef} className="w-full" style={{ height }} />
      <div className="absolute bottom-2 right-2 text-xs text-slate-600">
        Drag to explore · Scroll to zoom
      </div>
    </div>
  );
}
