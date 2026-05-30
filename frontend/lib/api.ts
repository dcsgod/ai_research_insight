/**
 * API client for the AI Research Intelligence Platform backend.
 * Centralizes all HTTP calls with typed responses and error handling.
 */

import type {
  Paper,
  Repository,
  Topic,
  TrendingItem,
  ForecastSeries,
  DashboardSummary,
  TrendSignal,
  TopicGraphData,
  PaginatedResponse,
  Timeframe,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_V1 = `${API_BASE}/api/v1`;

// ── Fetch Helper ──────────────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_V1}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ message: res.statusText }));
    throw new Error(err.message || `API error ${res.status}`);
  }
  return res.json();
}

// ── Mock Data (used when backend unavailable) ─────────────────────────────────

export const MOCK_DASHBOARD: DashboardSummary = {
  total_papers: 12847,
  total_repos: 8341,
  total_topics: 142,
  papers_today: 89,
  repos_trending: 47,
  top_category: 'cs.AI',
  avg_trend_score: 0.67,
  breakout_topics: 7,
  last_ingestion: new Date().toISOString(),
};

export const MOCK_TRENDING: TrendingItem[] = [
  { entity_id: '1', entity_type: 'topic', title: 'GraphRAG', description: 'Graph-augmented retrieval for enterprise knowledge systems', url: '#', trend_score: 0.94, rank: 1, delta_rank: 2, growth_velocity: 0.82, tags: ['RAG', 'Knowledge Graph', 'Enterprise'], insight: 'GraphRAG is rapidly gaining momentum due to enterprise retrieval applications and multi-agent workflows.' },
  { entity_id: '2', entity_type: 'paper', title: 'Mixture of Experts at Scale', description: 'Scaling sparse MoE architectures to 1T parameters with minimal compute', url: '#', trend_score: 0.87, rank: 2, delta_rank: -1, growth_velocity: 0.74, tags: ['MoE', 'Scaling', 'Efficiency'], insight: 'MoE architectures are becoming the dominant paradigm for training frontier models efficiently.' },
  { entity_id: '3', entity_type: 'repo', title: 'microsoft/graphrag', description: 'A modular graph-based Retrieval-Augmented Generation (RAG) system', url: '#', trend_score: 0.85, rank: 3, delta_rank: 5, growth_velocity: 0.91, tags: ['Python', 'LLM', 'RAG'], insight: 'GraphRAG repo crossed 20k stars this week, driven by enterprise adoption and Microsoft backing.' },
  { entity_id: '4', entity_type: 'topic', title: 'Multimodal LLMs', description: 'Vision-language models for unified understanding', url: '#', trend_score: 0.81, rank: 4, delta_rank: 0, growth_velocity: 0.63, tags: ['Vision', 'Language', 'Multimodal'], insight: 'New multimodal benchmarks show 40% improvement over text-only baselines in reasoning tasks.' },
  { entity_id: '5', entity_type: 'paper', title: 'RLHF-Free Alignment via DPO', description: 'Direct Preference Optimization without reward model overhead', url: '#', trend_score: 0.78, rank: 5, delta_rank: 3, growth_velocity: 0.58, tags: ['Alignment', 'RLHF', 'DPO'], insight: 'DPO is replacing RLHF in production alignment pipelines due to its simplicity and effectiveness.' },
  { entity_id: '6', entity_type: 'repo', title: 'vllm-project/vllm', description: 'High-throughput LLM serving engine with PagedAttention', url: '#', trend_score: 0.76, rank: 6, delta_rank: -2, growth_velocity: 0.71, tags: ['Python', 'Inference', 'LLM'], insight: 'vLLM adoption is accelerating as the de-facto standard for production LLM serving.' },
  { entity_id: '7', entity_type: 'topic', title: 'Agent Frameworks', description: 'Multi-agent orchestration and tool-use systems', url: '#', trend_score: 0.74, rank: 7, delta_rank: 4, growth_velocity: 0.67, tags: ['Agents', 'LLM', 'Automation'], insight: 'Agent frameworks are converging around standardized protocols like MCP, driving rapid ecosystem growth.' },
  { entity_id: '8', entity_type: 'paper', title: 'Flash Attention 3', description: 'IO-aware exact attention with hardware-aware optimizations for H100', url: '#', trend_score: 0.71, rank: 8, delta_rank: 1, growth_velocity: 0.55, tags: ['Efficiency', 'Attention', 'GPU'], insight: 'FlashAttention 3 delivers 2x speedup on H100s, becoming essential for frontier model training.' },
];

export const MOCK_PAPERS: Paper[] = [
  { id: '1', title: 'Attention Is All You Need: Revisited for 2025', abstract: 'We revisit the transformer architecture with modern improvements including rotary embeddings, grouped query attention, and flash attention, showing significant improvements in both quality and efficiency.', authors: ['Vaswani, A.', 'Brown, T.'], url: 'https://arxiv.org', pdf_url: 'https://arxiv.org', published_date: '2025-01-15', categories: ['cs.LG', 'cs.CL'], primary_category: 'cs.LG', source: 'arxiv', citation_count: 342, has_implementation: true, trend_score: 0.89, growth_velocity: 0.74, novelty_score: 0.82 },
  { id: '2', title: 'Scaling Laws for Mixture of Experts', abstract: 'We study how MoE scaling laws differ from dense model scaling laws, and provide guidelines for choosing the number of experts and routing strategies at scale.', authors: ['Fedus, W.', 'Zoph, B.', 'Shazeer, N.'], url: 'https://arxiv.org', pdf_url: 'https://arxiv.org', published_date: '2025-02-03', categories: ['cs.AI', 'cs.LG'], primary_category: 'cs.AI', source: 'arxiv', citation_count: 198, has_implementation: true, trend_score: 0.84, growth_velocity: 0.69, novelty_score: 0.77 },
  { id: '3', title: 'GraphRAG: From Local to Global Knowledge Retrieval', abstract: 'We introduce GraphRAG, a graph-based approach to RAG that constructs a knowledge graph from source documents and uses community detection for hierarchical summarization.', authors: ['Edge, D.', 'Trinh, H.'], url: 'https://arxiv.org', pdf_url: 'https://arxiv.org', github_url: 'https://github.com/microsoft/graphrag', published_date: '2025-01-28', categories: ['cs.AI', 'cs.IR'], primary_category: 'cs.AI', source: 'arxiv', citation_count: 287, has_implementation: true, trend_score: 0.92, growth_velocity: 0.88, novelty_score: 0.91 },
];

export const MOCK_REPOS: Repository[] = [
  { id: '1', name: 'graphrag', full_name: 'microsoft/graphrag', description: 'A modular graph-based Retrieval-Augmented Generation (RAG) system', url: 'https://github.com/microsoft/graphrag', language: 'Python', topics: ['rag', 'llm', 'knowledge-graph'], owner: 'microsoft', stars: 21847, forks: 2341, watchers: 21847, stars_today: 387, open_issues: 156, source: 'github', trend_score: 0.94, growth_velocity: 0.91, github_activity_score: 0.88, repo_updated_at: new Date().toISOString() },
  { id: '2', name: 'vllm', full_name: 'vllm-project/vllm', description: 'Easy, fast, and cheap LLM serving with PagedAttention', url: 'https://github.com/vllm-project/vllm', language: 'Python', topics: ['llm', 'serving', 'inference'], owner: 'vllm-project', stars: 35621, forks: 4892, watchers: 35621, stars_today: 241, open_issues: 743, source: 'github', trend_score: 0.87, growth_velocity: 0.71, github_activity_score: 0.84, repo_updated_at: new Date().toISOString() },
  { id: '3', name: 'ollama', full_name: 'ollama/ollama', description: 'Get up and running with large language models locally', url: 'https://github.com/ollama/ollama', language: 'Go', topics: ['llm', 'local', 'inference'], owner: 'ollama', stars: 98341, forks: 7812, watchers: 98341, stars_today: 412, open_issues: 892, source: 'github', trend_score: 0.83, growth_velocity: 0.79, github_activity_score: 0.92, repo_updated_at: new Date().toISOString() },
];

export const MOCK_TOPICS: Topic[] = [
  { id: 0, name: 'GraphRAG & Knowledge Graphs', keywords: ['graphrag', 'knowledge-graph', 'retrieval', 'enterprise', 'neo4j'], paper_count: 47, repo_count: 23, trend_score: 0.94, momentum: 0.88, velocity: 0.82, acceleration: 0.14 },
  { id: 1, name: 'Mixture of Experts', keywords: ['moe', 'sparse', 'routing', 'experts', 'scaling'], paper_count: 38, repo_count: 15, trend_score: 0.87, momentum: 0.74, velocity: 0.69, acceleration: 0.08 },
  { id: 2, name: 'Multimodal Vision-Language', keywords: ['multimodal', 'vision', 'language', 'vlm', 'image'], paper_count: 92, repo_count: 41, trend_score: 0.81, momentum: 0.66, velocity: 0.63, acceleration: 0.05 },
  { id: 3, name: 'RLHF & Alignment', keywords: ['rlhf', 'dpo', 'alignment', 'preference', 'reward'], paper_count: 64, repo_count: 29, trend_score: 0.78, momentum: 0.61, velocity: 0.58, acceleration: 0.03 },
  { id: 4, name: 'LLM Inference Optimization', keywords: ['quantization', 'inference', 'serving', 'throughput', 'kv-cache'], paper_count: 53, repo_count: 34, trend_score: 0.76, momentum: 0.72, velocity: 0.71, acceleration: 0.09 },
];

// ── API Functions ─────────────────────────────────────────────────────────────

export const api = {
  // Dashboard
  async getDashboard(): Promise<DashboardSummary> {
    try {
      return await apiFetch<DashboardSummary>('/trends/dashboard');
    } catch {
      return MOCK_DASHBOARD;
    }
  },

  // Trends
  async getTrending(timeframe: Timeframe = '7d', limit = 20, entity_type?: string): Promise<TrendingItem[]> {
    try {
      const params = new URLSearchParams({ timeframe, limit: String(limit) });
      if (entity_type) params.append('entity_type', entity_type);
      return await apiFetch<TrendingItem[]>(`/trends/?${params}`);
    } catch {
      return MOCK_TRENDING;
    }
  },

  async getTrendingTopics(limit = 10): Promise<Topic[]> {
    try {
      return await apiFetch<Topic[]>(`/trends/topics?limit=${limit}`);
    } catch {
      return MOCK_TOPICS;
    }
  },

  async getEntitySignals(entityId: string): Promise<TrendSignal[]> {
    try {
      return await apiFetch<TrendSignal[]>(`/trends/${entityId}/signals`);
    } catch {
      // Generate mock signal data
      const now = Date.now();
      return Array.from({ length: 30 }, (_, i) => ({
        date: new Date(now - (29 - i) * 86400000).toISOString().split('T')[0],
        value: Math.random() * 100 + i * 2,
        signal_type: 'mentions' as const,
      }));
    }
  },

  // Papers
  async getPapers(params?: {
    limit?: number;
    offset?: number;
    source?: string;
    category?: string;
  }): Promise<PaginatedResponse<Paper>> {
    try {
      const p = new URLSearchParams();
      if (params?.limit) p.append('limit', String(params.limit));
      if (params?.offset) p.append('offset', String(params.offset));
      if (params?.source) p.append('source', params.source);
      if (params?.category) p.append('category', params.category);
      return await apiFetch<PaginatedResponse<Paper>>(`/papers/?${p}`);
    } catch {
      return { items: MOCK_PAPERS, total: MOCK_PAPERS.length, page: 1, page_size: 20, has_next: false };
    }
  },

  async searchPapers(query: string, limit = 10): Promise<Paper[]> {
    try {
      return await apiFetch<Paper[]>(`/papers/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    } catch {
      return MOCK_PAPERS.filter(p =>
        p.title.toLowerCase().includes(query.toLowerCase())
      );
    }
  },

  async getPaperInsight(paperId: string): Promise<string> {
    try {
      const res = await apiFetch<{ insight: string }>(`/papers/${paperId}/insight`);
      return res.insight;
    } catch {
      return 'This paper is showing strong momentum in the AI research community with growing adoption and citation acceleration.';
    }
  },

  // Repos
  async getRepos(params?: { limit?: number; language?: string }): Promise<PaginatedResponse<Repository>> {
    try {
      const p = new URLSearchParams();
      if (params?.limit) p.append('limit', String(params.limit));
      if (params?.language) p.append('language', params.language);
      return await apiFetch<PaginatedResponse<Repository>>(`/repos/?${p}`);
    } catch {
      return { items: MOCK_REPOS, total: MOCK_REPOS.length, page: 1, page_size: 20, has_next: false };
    }
  },

  // Forecasts
  async getForecast(entityId: string, entityType: string): Promise<ForecastSeries | null> {
    try {
      return await apiFetch<ForecastSeries>(`/forecasts/${entityId}?entity_type=${entityType}`);
    } catch {
      const now = Date.now();
      const historical = Array.from({ length: 30 }, (_, i) => ({
        date: new Date(now - (29 - i) * 86400000).toISOString().split('T')[0],
        value: 20 + i * 2.5 + Math.random() * 10,
        lower: 15 + i * 2,
        upper: 25 + i * 3,
      }));
      const forecast = Array.from({ length: 14 }, (_, i) => ({
        date: new Date(now + (i + 1) * 86400000).toISOString().split('T')[0],
        value: historical[29].value + (i + 1) * 3 + Math.random() * 5,
        lower: historical[29].value + (i + 1) * 1.5,
        upper: historical[29].value + (i + 1) * 4.5,
      }));
      return {
        entity_id: entityId,
        entity_type: entityType as any,
        title: 'Trend Forecast',
        historical,
        forecast,
        momentum: 0.74,
        velocity: 0.62,
        acceleration: 0.08,
        breakout_detected: false,
      };
    }
  },

  // Topics
  async getTopics(limit = 20): Promise<Topic[]> {
    try {
      return await apiFetch<Topic[]>(`/topics/?limit=${limit}`);
    } catch {
      return MOCK_TOPICS;
    }
  },

  async getTopicGraph(): Promise<TopicGraphData> {
    try {
      return await apiFetch<TopicGraphData>('/topics/graph');
    } catch {
      return {
        nodes: MOCK_TOPICS.map((t, i) => ({
          id: i,
          name: t.name,
          size: t.paper_count + t.repo_count,
          keywords: t.keywords,
          trend_score: t.trend_score,
        })),
        edges: [
          { source: 0, target: 1, weight: 0.6 },
          { source: 0, target: 2, weight: 0.4 },
          { source: 1, target: 4, weight: 0.5 },
          { source: 2, target: 3, weight: 0.3 },
          { source: 3, target: 4, weight: 0.7 },
        ],
      };
    }
  },

  // Ingestion
  async triggerIngestion(sources?: string[]): Promise<{ status: string; job_id: string }> {
    try {
      return await apiFetch('/ingestion/run', {
        method: 'POST',
        body: JSON.stringify({ sources: sources || ['all'] }),
      });
    } catch {
      return { status: 'queued', job_id: 'mock-job-001' };
    }
  },

  // Daily insight
  async getDailyInsight(): Promise<string> {
    try {
      const res = await apiFetch<{ insight: string }>('/insights/daily');
      return res.insight;
    } catch {
      return 'GraphRAG and MoE architectures dominate today\'s AI research landscape. GitHub activity in the LLM inference space is at an all-time high, with vLLM and SGLang competing for production adoption. Multimodal models continue to close the gap with specialized vision systems.';
    }
  },
};

// ── WebSocket Client ──────────────────────────────────────────────────────────

export function createWSConnection(
  onMessage: (msg: any) => void,
  onConnect?: () => void,
  onDisconnect?: () => void
): WebSocket | null {
  if (typeof window === 'undefined') return null;
  const wsUrl = (process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000').replace(/^http/, 'ws');
  try {
    const ws = new WebSocket(`${wsUrl}/ws/live-feed`);
    ws.onopen = () => {
      console.log('WebSocket connected');
      onConnect?.();
    };
    ws.onmessage = (e) => {
      try {
        onMessage(JSON.parse(e.data));
      } catch {
        // ignore malformed messages
      }
    };
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      onDisconnect?.();
    };
    return ws;
  } catch (e) {
    console.error('WebSocket creation failed:', e);
    return null;
  }
}
