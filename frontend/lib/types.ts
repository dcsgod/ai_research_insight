/**
 * TypeScript types for the AI Research Intelligence Platform.
 * Mirrors backend Pydantic schemas.
 */

// ── Entity Types ──────────────────────────────────────────────────────────────

export type EntityType = 'paper' | 'repo' | 'topic';

export interface Paper {
  id: string;
  title: string;
  abstract?: string;
  authors: string[];
  url?: string;
  pdf_url?: string;
  github_url?: string;
  published_date?: string;
  categories: string[];
  primary_category?: string;
  source: 'arxiv' | 'huggingface' | 'paperswithcode';
  citation_count: number;
  has_implementation: boolean;
  trend_score: number;
  growth_velocity: number;
  novelty_score: number;
}

export interface Repository {
  id: string;
  name: string;
  full_name: string;
  description?: string;
  url: string;
  language?: string;
  topics: string[];
  owner: string;
  stars: number;
  forks: number;
  watchers: number;
  stars_today: number;
  open_issues: number;
  source: string;
  trend_score: number;
  growth_velocity: number;
  github_activity_score: number;
  repo_updated_at?: string;
}

export interface Topic {
  id: string | number;
  name: string;
  description?: string;
  keywords: string[];
  paper_count: number;
  repo_count: number;
  trend_score: number;
  momentum: number;
  velocity: number;
  acceleration: number;
  insight?: string;
}

// ── Trend & Ranking ───────────────────────────────────────────────────────────

export interface TrendScore {
  entity_id: string;
  entity_type: EntityType;
  growth_velocity: number;
  github_activity: number;
  citation_acceleration: number;
  community_engagement: number;
  novelty_score: number;
  final_score: number;
  computed_at: string;
}

export interface TrendingItem {
  entity_id: string;
  entity_type: EntityType;
  title: string;
  description?: string;
  url?: string;
  trend_score: number;
  rank: number;
  delta_rank: number; // +/- from previous period
  growth_velocity: number;
  tags: string[];
  insight?: string;
}

// ── Forecast ─────────────────────────────────────────────────────────────────

export interface ForecastPoint {
  date: string;
  value: number;
  lower: number;
  upper: number;
}

export interface ForecastSeries {
  entity_id: string;
  entity_type: EntityType;
  title: string;
  historical: ForecastPoint[];
  forecast: ForecastPoint[];
  momentum: number;
  velocity: number;
  acceleration: number;
  breakout_detected: boolean;
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export interface DashboardSummary {
  total_papers: number;
  total_repos: number;
  total_topics: number;
  papers_today: number;
  repos_trending: number;
  top_category: string;
  avg_trend_score: number;
  breakout_topics: number;
  last_ingestion: string;
}

// ── Signal ────────────────────────────────────────────────────────────────────

export interface TrendSignal {
  date: string;
  value: number;
  signal_type: 'mentions' | 'stars' | 'citations' | 'engagement';
}

// ── Graph ─────────────────────────────────────────────────────────────────────

export interface GraphNode {
  id: number;
  name: string;
  size: number;
  keywords: string[];
  trend_score?: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  weight: number;
}

export interface TopicGraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

export type WSMessageType =
  | 'connected'
  | 'heartbeat'
  | 'trend_update'
  | 'new_paper'
  | 'score_update'
  | 'subscribed'
  | 'pong'
  | 'error';

export interface WSMessage {
  type: WSMessageType;
  timestamp: string;
  data?: any;
  message?: string;
}

// ── API Response ──────────────────────────────────────────────────────────────

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  has_next: boolean;
}

export interface ApiError {
  error: boolean;
  status_code: number;
  message: string;
  path: string;
}

// ── UI State ──────────────────────────────────────────────────────────────────

export type Timeframe = '24h' | '7d' | '30d';
export type SortField = 'trend_score' | 'stars' | 'published_date' | 'citation_count';
export type SortOrder = 'asc' | 'desc';
