export type GpuStatus = {
  name?: string
  vram_total_mb?: number
  vram_free_mb?: number
  training_active: boolean
}

export type DependencyRef = {
  id: string
  type: string
  installed: boolean
}

export type FeatureDep = {
  feature: string
  description: string
  model_type: string
  installed: boolean
  install_hint?: string
}

export type InstalledModel = {
  id: string
  name: string
  model_type: string
  variant?: string
  size_bytes: number
  trigger_word?: string
  base_model_id?: string
  sample_image_url?: string
  depended_on_by?: string[]
  depends_on?: DependencyRef[]
}

export type ModelsResponse = {
  models: InstalledModel[]
  total_size_bytes: number
  feature_deps: FeatureDep[]
}

export type GeneratedImage = {
  path: string
  filename: string
  modified: number
  artifact_id?: string
  job_id?: string
  prompt?: string
  base_model_id?: string
  lora_name?: string
  lora_strength?: number
  seed?: number
  steps?: number
  guidance?: number
  width?: number
  height?: number
  size_bytes?: number
  generated_with?: string
  favorited?: boolean
}

export type GeneratedOutput = {
  date: string
  images: GeneratedImage[]
}

export type JobSummary = {
  job_id: string
  status: string
  steps?: number
  created_at: string
  resumed_from?: string
}

export type TrainingLineage = {
  dataset_name?: string
  dataset_image_count?: number
  base_model?: string
  jobs: JobSummary[]
}

export type SampleGroup = {
  step: number
  images: string[]
}

export type TrainingRun = {
  name: string
  config?: Record<string, unknown>
  samples: SampleGroup[]
  lora_path?: string
  lora_size?: number
  lineage?: TrainingLineage
  total_steps?: number
  sample_every?: number
}

export type TrainingStatusRun = {
  name: string
  is_running: boolean
  current_step?: number
  total_steps?: number
  percent?: number
  speed?: number
  elapsed?: string
  eta?: string
  loss?: number
  lr?: string
  arch?: string
  trigger_word?: string
  latest_checkpoint?: string
}

export type DatasetImage = {
  filename: string
  caption?: string
  image_url: string
}

export type DatasetOverview = {
  name: string
  image_count: number
  captioned_count: number
  coverage: number
  images: DatasetImage[]
}

export type SearchResult = {
  id: string
  name: string
  model_type: string
  author?: string
  description?: string
  size_bytes: number
  variants: { id: string; size_bytes: number; precision?: string }[]
  installed: boolean
  requires_auth: boolean
}

export type LossPoint = {
  step: number
  loss: number
}

export type DeleteOutputRequest = {
  artifact_id?: string
  path?: string
}

export type DeleteOutputResponse = {
  deleted_file: boolean
  deleted_records: number
}

export type GenerateRequest = {
  prompt: string
  negative_prompt?: string
  model_id: string
  width: number
  height: number
  steps: number
  guidance: number
  seed?: number
  num_images: number
  loras: Array<{ id: string; strength: number }>
}

export type EnhanceRequest = {
  prompt: string
  model_hint?: string
  intensity?: 'subtle' | 'moderate' | 'aggressive'
}

export type EnhanceResponse = {
  original: string
  enhanced: string
  backend: string
}

// ---------------------------------------------------------------------------
// Studio types
// ---------------------------------------------------------------------------

export type StudioSession = {
  id: string
  intent: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  input_images: string[]
  output_images: string[]
  events: AgentEvent[]
  created_at: string
  completed_at?: string
}

export type AgentEvent = {
  type: 'thinking' | 'tool_start' | 'tool_progress' | 'tool_complete' | 'output_ready' | 'error'
  message?: string
  tool?: string
  description?: string
  progress?: number
  detail?: string
  result?: string
  images?: string[]
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `HTTP ${res.status}`)
  }
  return (await res.json()) as T
}

export const api = {
  gpu: () => fetchJson<GpuStatus>('/api/gpu'),
  models: async (): Promise<ModelsResponse> => {
    const raw = await fetchJson<ModelsResponse | InstalledModel[]>('/api/models')
    // Handle old API format (plain array) gracefully
    if (Array.isArray(raw)) {
      return { models: raw, total_size_bytes: raw.reduce((s, m) => s + (m.size_bytes ?? 0), 0), feature_deps: [] }
    }
    return raw
  },
  deleteModel: (id: string) =>
    fetchJson<{ deleted: string }>(`/api/models/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  runs: () => fetchJson<string[]>('/api/runs'),
  run: (name: string) => fetchJson<TrainingRun>(`/api/runs/${encodeURIComponent(name)}`),
  status: () => fetchJson<TrainingStatusRun[]>('/api/status'),
  statusSingle: (name: string) =>
    fetchJson<TrainingStatusRun>(`/api/status/${encodeURIComponent(name)}`),
  datasets: () => fetchJson<string[]>('/api/datasets'),
  dataset: (name: string, limit = 50, offset = 0) =>
    fetchJson<DatasetOverview>(
      `/api/datasets/${encodeURIComponent(name)}?limit=${limit}&offset=${offset}`,
    ),
  outputs: () => fetchJson<GeneratedOutput[]>('/api/outputs'),
  deleteOutput: (req: DeleteOutputRequest) =>
    fetchJson<DeleteOutputResponse>('/api/outputs', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  favoriteOutput: (path: string) =>
    fetchJson<{ favorited: boolean }>('/api/outputs/favorite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),
  generate: (req: GenerateRequest) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  queueStatus: () => fetchJson<{ running: boolean; queue_length: number }>('/api/generate/queue'),
  clearQueue: () =>
    fetchJson<{ cleared: number }>('/api/generate/queue', { method: 'DELETE' }),
  enhance: (req: EnhanceRequest) =>
    fetchJson<EnhanceResponse>('/api/enhance', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),

  // Registry search & install
  searchRegistry: (q: string, type?: string) => {
    const params = new URLSearchParams({ q })
    if (type) params.set('type', type)
    return fetchJson<SearchResult[]>(`/api/registry/search?${params}`)
  },
  installModel: (id: string, variant?: string) =>
    fetchJson<{ installed: string[] }>('/api/models/install', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id, variant }),
    }),

  // Loss history
  lossHistory: (name: string) =>
    fetchJson<LossPoint[]>(`/api/runs/${encodeURIComponent(name)}/loss`),

  // Resume training
  resumeTraining: (name: string, checkpoint: string) =>
    fetchJson<{ started: string }>('/api/runs/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, checkpoint }),
    }),

  // Cancel training
  cancelTraining: (name: string) =>
    fetchJson<{ cancelled: boolean; pids_killed: number }>('/api/runs/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),

  // Delete training run
  deleteRun: (name: string) =>
    fetchJson<{ deleted: boolean }>('/api/runs/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    }),

  // Studio
  studioSessions: () => fetchJson<StudioSession[]>('/api/studio/sessions'),
  studioSession: (id: string) =>
    fetchJson<StudioSession>(`/api/studio/sessions/${encodeURIComponent(id)}`),
  studioCreateSession: (intent: string) =>
    fetchJson<{ id: string; status: string }>('/api/studio/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intent }),
    }),
  studioUploadImages: (sessionId: string, files: File[]) => {
    const formData = new FormData()
    for (const file of files) {
      formData.append('images', file, file.name)
    }
    return fetchJson<{ uploaded: number; images: string[] }>(
      `/api/studio/sessions/${encodeURIComponent(sessionId)}/images`,
      { method: 'POST', body: formData },
    )
  },
  studioStart: (sessionId: string) =>
    fetchJson<{ status: string; session_id: string }>(
      `/api/studio/sessions/${encodeURIComponent(sessionId)}/start`,
      { method: 'POST' },
    ),
  studioDelete: (sessionId: string) =>
    fetchJson<{ deleted: boolean }>(
      `/api/studio/sessions/${encodeURIComponent(sessionId)}`,
      { method: 'DELETE' },
    ),
}
