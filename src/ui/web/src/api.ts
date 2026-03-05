export type GpuStatus = {
  name?: string
  vram_total_mb?: number
  vram_free_mb?: number
  training_active: boolean
}

export type InstalledModel = {
  id: string
  name: string
  model_type: 'checkpoint' | 'lora'
  variant?: string
  size_bytes: number
  trigger_word?: string
  base_model_id?: string
  sample_image_url?: string
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
  models: () => fetchJson<InstalledModel[]>('/api/models'),
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
}
