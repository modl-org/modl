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
  kind?: string
  num_frames?: number
  fps?: number
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

export type RunSummary = {
  name: string
  status: string
  base_model?: string
  trigger_word?: string
  created_at?: string
  has_lora: boolean
  total_steps?: number
}

export type CheckpointInfo = {
  step: number
  path: string
  size_bytes: number
  promoted: boolean
}

export type TrainingRun = {
  name: string
  config?: Record<string, unknown>
  samples: SampleGroup[]
  lora_path?: string
  lora_size?: number
  lora_promoted?: boolean
  lineage?: TrainingLineage
  total_steps?: number
  sample_every?: number
  checkpoints: CheckpointInfo[]
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
  is_sampling?: boolean
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

export type LibraryLora = {
  id: string
  name: string
  trigger_word?: string
  base_model?: string
  lora_path: string
  thumbnail?: string
  step?: number
  training_run?: string
  config_json?: string
  tags?: string
  notes?: string
  size_bytes: number
  created_at: string
}

export type PromoteLoraRequest = {
  name: string
  trigger_word?: string
  base_model?: string
  lora_path: string
  thumbnail?: string
  step?: number
  training_run?: string
  config_json?: string
  tags?: string
}

export type LossPoint = {
  step: number
  loss: number
}

export type DatasetSummary = {
  name: string
  image_count: number
  captioned_count: number
  coverage: number
}

export type StartTrainingRequest = {
  dataset: string
  base_model: string
  name: string
  trigger_word: string
  lora_type: string
  preset?: string
  steps?: number
  rank?: number
  lr?: number
  optimizer?: string
  seed?: number
  class_word?: string
}

export type TrainingQueueItem = {
  id: number
  position: number
  name: string
  spec: Record<string, unknown>
  status: string
  created_at: string
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
  init_image?: string  // server-side path to init image for img2img
  mask?: string        // server-side path to mask image for inpainting
  strength?: number    // denoising strength (0.0-1.0)
  fast?: boolean       // use Lightning distillation LoRA
  num_frames?: number  // video frames (video models only)
  fps?: number         // video frame rate (video models only)
}

export type EditRequest = {
  prompt: string
  model_id: string
  images: string[] // server-side paths (uploaded via /api/upload)
  steps: number
  guidance: number
  seed?: number
  num_images: number
}

export type AnalysisResponse = {
  status: string
  output_path?: string
  error?: string
}

export type QueueJobSummary = {
  prompt: string
  model_id: string
  job_type: string
}

export type QueueStatus = {
  running: boolean
  queue_length: number
  current?: QueueJobSummary | null
  queue: QueueJobSummary[]
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

// ---------------------------------------------------------------------------
// Model family types (from /api/model-families)
// ---------------------------------------------------------------------------

export type ModelCapabilities = {
  txt2img: boolean
  img2img: boolean
  inpaint: boolean
  edit: boolean
  lora: boolean
  training: boolean
  txt2vid: boolean
  img2vid: boolean
}

export type ModelFamilyInfo = {
  id: string
  name: string
  arch_key: string
  transformer_b: number
  text_encoder_name: string
  text_encoder_b: number
  total_b: number
  vram_bf16_gb: number
  vram_fp8_gb: number
  capabilities: ModelCapabilities
  default_steps: number
  default_guidance: number
  default_resolution: number
  quality: number
  speed: number
  text_rendering: boolean
  default_frames: number
  default_fps: number
  description: string
}

export type ModelFamily = {
  id: string
  name: string
  vendor: string
  year: number
  models: ModelFamilyInfo[]
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
  modelFamilies: () => fetchJson<ModelFamily[]>('/api/model-families'),
  deleteModel: (id: string) =>
    fetchJson<{ deleted: string }>(`/api/models/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    }),
  runs: async (): Promise<RunSummary[]> => {
    const raw = await fetchJson<RunSummary[] | string[]>('/api/runs')
    // Handle old API format (plain string array) gracefully
    if (raw.length > 0 && typeof raw[0] === 'string') {
      return (raw as string[]).map((name) => ({
        name,
        status: 'unknown',
        has_lora: false,
      }))
    }
    return raw as RunSummary[]
  },
  run: (name: string) => fetchJson<TrainingRun>(`/api/runs/${encodeURIComponent(name)}`),
  status: () => fetchJson<TrainingStatusRun[]>('/api/status'),
  statusSingle: (name: string) =>
    fetchJson<TrainingStatusRun>(`/api/status/${encodeURIComponent(name)}`),
  datasets: async (): Promise<DatasetSummary[]> => {
    const raw = await fetchJson<DatasetSummary[] | string[]>('/api/datasets')
    // Handle old API format (plain string array) gracefully
    if (raw.length > 0 && typeof raw[0] === 'string') {
      return (raw as string[]).map((name) => ({
        name,
        image_count: 0,
        captioned_count: 0,
        coverage: 0,
      }))
    }
    return raw as DatasetSummary[]
  },
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
  batchDeleteOutputs: (items: Array<{ artifact_id?: string; path?: string }>) =>
    fetchJson<{ deleted_files: number; deleted_records: number; errors: string[] }>('/api/outputs/batch-delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(items),
    }),
  favoriteOutput: (path: string) =>
    fetchJson<{ favorited: boolean }>('/api/outputs/favorite', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    }),
  /** Upload a file (init image, mask) and return its server-side path. */
  upload: async (file: File): Promise<{ path: string }> => {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/api/upload', { method: 'POST', body: form })
    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: `HTTP ${res.status}` }))
      throw new Error(body.error ?? `Upload failed: HTTP ${res.status}`)
    }
    return res.json()
  },
  generate: (req: GenerateRequest) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  edit: (req: EditRequest) =>
    fetch('/api/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  upscale: (imagePath: string, scale = 4) =>
    fetchJson<AnalysisResponse>('/api/analysis/upscale', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath, scale }),
    }),
  removeBg: (imagePath: string) =>
    fetchJson<AnalysisResponse>('/api/analysis/remove-bg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_path: imagePath }),
    }),
  queueStatus: () => fetchJson<QueueStatus>('/api/generate/queue'),
  clearQueue: () =>
    fetchJson<{ cleared: number }>('/api/generate/queue', { method: 'DELETE' }),
  cancelQueueItem: (index: number) =>
    fetchJson<{ cancelled: boolean; queue_length?: number }>(`/api/generate/queue/${index}`, {
      method: 'DELETE',
    }),
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

  // Start new training run
  startTraining: (req: StartTrainingRequest) =>
    fetchJson<{ started: string }>('/api/runs/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),

  // Training queue
  trainingQueue: () => fetchJson<TrainingQueueItem[]>('/api/train/queue'),
  addToTrainingQueue: (req: StartTrainingRequest) =>
    fetchJson<{ id: number; name: string }>('/api/train/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  removeFromTrainingQueue: (id: number) =>
    fetch(`/api/train/queue/${id}`, { method: 'DELETE' }),
  reorderTrainingQueue: (id: number, position: number) =>
    fetch(`/api/train/queue/${id}/position`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ position }),
    }),

  // LoRA Library
  libraryLoras: () => fetchJson<LibraryLora[]>('/api/library/loras'),
  promoteLora: (req: PromoteLoraRequest) =>
    fetchJson<{ id: string }>('/api/library/loras', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  updateLibraryLora: (id: string, req: { name: string; tags?: string; notes?: string }) =>
    fetch(`/api/library/loras/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    }),
  deleteLibraryLora: (id: string) =>
    fetch(`/api/library/loras/${encodeURIComponent(id)}`, { method: 'DELETE' }),

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
