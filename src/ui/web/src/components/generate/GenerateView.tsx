import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  DownloadIcon,
  Maximize2Icon,
  Minimize2Icon,
} from 'lucide-react'
import { api, type GeneratedImage, type GeneratedOutput, type GenerateRequest, type GpuStatus, type InstalledModel } from '../../api'
import { useLocalStorage } from '../../hooks/useLocalStorage'
import { useSSE } from '../../hooks/useSSE'
import { CollapsibleSection } from '../ui/collapsible-section'
import { BatchPanel } from './BatchPanel'
import { GenerateActions } from './GenerateActions'
import { GenerateProgressBar, type GenerateProgressState } from './GenerateProgressBar'
import { GenerationGallery } from './GenerationGallery'
import { ImagePreview, type PreviewImage } from './ImagePreview'
import { LoraPanel } from './LoraPanel'
import { ModelPanel } from './ModelPanel'
import { PromptPanel } from './PromptPanel'
import { SamplingPanel } from './SamplingPanel'
import { SizePanel } from './SizePanel'
import { createDefaultGenerateFormState, type GenerateFormState } from './generate-state'

// ---------------------------------------------------------------------------
// GenerateView — pro sidebar layout with scrollable controls + fixed canvas
// ---------------------------------------------------------------------------

type Props = {
  /** Navigate to a different tab */
  setTab?: (tab: string) => void
}

export function GenerateView({ setTab: _setTab }: Props) {
  const queryClient = useQueryClient()

  // ── State ────────────────────────────────────────────────────────────
  const [form, setForm] = useLocalStorage<GenerateFormState>(
    'modl:generate-form-v2',
    createDefaultGenerateFormState,
  )
  const [progressState, setProgressState] = useState<GenerateProgressState>({ status: 'idle' })
  const [previewImages, setPreviewImages] = useState<PreviewImage[]>([])
  const expectedCountRef = useRef(1)
  const [canvasFit, setCanvasFit] = useState<'fit' | 'fill'>('fit')
  const [queueCount, setQueueCount] = useState(0)

  // ── Queries ──────────────────────────────────────────────────────────
  const { data: gpu = { training_active: false } as GpuStatus } = useQuery({
    queryKey: ['gpu'],
    queryFn: api.gpu,
    refetchInterval: 5000,
    staleTime: 4_000,
  })

  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60_000,
  })
  const models = modelsResponse?.models ?? []

  // Auto-select first checkpoint if none selected
  useEffect(() => {
    if (!form.base_model_id && models.length > 0) {
      const firstCheckpoint = models.find((m: InstalledModel) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')
      if (firstCheckpoint) {
        setForm((prev) => ({ ...prev, base_model_id: firstCheckpoint.id }))
      }
    }
  }, [models, form.base_model_id, setForm])

  const isGenerating = progressState.status === 'submitting' || progressState.status === 'streaming'

  // SSE connection — controlled by sseConnected state
  const [sseConnected, setSseConnected] = useState(false)

  const handleSSEMessage = useCallback(
    (message: string) => {
      // Skip initial running / running:queue:N pings
      if (message === 'running' || message.startsWith('running:queue:')) return

      const lower = message.toLowerCase()

      // Queue status events from server
      if (lower === 'queue:empty') {
        setQueueCount(0)
        return
      }
      if (lower.startsWith('queue:')) {
        const n = parseInt(message.slice(6), 10)
        if (Number.isFinite(n)) setQueueCount(n)
        return
      }

      // Check for error
      if (lower.startsWith('error:')) {
        const errMsg = message.slice(6).trim() || 'Generation failed.'
        console.error('[generate] server error:', errMsg)
        toast.error(errMsg)
        setProgressState({ status: 'error', message: errMsg })
        return
      }

      // Check for completion
      if (lower.includes('completed') || lower.includes('done')) {
        const count = expectedCountRef.current

        // Refresh outputs to find the new images
        void queryClient.invalidateQueries({ queryKey: ['outputs'] }).then(() => {
          const outputs = queryClient.getQueryData<GeneratedOutput[]>(['outputs']) ?? []
          const allImages: Array<{ url: string; seed?: number; modified: number }> = []
          for (const group of outputs) {
            for (const img of group.images) {
              allImages.push({ url: `/files/${img.path}`, seed: img.seed, modified: img.modified })
            }
          }
          allImages.sort((a, b) => b.modified - a.modified)
          setPreviewImages(allImages.slice(0, count))
        })

        toast.success(`Generated ${count} image${count !== 1 ? 's' : ''}`)
        setProgressState({ status: 'done', count, images: [] })
        return
      }

      // Update streaming log lines
      setProgressState((prev) => {
        if (prev.status !== 'streaming' && prev.status !== 'submitting') return prev
        const lines = prev.status === 'streaming' ? [...prev.lines.slice(-59), message] : [message]
        return { status: 'streaming', lines, step: prev.status === 'streaming' ? prev.step : undefined, totalSteps: prev.status === 'streaming' ? prev.totalSteps : undefined }
      })

      // Try to parse structured step progress
      try {
        const parsed = JSON.parse(message)
        if (parsed.step != null && parsed.total_steps != null) {
          setProgressState((prev) => {
            if (prev.status !== 'streaming') return prev
            return { ...prev, step: parsed.step, totalSteps: parsed.total_steps }
          })
        }
      } catch {
        // Not JSON — raw log line, already handled above
      }
    },
    [queryClient],
  )

  const handleSSEError = useCallback(() => {
    console.warn('[generate] SSE stream disconnected — will reconnect on next generate')
    setSseConnected(false)
    setProgressState((prev) =>
      prev.status === 'streaming' || prev.status === 'submitting'
        ? { status: 'error', message: 'Progress stream disconnected.' }
        : prev,
    )
  }, [])

  useSSE(sseConnected ? '/api/generate/stream' : null, handleSSEMessage, {
    onError: handleSSEError,
  })

  // ── Keyboard shortcut: Ctrl/Cmd + Enter ──────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        if (form.prompt.trim() && form.base_model_id) {
          handleGenerate()
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  })

  // ── Submit generation (supports enqueue) ─────────────────────────────
  const handleGenerate = useCallback(async () => {
    if (!form.prompt.trim() || !form.base_model_id) return

    // Build the request
    const req: GenerateRequest = {
      prompt: form.prompt,
      negative_prompt: form.negative_prompt.trim() || undefined,
      model_id: form.base_model_id,
      width: form.width,
      height: form.height,
      steps: form.steps,
      guidance: form.guidance,
      seed: form.seed,
      num_images: form.batch_count,
      loras: form.loras.map((l) => ({ id: l.id, strength: l.strength })),
    }

    // If not currently generating, this is a fresh start
    if (!isGenerating) {
      expectedCountRef.current = form.batch_count
      setPreviewImages([])
      setProgressState({ status: 'submitting' })
    }

    // Ensure SSE is connected
    setSseConnected(true)

    console.log('[generate] submitting:', req.model_id, req.prompt.slice(0, 60))

    try {
      const res = await api.generate(req)
      if (!res.ok) {
        let message = `HTTP ${res.status}`
        try {
          const body = await res.json()
          if (body.error) message = body.error
        } catch {
          const text = await res.text()
          if (text) message = text
        }
        throw new Error(message)
      }
      const body = await res.json()
      const queueLen = body.queue_length ?? 0
      setQueueCount(queueLen)

      if (queueLen > 0) {
        toast.info(`Enqueued — position ${queueLen}`)
      } else if (!isGenerating) {
        setProgressState({ status: 'streaming', lines: [] })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      console.error('[generate] submit failed:', message)
      toast.error(message)
      if (!isGenerating) {
        setProgressState({ status: 'error', message })
      }
    }
  }, [form, isGenerating])

  // ── Gallery click → load params ──────────────────────────────────────
  const handleGallerySelect = useCallback(
    (img: GeneratedImage) => {
      // Show image in preview
      setPreviewImages([{ url: `/files/${img.path}`, seed: img.seed }])

      // Optionally load metadata back into form
      if (img.prompt) setForm((prev) => ({ ...prev, prompt: img.prompt! }))
      if (img.seed != null) setForm((prev) => ({ ...prev, seed: img.seed! }))
      if (img.steps != null) setForm((prev) => ({ ...prev, steps: img.steps! }))
      if (img.guidance != null) setForm((prev) => ({ ...prev, guidance: img.guidance! }))
      if (img.width != null && img.height != null) {
        setForm((prev) => ({ ...prev, width: img.width!, height: img.height! }))
      }
      if (img.base_model_id) setForm((prev) => ({ ...prev, base_model_id: img.base_model_id! }))
    },
    [setForm],
  )

  // Count of checkpoints for warnings
  const checkpointCount = useMemo(
    () => models.filter((m: InstalledModel) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model').length,
    [models],
  )

  // Download current preview image
  const downloadImage = useCallback(() => {
    const img = previewImages[0]
    if (!img) return
    const a = document.createElement('a')
    a.href = img.url
    a.download = img.url.split('/').pop() ?? 'image.png'
    a.click()
  }, [previewImages])

  const activePreviewImage = previewImages[0]

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <div className="flex h-full">
      {/* ──────────────── Control Panel (Left Sidebar) ──────────────── */}
      <div className="flex w-[340px] shrink-0 flex-col border-r border-border/40 lg:w-[360px]">
        {/* Scrollable controls */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          {/* Warning */}
          {checkpointCount === 0 && (
            <div className="mb-4 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              No checkpoints installed. Run{' '}
              <code className="rounded bg-secondary px-1 py-0.5 font-mono text-[10px]">
                modl pull sdxl
              </code>{' '}
              to get started.
            </div>
          )}

          {/* ─── Prompt (always at top) ─── */}
          <PromptPanel
            form={form}
            setForm={setForm}
            modelHint={models.find((m) => m.id === form.base_model_id)?.name}
          />

          {/* ─── Model & LoRA ─── */}
          <CollapsibleSection title="Model">
            <div className="space-y-3">
              <ModelPanel models={models} form={form} setForm={setForm} />
              <LoraPanel models={models} form={form} setForm={setForm} />
            </div>
          </CollapsibleSection>

          {/* ─── Dimensions ─── */}
          <CollapsibleSection title="Dimensions">
            <div className="space-y-3">
              <SizePanel form={form} setForm={setForm} />
              <BatchPanel form={form} setForm={setForm} />
            </div>
          </CollapsibleSection>

          {/* ─── Sampling ─── */}
          <CollapsibleSection title="Sampling" defaultOpen={false}>
            <SamplingPanel form={form} setForm={setForm} />
          </CollapsibleSection>
        </div>

        {/* ─── Sticky Generate Button (bottom of sidebar) ─── */}
        <div className="shrink-0 border-t border-border/40 bg-[#0e0e18]/95 px-4 py-3 backdrop-blur">
          <GenerateActions
            form={form}
            gpu={gpu}
            isGenerating={isGenerating}
            queueCount={queueCount}
            onGenerate={handleGenerate}
            onInterrupt={() => {
              setProgressState({ status: 'idle' })
            }}
            onClearQueue={async () => {
              await api.clearQueue()
              setQueueCount(0)
              toast.info('Queue cleared')
            }}
          />
          <GenerateProgressBar state={progressState} />
          <p className="mt-1 text-center text-[10px] text-muted-foreground/30">
            <kbd className="rounded border border-border/40 bg-secondary/20 px-1 py-0.5 font-mono text-[9px]">
              Ctrl+Enter
            </kbd>
          </p>
        </div>
      </div>

      {/* ──────────────── Canvas (Right Area) ──────────────── */}
      <div className="relative flex min-w-0 flex-1 flex-col bg-[#080810]">
        {/* Main canvas */}
        <div className="relative flex flex-1 items-center justify-center overflow-hidden p-6">
          {/* Image preview */}
          <ImagePreview
            images={previewImages}
            isGenerating={isGenerating}
            expectedCount={form.batch_count}
            width={form.width}
            height={form.height}
            fitMode={canvasFit}
            onImageClick={(img) => window.open(img.url, '_blank')}
          />

          {/* Floating toolbar — visible when an image is shown */}
          {activePreviewImage && !isGenerating && (
            <div className="absolute right-4 top-4 flex items-center gap-1 rounded-lg border border-border/40 bg-background/80 px-1.5 py-1 shadow-lg backdrop-blur">
              <button
                type="button"
                onClick={() => setCanvasFit((f) => (f === 'fit' ? 'fill' : 'fit'))}
                className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                title={canvasFit === 'fit' ? 'View at 100%' : 'Fit to screen'}
              >
                {canvasFit === 'fit' ? (
                  <Maximize2Icon className="size-3.5" />
                ) : (
                  <Minimize2Icon className="size-3.5" />
                )}
              </button>
              <button
                type="button"
                onClick={downloadImage}
                className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                title="Download"
              >
                <DownloadIcon className="size-3.5" />
              </button>
            </div>
          )}
        </div>

        {/* History filmstrip (bottom of canvas) */}
        <div className="shrink-0 border-t border-border/30 bg-[#0a0a14]/90 px-4 py-2 backdrop-blur">
          <GenerationGallery
            onSelect={handleGallerySelect}
            activePath={previewImages[0]?.url?.replace('/files/', '') ?? null}
          />
        </div>
      </div>
    </div>
  )
}
