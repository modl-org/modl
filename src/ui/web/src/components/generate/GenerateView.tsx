import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { Separator } from '@/components/ui/separator'
import { api, type GeneratedImage, type GenerateRequest, type GpuStatus, type InstalledModel } from '../../api'
import { useLocalStorage } from '../../hooks/useLocalStorage'
import { BatchPanel } from './BatchPanel'
import { GenerateActions } from './GenerateActions'
import { GenerateProgressBar, type GenerateProgressState } from './GenerateProgressBar'
import { GenerationGallery } from './GenerationGallery'
import { ImagePreview, type PreviewImage } from './ImagePreview'
import { Img2ImgPanel } from './Img2ImgPanel'
import { LoraPanel } from './LoraPanel'
import { ModelPanel } from './ModelPanel'
import { PromptPanel } from './PromptPanel'
import { SamplingPanel } from './SamplingPanel'
import { SizePanel } from './SizePanel'
import { createDefaultGenerateFormState, type GenerateFormState } from './generate-state'

// ---------------------------------------------------------------------------
// GenerateView — A1111-inspired generate interface
//
// Layout:
//   ┌─────────────────────────────┬──────────────────┐
//   │  ImagePreview (center)      │  Controls (right) │
//   │                             │  ┌──────────────┐ │
//   │                             │  │ Model        │ │
//   │                             │  │ LoRA         │ │
//   │                             │  │ Size         │ │
//   │                             │  │ Sampling     │ │
//   │                             │  │ Batch        │ │
//   │                             │  │ Img2Img      │ │
//   │                             │  └──────────────┘ │
//   ├─────────────────────────────┼──────────────────┤
//   │  Prompt area                │  [Generate]       │
//   ├─────────────────────────────┴──────────────────┤
//   │  Progress bar                                   │
//   ├────────────────────────────────────────────────┤
//   │  Recent gallery strip                           │
//   └────────────────────────────────────────────────┘
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

  // ── Queries ──────────────────────────────────────────────────────────
  const { data: gpu = { training_active: false } as GpuStatus } = useQuery({
    queryKey: ['gpu'],
    queryFn: api.gpu,
    refetchInterval: 5000,
    staleTime: 4_000,
  })

  const { data: models = [] } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60_000,
  })

  // Auto-select first checkpoint if none selected
  useEffect(() => {
    if (!form.base_model_id && models.length > 0) {
      const firstCheckpoint = models.find((m: InstalledModel) => m.model_type === 'checkpoint')
      if (firstCheckpoint) {
        setForm((prev) => ({ ...prev, base_model_id: firstCheckpoint.id }))
      }
    }
  }, [models, form.base_model_id, setForm])

  const isGenerating = progressState.status === 'submitting' || progressState.status === 'streaming'

  // ── Submit generation ────────────────────────────────────────────────
  const handleGenerate = useCallback(async () => {
    if (isGenerating) return
    if (!form.prompt.trim() || !form.base_model_id) return

    expectedCountRef.current = form.batch_count
    setProgressState({ status: 'submitting' })
    setPreviewImages([])

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

    try {
      const res = await api.generate(req)
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `HTTP ${res.status}`)
      }
      setProgressState({ status: 'streaming', lines: [] })
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      setProgressState({ status: 'error', message })
    }
  }, [form, isGenerating])

  // ── SSE event stream ─────────────────────────────────────────────────
  useEffect(() => {
    if (progressState.status !== 'streaming') return

    let done = false
    const finish = (nextState: GenerateProgressState) => {
      if (done) return
      done = true
      setProgressState(nextState)
    }

    const eventSource = new EventSource('/api/generate/stream')

    eventSource.onmessage = (event) => {
      const message: string = event.data

      setProgressState((prev) => {
        if (prev.status !== 'streaming') return prev
        return {
          status: 'streaming',
          lines: [...prev.lines.slice(-59), message],
          step: prev.step,
          totalSteps: prev.totalSteps,
        }
      })

      const lower = message.toLowerCase()

      // Check for error
      if (lower.startsWith('error:')) {
        eventSource.close()
        finish({ status: 'error', message: message.slice(6).trim() || 'Generation failed.' })
        return
      }

      // Check for completion
      if (lower.includes('completed') || lower.includes('done')) {
        const count = expectedCountRef.current
        eventSource.close()

        // Refresh outputs to find the new images
        void queryClient.invalidateQueries({ queryKey: ['outputs'] }).then(() => {
          // Fetch latest outputs to populate preview
          api.outputs().then((outputs) => {
            const allImages: PreviewImage[] = []
            for (const group of outputs) {
              for (const img of group.images) {
                allImages.push({ url: `/files/${img.path}`, seed: img.seed })
              }
            }
            allImages.sort((a, b) => {
              // Newest first — use url as proxy since they contain timestamps
              return b.url.localeCompare(a.url)
            })
            setPreviewImages(allImages.slice(0, count))
          }).catch(() => {
            // Silently handle — images display on next refresh
          })
        })

        finish({ status: 'done', count, images: [] })
        toast.success(`Generated ${count} image${count !== 1 ? 's' : ''}`)
      }

      // Try to parse structured progress (step/total from JSON events)
      try {
        const parsed = JSON.parse(message)
        if (parsed.step != null && parsed.total_steps != null) {
          setProgressState((prev) => {
            if (prev.status !== 'streaming') return prev
            return { ...prev, step: parsed.step, totalSteps: parsed.total_steps }
          })
        }
      } catch {
        // Not JSON — that's fine, raw log line
      }
    }

    eventSource.onerror = () => {
      finish({ status: 'error', message: 'Progress stream disconnected.' })
      eventSource.close()
    }

    return () => {
      eventSource.close()
    }
  }, [progressState.status, queryClient])

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
    () => models.filter((m: InstalledModel) => m.model_type === 'checkpoint').length,
    [models],
  )

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <div className="flex h-full flex-col">
      {/* Main content area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="mx-auto max-w-6xl space-y-4">
          {/* Warnings */}
          {checkpointCount === 0 && (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
              No checkpoint models installed. Run{' '}
              <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">
                modl pull flux-schnell
              </code>{' '}
              or{' '}
              <code className="rounded bg-secondary px-1.5 py-0.5 font-mono text-xs">
                modl pull sdxl
              </code>{' '}
              to get started.
            </div>
          )}

          {/* ─── Two-column layout: Preview + Controls ─── */}
          <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
            {/* Left: Preview + Prompt + Actions */}
            <div className="flex flex-col gap-4">
              {/* Image preview */}
              <ImagePreview
                images={previewImages}
                isGenerating={isGenerating}
                expectedCount={form.batch_count}
                width={form.width}
                height={form.height}
                onImageClick={(img) => {
                  // Could open lightbox — for now just log
                  window.open(img.url, '_blank')
                }}
              />

              {/* Progress bar */}
              <GenerateProgressBar state={progressState} />

              {/* Prompt */}
              <PromptPanel
                form={form}
                setForm={setForm}
                modelHint={models.find((m) => m.id === form.base_model_id)?.name}
              />

              {/* Generate button */}
              <GenerateActions
                form={form}
                gpu={gpu}
                isGenerating={isGenerating}
                onGenerate={handleGenerate}
                onInterrupt={() => {
                  // TODO: wire up cancel API
                  setProgressState({ status: 'idle' })
                }}
              />
            </div>

            {/* Right: Controls sidebar */}
            <aside className="space-y-1 overflow-y-auto lg:max-h-[calc(100vh-8rem)]">
              <div className="space-y-4 rounded-lg border border-border/40 bg-card/30 p-3">
                <ModelPanel models={models} form={form} setForm={setForm} />
                <Separator className="opacity-30" />
                <LoraPanel models={models} form={form} setForm={setForm} />
                <Separator className="opacity-30" />
                <SizePanel form={form} setForm={setForm} />
                <Separator className="opacity-30" />
                <SamplingPanel form={form} setForm={setForm} />
                <Separator className="opacity-30" />
                <BatchPanel form={form} setForm={setForm} />
                <Separator className="opacity-30" />
                <Img2ImgPanel form={form} setForm={setForm} />
              </div>
            </aside>
          </div>

          {/* ─── Bottom gallery strip ─── */}
          <GenerationGallery
            onSelect={handleGallerySelect}
            activePath={previewImages[0]?.url?.replace('/files/', '') ?? null}
          />
        </div>
      </div>
    </div>
  )
}
