import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import {
  DownloadIcon,
  PencilIcon,
  SparklesIcon,
} from 'lucide-react'
import { api, type EditRequest, type GeneratedImage, type GeneratedOutput, type GenerateRequest, type GpuStatus, type InstalledModel, type ModelFamily } from '../../api'
import { useGpuStatus } from '../../hooks/useGpuStatus'
import { useSSE } from '../../hooks/useSSE'
import { CollapsibleSection } from '../ui/collapsible-section'
import { BatchPanel } from './BatchPanel'
import { GenerateActions } from './GenerateActions'
import { GenerateProgressBar, type GenerateProgressState } from './GenerateProgressBar'
import { ImagePreview, type PreviewImage, type GeneratingContext } from './ImagePreview'
import { SessionStrip, type SessionItem } from './SessionStrip'
import { Img2ImgPanel } from './Img2ImgPanel'
import { LoraPanel } from './LoraPanel'
import { ModelPanel } from './ModelPanel'
import { PromptPanel } from './PromptPanel'
import { SamplingPanel } from './SamplingPanel'
import { SizePanel } from './SizePanel'
import { EditImagesPanel } from './EditImagesPanel'
import { findModelFamily, modelDefaults, randomSeed, type GenerateFormState, type GenerationMode } from './generate-state'

// ---------------------------------------------------------------------------
// GenerateView — pro sidebar layout with scrollable controls + fixed canvas
// ---------------------------------------------------------------------------

type Props = {
  /** Shared form state from App */
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  /** Navigate to a different tab */
  setTab?: (tab: string) => void
}

export function GenerateView({ form, setForm, setTab: _setTab }: Props) {
  const queryClient = useQueryClient()

  // ── State ────────────────────────────────────────────────────────────
  const [progressState, setProgressState] = useState<GenerateProgressState>({ status: 'idle' })
  const [previewImages, setPreviewImages] = useState<PreviewImage[]>([])
  const expectedCountRef = useRef(1)
  const [, setQueueCount] = useState(0)

  // ── Session strip state ─────────────────────────────────────────────
  const [sessionItems, setSessionItems] = useState<SessionItem[]>([])
  // Whether user manually clicked a session card (suppresses auto-advance)
  const userFocusedRef = useRef(false)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const sessionIdCounter = useRef(0)

  // ── Refs for stable callbacks (fix re-render cascade) ─────────────
  const formRef = useRef(form)
  formRef.current = form
  const sessionItemsRef = useRef(sessionItems)
  sessionItemsRef.current = sessionItems

  const isGenerating = progressState.status === 'submitting' || progressState.status === 'streaming'
  const isGeneratingRef = useRef(isGenerating)
  isGeneratingRef.current = isGenerating

  // ── Queries ──────────────────────────────────────────────────────────
  const { data: gpu = { training_active: false } as GpuStatus } = useGpuStatus()

  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 5 * 60_000,
  })
  const models = modelsResponse?.models ?? []

  const { data: families = [] } = useQuery<ModelFamily[]>({
    queryKey: ['model-families'],
    queryFn: api.modelFamilies,
    staleTime: 60 * 60_000, // static data, rarely changes
  })

  // Auto-select first checkpoint if none selected
  useEffect(() => {
    if (!form.base_model_id && models.length > 0) {
      const firstCheckpoint = models.find((m: InstalledModel) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')
      if (firstCheckpoint) {
        setForm((prev) => ({ ...prev, base_model_id: firstCheckpoint.id }))
      }
    }
  }, [models, form.base_model_id, setForm])

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
        // Mark active session item as error
        setSessionItems((prev) => {
          const idx = prev.findIndex((s) => s.status === 'active')
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = { ...updated[idx], status: 'error', error: errMsg }
          // Promote next queued → active
          const nextQueued = updated.findIndex((s) => s.status === 'queued')
          if (nextQueued !== -1) {
            updated[nextQueued] = { ...updated[nextQueued], status: 'active' }
          } else {
            // No more queued items — close SSE
            setSseConnected(false)
          }
          return updated
        })
        return
      }

      // Check for completion
      if (lower.includes('completed') || lower.includes('done')) {
        const count = expectedCountRef.current

        // Refresh outputs to find the new images
        void queryClient.invalidateQueries({ queryKey: ['outputs'] }).then(() => {
          const outputs = queryClient.getQueryData<GeneratedOutput[]>(['outputs']) ?? []
          const allImages: Array<{ url: string; seed?: number; modified: number; path?: string }> = []
          for (const group of outputs) {
            for (const img of group.images) {
              allImages.push({ url: `/files/${img.path}`, seed: img.seed, modified: img.modified, path: img.path })
            }
          }
          allImages.sort((a, b) => b.modified - a.modified)
          const completedImages = allImages.slice(0, count)

          // Auto-advance canvas (unless user is reviewing an older card)
          if (!userFocusedRef.current) {
            setPreviewImages(completedImages)
          }

          // Update the active session item → completed
          setSessionItems((prev) => {
            const idx = prev.findIndex((s) => s.status === 'active')
            if (idx === -1) return prev
            const updated = [...prev]
            updated[idx] = {
              ...updated[idx],
              status: 'completed',
              images: completedImages.map((ci) => ({
                url: ci.url,
                seed: ci.seed,
                path: ci.path,
              })),
              step: undefined,
              totalSteps: undefined,
            }
            if (!userFocusedRef.current) {
              setActiveSessionId(updated[idx].id)
            }
            // Promote next queued → active
            const nextQueued = updated.findIndex((s) => s.status === 'queued')
            if (nextQueued !== -1) {
              updated[nextQueued] = { ...updated[nextQueued], status: 'active' }
            } else {
              // No more queued items — close SSE
              setSseConnected(false)
            }
            return updated
          })
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
          // Update active session item progress
          setSessionItems((prev) => {
            const idx = prev.findIndex((s) => s.status === 'active')
            if (idx === -1) return prev
            const updated = [...prev]
            updated[idx] = { ...updated[idx], step: parsed.step, totalSteps: parsed.total_steps }
            return updated
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

  // ── Mode switch ────────────────────────────────────────────────────
  // Remember the model used before entering edit mode so we can restore it
  const preEditModelRef = useRef<string | null>(null)

  const handleModeSwitch = useCallback((mode: GenerationMode) => {
    setForm((prev) => {
      const updates: Partial<GenerateFormState> = { mode }
      if (mode === 'edit') {
        // Save current model before switching to edit
        preEditModelRef.current = prev.base_model_id
        // Auto-select qwen-image-edit if available + apply its defaults
        const editModel = models.find(
          (m) => m.id === 'qwen-image-edit' || m.name.toLowerCase().includes('qwen-image-edit'),
        )
        if (editModel) {
          updates.base_model_id = editModel.id
          const info = findModelFamily(editModel.name, families)
          const defaults = modelDefaults(editModel.name, info)
          updates.steps = defaults.steps
          updates.guidance = defaults.guidance
        }
      } else {
        // Switching back to generate — restore previous model if current is edit-only
        const currentInfo = findModelFamily(prev.base_model_id, families)
        const currentSupportsTxt2img = !currentInfo || currentInfo.capabilities.txt2img
        if (!currentSupportsTxt2img) {
          // Try to restore the model used before entering edit
          const savedId = preEditModelRef.current
          const savedModel = savedId ? models.find((m) => m.id === savedId) : null
          if (savedModel) {
            updates.base_model_id = savedModel.id
            const info = findModelFamily(savedModel.name, families)
            const defaults = modelDefaults(savedModel.name, info)
            updates.steps = defaults.steps
            updates.guidance = defaults.guidance
          } else {
            // Fall back to first installed checkpoint that supports txt2img
            const fallback = models.find((m) => {
              if (m.model_type !== 'checkpoint' && m.model_type !== 'diffusion_model') return false
              const info = findModelFamily(m.name, families)
              return !info || info.capabilities.txt2img
            })
            if (fallback) {
              updates.base_model_id = fallback.id
              const info = findModelFamily(fallback.name, families)
              const defaults = modelDefaults(fallback.name, info)
              updates.steps = defaults.steps
              updates.guidance = defaults.guidance
            }
          }
        }
        // Clear edit images when leaving edit mode
        updates.edit_images = []
      }
      return { ...prev, ...updates }
    })
  }, [models, families, setForm])

  // Top-level mode for the view (generate vs edit)
  const isEditMode = form.mode === 'edit'

  // ── Submit job helper (shared by generate + edit) ──────────────────
  type SubmitJobParams = {
    prefix: string
    jobType: 'generate' | 'edit'
    /** Build the request. Receives the resolved seed. Return null to abort. */
    buildRequest: (currentForm: GenerateFormState, seed: number) => Promise<Record<string, unknown> | null>
    /** API call to submit */
    submit: (req: never) => Promise<Response>
  }

  const submitJob = useCallback(async ({ prefix, jobType, buildRequest, submit }: SubmitJobParams) => {
    const currentForm = formRef.current
    if (!currentForm.prompt.trim() || !currentForm.base_model_id) return

    // Randomize seed if not locked
    let seed = currentForm.seed
    if (!currentForm.seed_locked) {
      seed = randomSeed()
      setForm((prev) => ({ ...prev, seed }))
    }

    // Build request (may abort by returning null)
    const req = await buildRequest(currentForm, seed)
    if (!req) return

    // Session management
    const sessionId = `${prefix}-${++sessionIdCounter.current}-${Date.now()}`
    const isFirst = !isGeneratingRef.current

    if (isFirst) {
      expectedCountRef.current = currentForm.batch_count
      setPreviewImages([])
      setProgressState({ status: 'submitting' })
    }

    const newItem: SessionItem = {
      id: sessionId,
      status: isFirst ? 'active' : 'queued',
      prompt: currentForm.prompt,
      model_id: currentForm.base_model_id,
      job_type: jobType,
      batch_count: currentForm.batch_count,
    }
    setSessionItems((prev) => [...prev, newItem])
    userFocusedRef.current = false

    // Ensure SSE is connected
    setSseConnected(true)

    console.log(`[${jobType}] submitting:`, currentForm.base_model_id, currentForm.prompt.slice(0, 60))

    try {
      const res = await submit(req as never)
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
      } else if (isFirst) {
        setProgressState({ status: 'streaming', lines: [] })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      console.error(`[${jobType}] submit failed:`, message)
      toast.error(message)
      setSessionItems((prev) =>
        prev.map((s) => (s.id === sessionId ? { ...s, status: 'error' as const, error: message } : s)),
      )
      if (isFirst) {
        setProgressState({ status: 'error', message })
      }
    }
  }, [setForm])

  // ── Submit generation (supports enqueue) ─────────────────────────────
  const handleGenerate = useCallback(() => submitJob({
    prefix: 'gen',
    jobType: 'generate',
    buildRequest: async (f, seed) => {
      // Upload init image if in img2img mode
      let initImagePath: string | undefined
      if (f.mode === 'img2img' && f.init_image_file) {
        if (!(f.init_image_file instanceof File)) {
          toast.error('Image reference expired — please re-add the image')
          return null
        }
        try {
          const uploaded = await api.upload(f.init_image_file)
          initImagePath = uploaded.path
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err)
          toast.error(`Image upload failed: ${message}`)
          return null
        }
      }

      return {
        prompt: f.prompt,
        negative_prompt: f.negative_prompt.trim() || undefined,
        model_id: f.base_model_id,
        width: f.width,
        height: f.height,
        steps: f.steps,
        guidance: f.guidance,
        seed,
        num_images: f.batch_count,
        loras: f.loras.filter((l) => l.enabled !== false).map((l) => ({ id: l.id, strength: l.strength })),
        init_image: initImagePath,
        strength: initImagePath ? f.denoise_strength : undefined,
        fast: f.fast || undefined,
      } satisfies GenerateRequest
    },
    submit: api.generate,
  }), [submitJob])

  // ── Submit edit ────────────────────────────────────────────────────
  const handleEdit = useCallback(() => submitJob({
    prefix: 'edit',
    jobType: 'edit',
    buildRequest: async (f, seed) => {
      const editImages = f.edit_images ?? []
      if (editImages.length === 0) return null

      // Resolve all edit images to server-side paths
      const uploadedPaths: string[] = []
      try {
        for (const img of editImages) {
          if (img.type === 'server') {
            uploadedPaths.push(img.serverPath)
          } else {
            if (!(img.file instanceof File)) {
              toast.error('Image reference expired — please re-add the image')
              return null
            }
            const uploaded = await api.upload(img.file)
            uploadedPaths.push(uploaded.path)
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        toast.error(`Image upload failed: ${message}`)
        return null
      }

      return {
        prompt: f.prompt,
        model_id: f.base_model_id,
        images: uploadedPaths,
        steps: f.steps,
        guidance: f.guidance,
        seed,
        num_images: f.batch_count,
      } satisfies EditRequest
    },
    submit: api.edit,
  }), [submitJob])

  // ── Keyboard shortcut: Ctrl/Cmd + Enter ──────────────────────────────
  const handleGenerateRef = useRef(handleGenerate)
  handleGenerateRef.current = handleGenerate
  const handleEditRef = useRef(handleEdit)
  handleEditRef.current = handleEdit

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        const f = formRef.current
        if (f.prompt.trim() && f.base_model_id) {
          if (f.mode === 'edit') {
            handleEditRef.current()
          } else {
            handleGenerateRef.current()
          }
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // ── Gallery click → load params ──────────────────────────────────────
  const handleGallerySelect = useCallback(
    (img: GeneratedImage) => {
      // Mark as user-focused so auto-advance doesn't override
      userFocusedRef.current = true
      setActiveSessionId(null)
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

  // ── Send image to edit mode ─────────────────────────────────────────
  const sendToEdit = useCallback(
    (imageUrl: string, serverPath: string) => {
      setForm((prev) => {
        // Auto-select qwen-image-edit if available + apply its defaults
        let modelId = prev.base_model_id
        let steps = prev.steps
        let guidance = prev.guidance
        const editModel = models.find(
          (m) => m.id === 'qwen-image-edit' || m.name.toLowerCase().includes('qwen-image-edit'),
        )
        if (editModel) {
          modelId = editModel.id
          const info = findModelFamily(editModel.name, families)
          const defaults = modelDefaults(editModel.name, info)
          steps = defaults.steps
          guidance = defaults.guidance
        }

        return {
          ...prev,
          mode: 'edit' as const,
          base_model_id: modelId,
          steps,
          guidance,
          edit_images: [
            { type: 'server' as const, preview: imageUrl, serverPath },
          ],
        }
      })
    },
    [models, families, setForm],
  )

  // ── Session strip handlers ──────────────────────────────────────────
  const handleSessionSelect = useCallback(
    (item: SessionItem) => {
      userFocusedRef.current = true
      setActiveSessionId(item.id)
      if (item.status === 'completed' && item.images && item.images.length > 0) {
        setPreviewImages(item.images.map((img) => ({ url: img.url, seed: img.seed })))
      }
      if (item.status === 'error' && item.error) {
        toast.error(item.error)
      }
    },
    [],
  )

  const handleCancelQueued = useCallback(
    async (sessionId: string) => {
      // Read from ref to avoid depending on sessionItems (re-render cascade)
      const items = sessionItemsRef.current
      const queuedItems = items.filter((s) => s.status === 'queued')
      const queueIdx = queuedItems.findIndex((s) => s.id === sessionId)
      if (queueIdx >= 0) {
        await api.cancelQueueItem(queueIdx)
      }
      setSessionItems((prev) => prev.filter((s) => s.id !== sessionId))
    },
    [],
  )

  const activePreviewImage = previewImages[0]

  // ── Generating context for the loading card ───────────────────────────
  const generatingContext = useMemo((): GeneratingContext | undefined => {
    if (!isGenerating) return undefined
    const activeSession = sessionItems.find((s) => s.status === 'active')
    if (!activeSession) return undefined
    // Queue position: count how many items are ahead (0 = actively generating)
    const queuedBefore = sessionItems.filter((s) => s.status === 'queued').length
    const selectedModel = models.find((m) => m.id === activeSession.model_id)
    return {
      prompt: activeSession.prompt,
      modelName: selectedModel?.name ?? activeSession.model_id,
      step: activeSession.step,
      totalSteps: activeSession.totalSteps,
      queuePosition: progressState.status === 'submitting' ? (queuedBefore > 0 ? queuedBefore : 0) : 0,
      onCancel: () => {
        setProgressState({ status: 'idle' })
      },
    }
  }, [isGenerating, sessionItems, models, progressState.status])

  // ── Render ───────────────────────────────────────────────────────────
  return (
    <div className="flex h-full">
      {/* ──────────────── Control Panel (Left Sidebar) ──────────────── */}
      <div className="flex w-[340px] shrink-0 flex-col border-r border-border/40 lg:w-[360px]">
        {/* Mode switcher */}
        <div className="flex shrink-0 border-b border-border/40 px-4 py-2">
          <div className="flex w-full rounded-lg bg-secondary/30 p-0.5">
            <button
              type="button"
              onClick={() => handleModeSwitch(form.mode === 'edit' ? 'txt2img' : form.mode)}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                !isEditMode
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <SparklesIcon className="size-3" />
              Generate
            </button>
            <button
              type="button"
              onClick={() => handleModeSwitch('edit')}
              className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                isEditMode
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <PencilIcon className="size-3" />
              Edit
            </button>
          </div>
        </div>

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
            placeholder={isEditMode ? 'Describe how to edit the image(s)...' : undefined}
          />

          {/* ─── Model & LoRA ─── */}
          <CollapsibleSection title="Model">
            <div className="space-y-3">
              <ModelPanel models={models} families={families} form={form} setForm={setForm} />
              <LoraPanel models={models} families={families} form={form} setForm={setForm} />
            </div>
          </CollapsibleSection>

          {/* ─── Edit Images (edit mode only) ─── */}
          {isEditMode && (
            <CollapsibleSection title="Source Images" defaultOpen={true}>
              <EditImagesPanel form={form} setForm={setForm} />
            </CollapsibleSection>
          )}

          {/* ─── Dimensions ─── */}
          {!isEditMode && (
            <CollapsibleSection title="Dimensions">
              <div className="space-y-3">
                <SizePanel form={form} setForm={setForm} />
                <BatchPanel form={form} setForm={setForm} />
              </div>
            </CollapsibleSection>
          )}

          {/* ─── Reference Image (not shown in edit mode) ─── */}
          {!isEditMode && (
            <CollapsibleSection title="Reference Image" defaultOpen={false}>
              <Img2ImgPanel
                form={form}
                setForm={setForm}
                modelInfo={
                  models.find((m) => m.id === form.base_model_id)
                    ? findModelFamily(models.find((m) => m.id === form.base_model_id)!.name, families)
                    : null
                }
              />
            </CollapsibleSection>
          )}

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
            isEditMode={isEditMode}
            onGenerate={isEditMode ? handleEdit : handleGenerate}
            onInterrupt={() => {
              setProgressState({ status: 'idle' })
            }}
            onClearQueue={async () => {
              await api.clearQueue()
              setQueueCount(0)
              // Remove queued session items
              setSessionItems((prev) => prev.filter((s) => s.status !== 'queued'))
              toast.info('Queue cleared')
            }}
            sessionItems={sessionItems}
            onRemoveQueueItem={handleCancelQueued}
            models={models}
            families={families}
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
            onImageClick={(img) => window.open(img.url, '_blank')}
            generating={generatingContext}
          />

          {/* Floating toolbar — visible when an image is shown */}
          {activePreviewImage && !isGenerating && (
            <div className="absolute right-4 top-4 flex items-center gap-1 rounded-lg border border-border/40 bg-background/80 px-1.5 py-1 shadow-lg backdrop-blur">
              <button
                type="button"
                onClick={downloadImage}
                className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                title="Download"
              >
                <DownloadIcon className="size-3.5" />
              </button>
              <div className="mx-0.5 h-4 w-px bg-border/40" />
              <button
                type="button"
                onClick={() => {
                  const img = previewImages[0]
                  if (!img) return
                  // Extract server path from URL: /files/outputs/... → outputs/...
                  const serverPath = img.url.replace(/^\/files\//, '')
                  sendToEdit(img.url, serverPath)
                }}
                className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                title="Send to Edit mode"
              >
                <PencilIcon className="size-3.5" />
              </button>
            </div>
          )}
        </div>

        {/* Session strip + history (bottom of canvas) */}
        <div className="shrink-0 border-t border-border/30 bg-[#0a0a14]/90 px-4 py-2 backdrop-blur">
          <SessionStrip
            sessionItems={sessionItems}
            onSessionSelect={handleSessionSelect}
            onHistorySelect={handleGallerySelect}
            onCancelQueued={handleCancelQueued}
            activePath={previewImages[0]?.url?.replace('/files/', '') ?? null}
            activeSessionId={activeSessionId}
          />
        </div>
      </div>
    </div>
  )
}
