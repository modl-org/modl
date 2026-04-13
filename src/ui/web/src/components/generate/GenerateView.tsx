import { useCallback, useEffect, useMemo, useRef } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  DownloadIcon,
  PencilIcon,
  SparklesIcon,
} from 'lucide-react'
import { api, type GeneratedImage, type GpuStatus, type InstalledModel, type ModelFamily } from '../../api'
import { STALE_SLOW, STALE_STATIC } from '@/lib/query-keys'
import { useGpuStatus } from '../../hooks/useGpuStatus'
import { useForm } from '../../contexts/FormContext'
import { CollapsibleSection } from '../ui/collapsible-section'
import { BatchPanel } from './BatchPanel'
import { GenerateActions } from './GenerateActions'
import { GenerateProgressBar } from './GenerateProgressBar'
import { ImagePreview, type GeneratingContext } from './ImagePreview'
import { SessionStrip } from './SessionStrip'
import { Img2ImgPanel } from './Img2ImgPanel'
import { LoraPanel } from './LoraPanel'
import { ModelPanel } from './ModelPanel'
import { PromptPanel } from './PromptPanel'
import { SamplingPanel } from './SamplingPanel'
import { SizePanel } from './SizePanel'
import { VideoPanel } from './VideoPanel'
import { EditImagesPanel } from './EditImagesPanel'
import { buildSendToEdit, findModelFamily, modelDefaults, type GenerateFormState, type GenerationMode } from './generate-state'
import { useGenerateQueue } from './useGenerateQueue'
import { useGenerateSubmit } from './useGenerateSubmit'

// ---------------------------------------------------------------------------
// GenerateView — pro sidebar layout with scrollable controls + fixed canvas
// ---------------------------------------------------------------------------

export function GenerateView() {
  const { form, setForm } = useForm()
  // ── Queue, streaming & preview ─────────────────────────────────────
  const queue = useGenerateQueue()

  // ── Queries ────────────────────────────────────────────────────────
  const { data: gpu = { training_active: false } as GpuStatus } = useGpuStatus()

  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: STALE_SLOW,
  })
  const models = modelsResponse?.models ?? []

  const { data: families = [] } = useQuery<ModelFamily[]>({
    queryKey: ['model-families'],
    queryFn: api.modelFamilies,
    staleTime: STALE_STATIC,
  })

  // ── Submit ─────────────────────────────────────────────────────────
  const { handleGenerate, handleEdit, submitCurrent } = useGenerateSubmit({
    form,
    setForm,
    nextSessionId: queue.nextSessionId,
    enqueueJob: queue.enqueueJob,
    onSubmitted: queue.onSubmitted,
    markSubmitError: queue.markSubmitError,
  })

  // Auto-select first checkpoint if none selected
  useEffect(() => {
    if (!form.base_model_id && models.length > 0) {
      const first = models.find((m: InstalledModel) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')
      if (first) setForm((prev) => ({ ...prev, base_model_id: first.id }))
    }
  }, [models, form.base_model_id, setForm])

  // ── Mode switch ───────────────────────────────────────────────────
  const preEditModelRef = useRef<string | null>(null)

  const handleModeSwitch = useCallback((mode: GenerationMode) => {
    setForm((prev) => {
      const updates: Partial<GenerateFormState> = { mode }
      if (mode === 'edit') {
        preEditModelRef.current = prev.base_model_id
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
        const currentInfo = findModelFamily(prev.base_model_id, families)
        const currentSupportsTxt2img = !currentInfo || currentInfo.capabilities.txt2img
        if (!currentSupportsTxt2img) {
          const savedId = preEditModelRef.current
          const savedModel = savedId ? models.find((m) => m.id === savedId) : null
          if (savedModel) {
            updates.base_model_id = savedModel.id
            const info = findModelFamily(savedModel.name, families)
            const defaults = modelDefaults(savedModel.name, info)
            updates.steps = defaults.steps
            updates.guidance = defaults.guidance
          } else {
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
        updates.edit_images = []
      }
      return { ...prev, ...updates }
    })
  }, [models, families, setForm])

  const isEditMode = form.mode === 'edit'

  // ── Keyboard shortcut: Ctrl/Cmd + Enter ────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault()
        submitCurrent()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [submitCurrent])

  // ── Gallery click → load params ───────────────────────────────────
  const { selectGalleryImage } = queue

  const handleGallerySelect = useCallback(
    (img: GeneratedImage) => {
      selectGalleryImage(img)
      setForm((prev) => ({
        ...prev,
        ...(img.prompt ? { prompt: img.prompt } : {}),
        ...(img.seed != null ? { seed: img.seed } : {}),
        ...(img.steps != null ? { steps: img.steps } : {}),
        ...(img.guidance != null ? { guidance: img.guidance } : {}),
        ...(img.width != null && img.height != null ? { width: img.width, height: img.height } : {}),
        ...(img.base_model_id ? { base_model_id: img.base_model_id } : {}),
      }))
    },
    [setForm, selectGalleryImage],
  )

  // ── Derived values ────────────────────────────────────────────────
  const selectedModel = useMemo(
    () => models.find((m) => m.id === form.base_model_id),
    [models, form.base_model_id],
  )

  const isVideoModel = useMemo(() => {
    if (!selectedModel) return false
    const info = findModelFamily(selectedModel.name, families)
    return info?.capabilities?.txt2vid || info?.capabilities?.img2vid || false
  }, [selectedModel, families])

  const checkpointCount = useMemo(
    () => models.filter((m: InstalledModel) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model').length,
    [models],
  )

  const downloadImage = useCallback(() => {
    const img = queue.previewImages[0]
    if (!img) return
    const a = document.createElement('a')
    a.href = img.url
    a.download = img.url.split('/').pop() ?? 'image.png'
    a.click()
  }, [queue.previewImages])

  // ── Send image to edit mode ───────────────────────────────────────
  const sendToEdit = useCallback(
    (imageUrl: string, serverPath: string) => {
      setForm(buildSendToEdit(imageUrl, serverPath, models, families))
    },
    [models, families, setForm],
  )

  // ── Generating context for loading card ───────────────────────────
  const generatingContext = useMemo((): GeneratingContext | undefined => {
    if (!queue.isGenerating) return undefined
    const activeSession = queue.sessionItems.find((s) => s.status === 'active')
    if (!activeSession) return undefined
    const queuedBefore = queue.sessionItems.filter((s) => s.status === 'queued').length
    // In edit mode, pass the first source image as blurred background
    const sourceImageUrl = isEditMode && form.edit_images.length > 0
      ? form.edit_images[0].preview
      : undefined
    return {
      prompt: activeSession.prompt,
      modelName: selectedModel?.name ?? activeSession.model_id,
      step: activeSession.step,
      totalSteps: activeSession.totalSteps,
      queuePosition: queue.progressState.status === 'submitting' ? (queuedBefore > 0 ? queuedBefore : 0) : 0,
      onCancel: queue.interrupt,
      sourceImageUrl,
    }
  }, [queue.isGenerating, queue.sessionItems, selectedModel, queue.progressState.status, queue.interrupt, isEditMode, form.edit_images])

  const activePreviewImage = queue.previewImages[0]

  // ── Render ────────────────────────────────────────────────────────
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
          {checkpointCount === 0 && (
            <div className="mb-4 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              No checkpoints installed. Run{' '}
              <code className="rounded bg-secondary px-1 py-0.5 font-mono text-[10px]">
                modl pull sdxl
              </code>{' '}
              to get started.
            </div>
          )}

          <PromptPanel
            form={form}
            setForm={setForm}
            modelHint={selectedModel?.name}
            placeholder={isEditMode ? 'Describe how to edit the image(s)...' : undefined}
          />

          <CollapsibleSection title="Model">
            <div className="space-y-3">
              <ModelPanel models={models} families={families} form={form} setForm={setForm} />
              <LoraPanel models={models} families={families} form={form} setForm={setForm} />
            </div>
          </CollapsibleSection>

          {isEditMode && (
            <CollapsibleSection title="Source Images" defaultOpen={true}>
              <EditImagesPanel form={form} setForm={setForm} />
            </CollapsibleSection>
          )}

          {!isEditMode && (
            <CollapsibleSection title="Dimensions">
              <div className="space-y-3">
                <SizePanel form={form} setForm={setForm} />
                <BatchPanel form={form} setForm={setForm} />
              </div>
            </CollapsibleSection>
          )}

          {!isEditMode && isVideoModel && (
            <CollapsibleSection title="Video" defaultOpen={true}>
              <VideoPanel form={form} setForm={setForm} />
            </CollapsibleSection>
          )}

          {!isEditMode && (
            <CollapsibleSection title="Reference Image" defaultOpen={false}>
              <Img2ImgPanel
                form={form}
                setForm={setForm}
                modelInfo={selectedModel ? findModelFamily(selectedModel.name, families) : null}
              />
            </CollapsibleSection>
          )}

          <CollapsibleSection title="Sampling" defaultOpen={false}>
            <SamplingPanel form={form} setForm={setForm} />
          </CollapsibleSection>
        </div>

        {/* Sticky Generate Button */}
        <div className="shrink-0 border-t border-border/40 bg-[#0e0e18]/95 px-4 py-3 backdrop-blur">
          <GenerateActions
            form={form}
            gpu={gpu}
            isGenerating={queue.isGenerating}
            isEditMode={isEditMode}
            onGenerate={isEditMode ? handleEdit : handleGenerate}
            onInterrupt={queue.interrupt}
            onClearQueue={queue.clearQueue}
            sessionItems={queue.sessionItems}
            onRemoveQueueItem={queue.cancelQueued}
            models={models}
            families={families}
          />
          <GenerateProgressBar state={queue.progressState} />
          <p className="mt-1 text-center text-[10px] text-muted-foreground/30">
            <kbd className="rounded border border-border/40 bg-secondary/20 px-1 py-0.5 font-mono text-[9px]">
              Ctrl+Enter
            </kbd>
          </p>
        </div>
      </div>

      {/* ──────────────── Canvas (Right Area) ──────────────── */}
      <div className="relative flex min-w-0 flex-1 flex-col bg-[#080810]">
        <div className="relative flex flex-1 items-center justify-center overflow-hidden p-6">
          <ImagePreview
            images={queue.previewImages}
            isGenerating={queue.isGenerating}
            expectedCount={form.batch_count}
            width={form.width}
            height={form.height}
            onImageClick={(img) => window.open(img.url, '_blank')}
            onClear={queue.clearPreview}
            generating={generatingContext}
          />

          {activePreviewImage && !queue.isGenerating && (
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
                  const img = queue.previewImages[0]
                  if (!img) return
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

        <div className="shrink-0 border-t border-border/30 bg-[#0a0a14]/90 px-4 py-2 backdrop-blur">
          <SessionStrip
            sessionItems={queue.sessionItems}
            onSessionSelect={queue.selectSession}
            onHistorySelect={handleGallerySelect}
            onCancelQueued={queue.cancelQueued}
            activePath={queue.previewImages[0]?.url?.replace('/files/', '') ?? null}
            activeSessionId={queue.activeSessionId}
          />
        </div>
      </div>
    </div>
  )
}
