import { useMemo } from 'react'
import { AlertTriangleIcon, Loader2Icon, PencilIcon, SparklesIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { GpuStatus, InstalledModel, ModelFamily } from '../../api'
import { findModelFamily, type GenerateFormState } from './generate-state'
import type { SessionItem } from './SessionStrip'

type Props = {
  form: GenerateFormState
  gpu: GpuStatus
  isGenerating: boolean
  isEditMode?: boolean
  onGenerate: () => void
  onInterrupt?: () => void
  onClearQueue?: () => void
  /** Session items for the inline queue panel */
  sessionItems?: SessionItem[]
  /** Remove a queued item by session ID */
  onRemoveQueueItem?: (id: string) => void
  /** For VRAM warning calculation */
  models?: InstalledModel[]
  families?: ModelFamily[]
}

export function GenerateActions({
  form,
  gpu,
  isEditMode,
  onGenerate,
  onInterrupt,
  onClearQueue,
  sessionItems = [],
  onRemoveQueueItem,
  models = [],
  families = [],
}: Props) {
  const canSubmit = useMemo(() => {
    const baseOk = !gpu.training_active && form.prompt.trim().length > 0 && form.base_model_id.length > 0
    if (isEditMode) return baseOk && (form.edit_images?.length ?? 0) > 0
    return baseOk
  }, [form.base_model_id, form.prompt, form.edit_images.length, gpu.training_active, isEditMode])

  const actionWord = isEditMode ? 'Edit' : 'Generate'
  const buttonLabel = gpu.training_active
    ? 'GPU busy — training'
    : `${actionWord}${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`

  // Active + queued items
  const activeAndQueued = sessionItems.filter((s) => s.status === 'active' || s.status === 'queued')
  const queuedItems = activeAndQueued.filter((s) => s.status === 'queued')

  return (
    <div className="w-full space-y-2">
      {/* Generate button — always shows Generate/Edit, never changes to Enqueue */}
      <Button
        type="submit"
        disabled={!canSubmit}
        className="h-10 w-full gap-2 px-6"
        onClick={onGenerate}
      >
        {isEditMode ? (
          <PencilIcon className="size-4" />
        ) : (
          <SparklesIcon className="size-4" />
        )}
        {buttonLabel}
      </Button>

      {/* Queue — always visible when there are active/queued items */}
      {activeAndQueued.length > 0 && (
        <div className="space-y-0.5 rounded-lg border border-border/30 bg-secondary/10 px-2.5 py-2">
          {activeAndQueued.map((job) => {
            const isActive = job.status === 'active'
            return (
              <div key={job.id} className="group flex items-center gap-2 py-0.5">
                {isActive ? (
                  <Loader2Icon className="size-3 shrink-0 animate-spin text-primary" />
                ) : (
                  <div className="size-1.5 shrink-0 rounded-full bg-muted-foreground/20" />
                )}
                <span className={`min-w-0 truncate text-[11px] ${isActive ? 'font-medium text-foreground' : 'text-muted-foreground/60'}`}>
                  {job.prompt.length > 40 ? job.prompt.slice(0, 40) + '\u2026' : job.prompt}
                </span>
                {isActive && job.step != null && job.totalSteps != null && job.totalSteps > 0 && (
                  <span className="shrink-0 text-[9px] text-muted-foreground/40">
                    {job.step}/{job.totalSteps}
                  </span>
                )}
                {/* Cancel button — active item uses onInterrupt, queued items use onRemoveQueueItem */}
                {isActive && onInterrupt && (
                  <button
                    type="button"
                    onClick={onInterrupt}
                    className="ml-auto shrink-0 rounded p-0.5 text-muted-foreground/30 opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive"
                    title="Stop"
                  >
                    <XIcon className="size-3" />
                  </button>
                )}
                {!isActive && onRemoveQueueItem && (
                  <button
                    type="button"
                    onClick={() => onRemoveQueueItem(job.id)}
                    className="ml-auto shrink-0 rounded p-0.5 text-muted-foreground/30 opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive"
                    title="Remove"
                  >
                    <XIcon className="size-3" />
                  </button>
                )}
              </div>
            )
          })}
          {/* Clear all queued — only when 2+ queued */}
          {queuedItems.length > 1 && onClearQueue && (
            <div className="pt-1">
              <button
                type="button"
                onClick={onClearQueue}
                className="text-[10px] text-muted-foreground/40 transition-colors hover:text-destructive"
              >
                Clear all queued
              </button>
            </div>
          )}
        </div>
      )}

      {/* GPU info with VRAM warning */}
      <GpuInfoLine gpu={gpu} form={form} models={models} families={families} />
    </div>
  )
}

// ---------------------------------------------------------------------------

function GpuInfoLine({
  gpu,
  form,
  models,
  families,
}: {
  gpu: GpuStatus
  form: GenerateFormState
  models: InstalledModel[]
  families: ModelFamily[]
}) {
  if (gpu.training_active) {
    return (
      <div className="flex items-center justify-center gap-1.5 text-[10px] text-amber-400/80">
        <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
        Training active
      </div>
    )
  }

  const vramFreeMb = gpu.vram_free_mb
  if (vramFreeMb == null) return null

  const vramFreeGb = vramFreeMb / 1024

  // Estimate VRAM needed for the selected model
  const selectedModel = models.find((m) => m.id === form.base_model_id)
  const modelInfo = selectedModel ? findModelFamily(selectedModel.name, families) : null
  const estimateGb = modelInfo
    ? (selectedModel?.variant?.includes('fp8') || selectedModel?.variant?.includes('gguf'))
      ? modelInfo.vram_fp8_gb
      : modelInfo.vram_bf16_gb
    : null

  const isLow = estimateGb != null && vramFreeGb < estimateGb

  if (isLow) {
    return (
      <div className="flex items-center justify-center gap-1.5 rounded border border-amber-500/30 bg-amber-500/5 px-2 py-1 text-[10px] text-amber-400">
        <AlertTriangleIcon className="size-3 shrink-0" />
        <span>
          {vramFreeGb.toFixed(1)} GB free — model needs ~{estimateGb.toFixed(0)} GB
        </span>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center gap-2 text-[10px] text-muted-foreground/50">
      GPU: {vramFreeGb.toFixed(1)} GB free
    </div>
  )
}
