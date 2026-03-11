import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangleIcon, ChevronUpIcon, ListIcon, Loader2Icon, PencilIcon, SparklesIcon, SquareIcon, XIcon } from 'lucide-react'
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
  isGenerating,
  isEditMode,
  onGenerate,
  onInterrupt,
  onClearQueue,
  sessionItems = [],
  onRemoveQueueItem,
  models = [],
  families = [],
}: Props) {
  const [showQueuePanel, setShowQueuePanel] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  const canSubmit = useMemo(() => {
    const baseOk = !gpu.training_active && form.prompt.trim().length > 0 && form.base_model_id.length > 0
    if (isEditMode) return baseOk && (form.edit_images?.length ?? 0) > 0
    return baseOk
  }, [form.base_model_id, form.prompt, form.edit_images.length, gpu.training_active, isEditMode])

  const actionWord = isEditMode ? 'Edit' : 'Generate'
  const buttonLabel = gpu.training_active
    ? 'GPU busy — training'
    : isGenerating
      ? `Enqueue${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`
      : `${actionWord}${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`

  // Active + queued items for the queue panel
  const activeAndQueued = sessionItems.filter((s) => s.status === 'active' || s.status === 'queued')
  const activeItem = activeAndQueued.find((s) => s.status === 'active')
  const hasQueueItems = activeAndQueued.length > 0

  // Close panel on outside click
  useEffect(() => {
    if (!showQueuePanel) return
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setShowQueuePanel(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showQueuePanel])

  return (
    <div className="w-full space-y-2">
      {/* Generate / Enqueue + Stop */}
      <div className="flex gap-2">
        <Button
          type="submit"
          disabled={!canSubmit}
          className="h-10 flex-1 gap-2 px-6"
          onClick={onGenerate}
        >
          {isGenerating ? (
            <ListIcon className="size-4" />
          ) : isEditMode ? (
            <PencilIcon className="size-4" />
          ) : (
            <SparklesIcon className="size-4" />
          )}
          {buttonLabel}
        </Button>
        {isGenerating && (
          <Button
            type="button"
            variant="destructive"
            className="h-10 gap-1.5 px-3"
            onClick={onInterrupt}
            title="Stop current generation"
          >
            <SquareIcon className="size-3.5" />
            <span className="text-xs">Stop</span>
          </Button>
        )}
      </div>

      {/* Queue status bar — rich, clickable, expandable */}
      <div ref={panelRef} className="relative">
        {/* Expandable queue panel */}
        {showQueuePanel && hasQueueItems && (
          <div className="absolute inset-x-0 bottom-full mb-2 overflow-hidden rounded-lg border border-border/50 bg-[#0e0e18] shadow-xl">
            {/* Header */}
            <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
                Queue &middot; {activeAndQueued.length} {activeAndQueued.length === 1 ? 'job' : 'jobs'}
              </span>
              <button
                type="button"
                onClick={() => setShowQueuePanel(false)}
                className="rounded p-0.5 text-muted-foreground/40 transition-colors hover:text-foreground"
              >
                <XIcon className="size-3" />
              </button>
            </div>

            {/* Job list */}
            <div className="max-h-48 overflow-y-auto">
              {activeAndQueued.map((job, idx) => {
                const isActive = job.status === 'active'
                return (
                  <div
                    key={job.id}
                    className={`flex items-center gap-2.5 border-b border-border/10 px-3 py-2 last:border-0 ${
                      isActive ? 'bg-primary/5' : ''
                    }`}
                  >
                    {/* Position indicator */}
                    <div
                      className={`flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-bold ${
                        isActive
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-secondary text-muted-foreground/50'
                      }`}
                    >
                      {isActive ? (
                        <Loader2Icon className="size-3 animate-spin" />
                      ) : (
                        idx + 1
                      )}
                    </div>

                    {/* Job info */}
                    <div className="min-w-0 flex-1">
                      <p className={`truncate text-xs ${isActive ? 'text-foreground' : 'text-muted-foreground'}`}>
                        {job.prompt}
                      </p>
                      <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground/40">
                        <span>{job.model_id}</span>
                        {job.batch_count > 1 && (
                          <>
                            <span>&middot;</span>
                            <span>{job.batch_count}x</span>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Status + remove */}
                    <div className="flex shrink-0 items-center gap-2">
                      {isActive && (
                        <span className="rounded-full bg-primary/15 px-1.5 py-0.5 text-[9px] font-semibold text-primary">
                          generating
                        </span>
                      )}
                      {!isActive && onRemoveQueueItem && (
                        <button
                          type="button"
                          onClick={() => onRemoveQueueItem(job.id)}
                          className="rounded border border-border/30 px-1.5 py-0.5 text-[10px] text-muted-foreground/40 transition-colors hover:border-destructive/50 hover:text-destructive"
                        >
                          remove
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Footer: clear all */}
            {activeAndQueued.filter((s) => s.status === 'queued').length > 1 && onClearQueue && (
              <div className="border-t border-border/30 px-3 py-1.5">
                <button
                  type="button"
                  onClick={() => {
                    onClearQueue()
                    setShowQueuePanel(false)
                  }}
                  className="text-[10px] text-muted-foreground/50 transition-colors hover:text-destructive"
                >
                  Clear all queued
                </button>
              </div>
            )}
          </div>
        )}

        {/* Queue status — mini-list when >1 job, single row otherwise */}
        {activeAndQueued.length > 1 ? (
          <div className="space-y-0.5 rounded-lg border border-border/30 bg-secondary/15 px-2.5 py-1.5">
            {activeAndQueued.slice(0, 4).map((job) => {
              const isJobActive = job.status === 'active'
              return (
                <div key={job.id} className="flex items-center gap-2">
                  {isJobActive ? (
                    <Loader2Icon className="size-3 shrink-0 animate-spin text-primary" />
                  ) : (
                    <div className="size-1.5 shrink-0 rounded-full bg-muted-foreground/20" />
                  )}
                  <span className={`min-w-0 truncate text-[11px] ${isJobActive ? 'font-medium text-foreground' : 'text-muted-foreground/60'}`}>
                    {job.prompt.length > 35 ? job.prompt.slice(0, 35) + '\u2026' : job.prompt}
                  </span>
                  {!isJobActive && onRemoveQueueItem && (
                    <button
                      type="button"
                      onClick={() => onRemoveQueueItem(job.id)}
                      className="ml-auto shrink-0 rounded p-0.5 text-muted-foreground/30 transition-colors hover:text-destructive"
                    >
                      <XIcon className="size-3" />
                    </button>
                  )}
                </div>
              )
            })}
            {activeAndQueued.length > 4 && (
              <span className="block text-[10px] text-muted-foreground/40">+{activeAndQueued.length - 4} more</span>
            )}
            <button
              type="button"
              onClick={() => setShowQueuePanel((o) => !o)}
              className="flex w-full items-center justify-center pt-0.5 text-[10px] text-muted-foreground/40 hover:text-muted-foreground"
            >
              <ChevronUpIcon className={`size-3 transition-transform ${showQueuePanel ? 'rotate-180' : ''}`} />
            </button>
          </div>
        ) : (
          <button
            type="button"
            disabled={!hasQueueItems}
            onClick={() => hasQueueItems && setShowQueuePanel((o) => !o)}
            className={`flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 transition-colors ${
              hasQueueItems
                ? 'border border-border/30 bg-secondary/15 hover:bg-secondary/25 cursor-pointer'
                : 'cursor-default'
            }`}
          >
            <div className="flex min-w-0 items-center gap-2">
              {isGenerating && activeItem && (
                <Loader2Icon className="size-3 shrink-0 animate-spin text-primary" />
              )}
              {hasQueueItems ? (
                <span className="truncate text-xs text-muted-foreground">
                  <span className="font-medium text-primary">
                    {activeItem
                      ? activeItem.prompt.length > 30
                        ? activeItem.prompt.slice(0, 30) + '\u2026'
                        : activeItem.prompt
                      : 'Queued'}
                  </span>
                </span>
              ) : (
                <span className="text-[11px] text-muted-foreground/25">Queue empty</span>
              )}
            </div>
            {hasQueueItems && (
              <ChevronUpIcon className={`size-3 shrink-0 text-muted-foreground/40 transition-transform ${showQueuePanel ? 'rotate-180' : ''}`} />
            )}
          </button>
        )}
      </div>

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
