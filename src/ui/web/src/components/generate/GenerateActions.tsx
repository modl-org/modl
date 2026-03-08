import { useMemo } from 'react'
import { ListIcon, SparklesIcon, SquareIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { GpuStatus } from '../../api'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  gpu: GpuStatus
  isGenerating: boolean
  queueCount: number
  onGenerate: () => void
  onInterrupt?: () => void
  onClearQueue?: () => void
}

export function GenerateActions({ form, gpu, isGenerating, queueCount, onGenerate, onInterrupt, onClearQueue }: Props) {
  const canSubmit = useMemo(() => {
    return (
      !gpu.training_active &&
      form.prompt.trim().length > 0 &&
      form.base_model_id.length > 0
    )
  }, [form.base_model_id, form.prompt, gpu.training_active])

  const buttonLabel = gpu.training_active
    ? 'GPU busy — training'
    : isGenerating
      ? `Enqueue${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`
      : `Generate${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`

  return (
    <div className="w-full space-y-2">
      <div className="flex gap-2">
        <Button
          type="submit"
          disabled={!canSubmit}
          className="h-10 flex-1 gap-2 px-6"
          onClick={onGenerate}
        >
          {isGenerating ? (
            <ListIcon className="size-4" />
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
          </Button>
        )}
      </div>

      {/* Queue indicator */}
      {queueCount > 0 && (
        <div className="flex items-center justify-between rounded bg-secondary/30 px-2.5 py-1">
          <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <ListIcon className="size-3" />
            {queueCount} queued
          </span>
          <button
            type="button"
            className="rounded p-0.5 text-muted-foreground/60 transition-colors hover:text-foreground"
            onClick={onClearQueue}
            title="Clear queue"
          >
            <XIcon className="size-3" />
          </button>
        </div>
      )}

      {/* GPU info — below the button, centered */}
      <div className="flex items-center justify-center gap-2">
        {!gpu.training_active && gpu.vram_free_mb != null && (
          <span className="text-[10px] text-muted-foreground/50">
            GPU: {(gpu.vram_free_mb / 1024).toFixed(1)} GB free
          </span>
        )}
        {gpu.training_active && (
          <span className="flex items-center gap-1.5 text-[10px] text-amber-400/80">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
            Training active
          </span>
        )}
      </div>
    </div>
  )
}
