import { useMemo } from 'react'
import { LoaderCircleIcon, SparklesIcon, SquareIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { GpuStatus } from '../../api'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  gpu: GpuStatus
  isGenerating: boolean
  onGenerate: () => void
  onInterrupt?: () => void
}

export function GenerateActions({ form, gpu, isGenerating, onGenerate, onInterrupt }: Props) {
  const canSubmit = useMemo(() => {
    return (
      !gpu.training_active &&
      !isGenerating &&
      form.prompt.trim().length > 0 &&
      form.base_model_id.length > 0
    )
  }, [form.base_model_id, form.prompt, gpu.training_active, isGenerating])

  const buttonLabel = gpu.training_active
    ? 'GPU busy — training'
    : isGenerating
      ? 'Generating...'
      : `Generate${form.batch_count > 1 ? ` (${form.batch_count})` : ''}`

  return (
    <div className="flex items-center gap-3">
      {isGenerating ? (
        <Button
          type="button"
          variant="destructive"
          className="h-10 gap-2 px-6"
          onClick={onInterrupt}
        >
          <SquareIcon className="size-3.5" />
          Interrupt
        </Button>
      ) : (
        <Button
          type="submit"
          disabled={!canSubmit}
          className="h-10 gap-2 px-6"
          onClick={onGenerate}
        >
          {isGenerating ? (
            <LoaderCircleIcon className="size-4 animate-spin" />
          ) : (
            <SparklesIcon className="size-4" />
          )}
          {buttonLabel}
        </Button>
      )}

      {/* GPU info */}
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
  )
}
