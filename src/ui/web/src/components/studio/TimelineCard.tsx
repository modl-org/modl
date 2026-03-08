import { Check, Circle, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { AgentEvent } from '../../api'

type Props = {
  event: AgentEvent
  isActive: boolean
  isCompleted: boolean
}

export function TimelineCard({ event, isActive, isCompleted }: Props) {
  const label = getLabel(event)
  const detail = getDetail(event)
  const progress = event.progress

  return (
    <div
      className={cn(
        'flex gap-3 rounded-lg border px-4 py-3 transition-colors',
        isActive
          ? 'border-primary/30 bg-primary/5'
          : isCompleted
            ? 'border-border/30 bg-background'
            : 'border-border/20 bg-background/50 opacity-60',
      )}
    >
      {/* Status icon */}
      <div className="mt-0.5 shrink-0">
        {isCompleted ? (
          <div className="flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/20">
            <Check className="h-3 w-3 text-emerald-400" />
          </div>
        ) : isActive ? (
          <Loader2 className="h-5 w-5 animate-spin text-primary" />
        ) : (
          <Circle className="h-5 w-5 text-muted-foreground/40" />
        )}
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        <p
          className={cn(
            'text-sm font-medium',
            isCompleted
              ? 'text-muted-foreground'
              : isActive
                ? 'text-foreground'
                : 'text-muted-foreground/60',
          )}
        >
          {label}
        </p>

        {detail && (isActive || isCompleted) && (
          <p className="mt-0.5 text-xs text-muted-foreground/80 truncate">
            {detail}
          </p>
        )}

        {/* Progress bar */}
        {isActive && progress != null && progress > 0 && progress < 1 && (
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-[width] duration-300"
              style={{ width: `${Math.round(progress * 100)}%` }}
            />
          </div>
        )}
      </div>

      {/* Progress percentage */}
      {isActive && progress != null && progress > 0 && progress < 1 && (
        <span className="shrink-0 text-xs font-medium text-primary tabular-nums">
          {Math.round(progress * 100)}%
        </span>
      )}
    </div>
  )
}

function getLabel(event: AgentEvent): string {
  switch (event.type) {
    case 'thinking':
      return event.message ?? 'Thinking...'
    case 'tool_start':
      return event.description ?? `Running ${event.tool ?? 'tool'}...`
    case 'tool_progress':
      return event.description ?? event.detail ?? `${event.tool ?? 'Tool'} in progress`
    case 'tool_complete':
      return toolDisplayName(event.tool) ?? 'Step complete'
    case 'output_ready':
      return 'Your photos are ready!'
    case 'error':
      return event.message ?? 'An error occurred'
    default:
      return 'Processing...'
  }
}

function getDetail(event: AgentEvent): string | undefined {
  switch (event.type) {
    case 'tool_start':
      return event.description
    case 'tool_progress':
      return event.detail
    case 'tool_complete':
      return event.result
    default:
      return undefined
  }
}

function toolDisplayName(tool?: string): string | undefined {
  switch (tool) {
    case 'analyze_images':
      return 'Understanding your photos'
    case 'create_dataset':
      return 'Preparing dataset'
    case 'caption_images':
      return 'Captioning images'
    case 'select_base_model':
      return 'Selecting model'
    case 'train_lora':
      return 'Training custom model'
    case 'enhance_prompt':
      return 'Crafting prompts'
    case 'generate_images':
      return 'Creating your photos'
    default:
      return tool
  }
}
