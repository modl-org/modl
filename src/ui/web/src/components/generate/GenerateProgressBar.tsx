import {
  AlertTriangleIcon,
  CheckCircle2Icon,
  LoaderCircleIcon,
  RadioIcon,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Generation progress states
// ---------------------------------------------------------------------------

export type GenerateProgressState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'streaming'; lines: string[]; step?: number; totalSteps?: number }
  | { status: 'done'; count: number; images: string[] }
  | { status: 'error'; message: string }

type Props = {
  state: GenerateProgressState
}

export function GenerateProgressBar({ state }: Props) {
  if (state.status === 'idle') return null

  return (
    <div className="flex items-center gap-2 rounded-md border border-border/40 bg-secondary/10 px-3 py-2">
      {/* Icon */}
      {state.status === 'submitting' && (
        <LoaderCircleIcon className="size-3.5 shrink-0 animate-spin text-primary/70" />
      )}
      {state.status === 'streaming' && (
        <RadioIcon className="size-3.5 shrink-0 text-emerald-400" />
      )}
      {state.status === 'done' && (
        <CheckCircle2Icon className="size-3.5 shrink-0 text-emerald-400" />
      )}
      {state.status === 'error' && (
        <AlertTriangleIcon className="size-3.5 shrink-0 text-destructive" />
      )}

      {/* Text */}
      <div className="min-w-0 flex-1">
        {state.status === 'submitting' && (
          <span className="text-xs text-muted-foreground">Submitting generation job...</span>
        )}
        {state.status === 'streaming' && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground">
              {state.lines.length > 0 ? state.lines[state.lines.length - 1] : 'Generating...'}
            </span>
            {state.step != null && state.totalSteps != null && (
              <div className="flex items-center gap-1.5">
                <div className="h-1 w-20 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full bg-primary transition-all"
                    style={{
                      width: `${Math.min(100, (state.step / state.totalSteps) * 100)}%`,
                    }}
                  />
                </div>
                <span className="font-mono text-[10px] text-muted-foreground/60">
                  {state.step}/{state.totalSteps}
                </span>
              </div>
            )}
          </div>
        )}
        {state.status === 'done' && (
          <span className="text-xs text-emerald-300">
            Generated {state.count} image{state.count !== 1 ? 's' : ''}
          </span>
        )}
        {state.status === 'error' && (
          <span className="text-xs text-destructive">{state.message}</span>
        )}
      </div>
    </div>
  )
}
