import { LoaderCircleIcon, SparklesIcon, XIcon } from 'lucide-react'

// ---------------------------------------------------------------------------
// ImagePreview — the main preview area
//
// Shows:
//   - Placeholder when idle
//   - Rich loading card during generation (with prompt, model, progress)
//   - Single image result
//   - Grid for multi-batch results
// ---------------------------------------------------------------------------

export type PreviewImage = {
  url: string
  seed?: number
  index?: number
}

export type GeneratingContext = {
  prompt: string
  modelName: string
  step?: number
  totalSteps?: number
  /** 0 = actively generating, >0 = waiting in queue at this position */
  queuePosition: number
  onCancel?: () => void
}

type Props = {
  images: PreviewImage[]
  isGenerating: boolean
  /** Expected number of images (for skeleton placeholders) */
  expectedCount?: number
  width: number
  height: number
  onImageClick?: (img: PreviewImage) => void
  /** Context about what's being generated — shown in the loading state */
  generating?: GeneratingContext
}

export function ImagePreview({
  images,
  isGenerating,
  expectedCount = 1,
  width,
  height,
  onImageClick,
  generating,
}: Props) {
  const aspectRatio = width / height
  const hasImages = images.length > 0
  const isGrid = images.length > 1 || expectedCount > 1

  // Determine grid columns based on count
  const gridCols =
    expectedCount <= 1
      ? 1
      : expectedCount <= 4
        ? 2
        : expectedCount <= 9
          ? 3
          : 4

  // ── Idle placeholder ──
  if (!hasImages && !isGenerating) {
    return (
      <div
        className="relative flex w-full max-w-lg items-center justify-center rounded-lg border border-dashed border-border/40 bg-secondary/10"
        style={{ aspectRatio: isGrid ? undefined : aspectRatio, minHeight: isGrid ? 300 : 200 }}
      >
        <div className="flex flex-col items-center gap-2 text-muted-foreground/30">
          <div className="text-4xl">🖼</div>
          <span className="text-xs">Generated images will appear here</span>
          <span className="font-mono text-[10px]">
            {width}×{height}
          </span>
        </div>
      </div>
    )
  }

  // ── Generating / in-queue — rich loading card ──
  if (!hasImages && isGenerating && generating) {
    const { prompt, modelName, step, totalSteps, queuePosition, onCancel } = generating
    const isQueued = queuePosition > 0
    const progressPct =
      step != null && totalSteps != null && totalSteps > 0
        ? Math.min(100, (step / totalSteps) * 100)
        : undefined

    return (
      <div className="flex w-full max-w-md flex-col items-center gap-6">
        {/* Animated card */}
        <div
          className="relative w-full overflow-hidden rounded-2xl border border-primary/20"
          style={{ aspectRatio: Math.max(aspectRatio, 0.75) }}
        >
          {/* Animated gradient background */}
          <div
            className="absolute inset-0"
            style={{
              background: 'linear-gradient(135deg, #0e0e18 0%, #1a1030 30%, #150d25 60%, #0e0e18 100%)',
              backgroundSize: '400% 400%',
              animation: 'gradientShift 4s ease-in-out infinite',
            }}
          />

          {/* Subtle shimmer overlay */}
          <div
            className="absolute inset-0 opacity-30"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, rgba(167,139,250,0.08) 50%, transparent 100%)',
              backgroundSize: '200% 100%',
              animation: 'shimmer 2s linear infinite',
            }}
          />

          {/* Content */}
          <div className="relative flex h-full flex-col items-center justify-center gap-4 p-8">
            {/* Spinner / queue indicator */}
            {isQueued ? (
              <div className="flex flex-col items-center gap-2">
                <div className="flex size-12 items-center justify-center rounded-full border border-primary/30 bg-primary/10">
                  <span className="text-lg font-bold text-primary">{queuePosition}</span>
                </div>
                <span className="text-[11px] font-medium uppercase tracking-wider text-primary/60">
                  In queue
                </span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                {progressPct != null ? (
                  <div className="relative flex items-center justify-center">
                    <svg className="size-20 -rotate-90" viewBox="0 0 80 80">
                      <circle cx="40" cy="40" r="34" fill="none" stroke="currentColor" strokeWidth="3" className="text-primary/10" />
                      <circle
                        cx="40" cy="40" r="34" fill="none" stroke="currentColor" strokeWidth="4"
                        strokeDasharray={`${(progressPct / 100) * 213.6} 213.6`}
                        strokeLinecap="round"
                        className="text-primary transition-all duration-300"
                      />
                    </svg>
                    <span className="absolute text-lg font-bold tabular-nums text-primary">
                      {Math.round(progressPct)}%
                    </span>
                  </div>
                ) : (
                  <SparklesIcon className="size-10 animate-pulse text-primary/50" />
                )}
                <span className="text-[11px] font-medium uppercase tracking-wider text-primary/60">
                  {progressPct != null
                    ? `Step ${step}/${totalSteps}`
                    : 'Generating'
                  }
                </span>
              </div>
            )}

            {/* Progress bar (when actively generating) */}
            {progressPct != null && !isQueued && (
              <div className="w-full max-w-[200px]">
                <div className="h-1 w-full overflow-hidden rounded-full bg-primary/10">
                  <div
                    className="h-full rounded-full bg-primary/60 transition-all duration-300"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
                <div className="mt-1 text-center font-mono text-[10px] text-muted-foreground/40">
                  {step}/{totalSteps}
                </div>
              </div>
            )}

            {/* Prompt echo — styled distinctly from the editable prompt */}
            <p className="max-w-[280px] text-center text-[11px] italic leading-relaxed text-muted-foreground/30">
              &ldquo;{prompt.length > 120 ? prompt.slice(0, 120) + '\u2026' : prompt}&rdquo;
            </p>
          </div>

          {/* Cancel button — only shown when queued, not during active generation */}
          {onCancel && isQueued && (
            <button
              type="button"
              onClick={onCancel}
              className="absolute right-3 top-3 flex items-center gap-1 rounded-lg border border-border/30 bg-background/40 px-2 py-1 text-[10px] text-muted-foreground/50 backdrop-blur transition-colors hover:border-destructive/50 hover:text-destructive"
            >
              <XIcon className="size-3" />
              Cancel
            </button>
          )}
        </div>

        {/* Details below card */}
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground/30">
          <span>{modelName}</span>
          <span>&middot;</span>
          <span className="font-mono">{width}×{height}</span>
          {expectedCount > 1 && (
            <>
              <span>&middot;</span>
              <span>{expectedCount}x</span>
            </>
          )}
        </div>

      </div>
    )
  }

  // ── Grid or single layout ──
  if (isGrid) {
    return (
      <div
        className="grid gap-1.5 rounded-lg"
        style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)` }}
      >
        {Array.from({ length: Math.max(images.length, isGenerating ? expectedCount : 0) }).map(
          (_, i) => {
            const img = images[i]
            if (img) {
              return (
                <div
                  key={img.url}
                  className="group relative overflow-hidden rounded-md border border-border/40 bg-secondary/20"
                  style={{ aspectRatio }}
                >
                  <img
                    src={img.url}
                    alt={`Generated ${i + 1}`}
                    className="h-full w-full cursor-pointer object-cover transition-transform hover:scale-[1.02]"
                    onClick={() => onImageClick?.(img)}
                  />
                  {img.seed != null && (
                    <span className="absolute bottom-1 right-1 rounded bg-background/80 px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
                      seed: {img.seed}
                    </span>
                  )}
                </div>
              )
            }
            // Skeleton placeholder
            return (
              <div
                key={`skeleton-${i}`}
                className="relative flex items-center justify-center overflow-hidden rounded-md border border-primary/10 bg-primary/5"
                style={{ aspectRatio }}
              >
                {isGenerating && (
                  <LoaderCircleIcon className="size-6 animate-spin text-primary/20" />
                )}
                <div className="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-primary/5 to-transparent" />
              </div>
            )
          },
        )}
      </div>
    )
  }

  // ── Single image — scales down to fit the available space without cropping ──
  const img = images[0]
  const isTall = aspectRatio < 1
  return (
    <div
      className="group relative flex items-center justify-center"
      style={{
        maxHeight: '100%',
        maxWidth: '100%',
        width: isTall ? 'auto' : '100%',
        height: isTall ? '100%' : 'auto',
      }}
    >
      {img ? (
        <>
          <img
            src={img.url}
            alt="Generated"
            className="max-h-full max-w-full cursor-pointer rounded-lg object-contain"
            style={{ aspectRatio }}
            onClick={() => onImageClick?.(img)}
          />
          {img.seed != null && (
            <span className="absolute bottom-2 right-2 rounded bg-background/80 px-2 py-1 font-mono text-[10px] text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
              seed: {img.seed}
            </span>
          )}
        </>
      ) : (
        <div
          className="flex items-center justify-center rounded-lg border border-primary/10 bg-primary/5"
          style={{ aspectRatio, maxHeight: '100%', maxWidth: isTall ? 'none' : 500, width: isTall ? 'auto' : '100%', height: isTall ? '100%' : 'auto' }}
        >
          <LoaderCircleIcon className="size-8 animate-spin text-primary/20" />
        </div>
      )}
    </div>
  )
}
