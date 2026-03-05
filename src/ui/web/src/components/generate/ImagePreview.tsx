import { LoaderCircleIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

// ---------------------------------------------------------------------------
// ImagePreview — the main preview area
//
// Shows:
//   - Placeholder when idle
//   - Spinner overlay during generation
//   - Single image result
//   - Grid for multi-batch results
// ---------------------------------------------------------------------------

export type PreviewImage = {
  url: string
  seed?: number
  index?: number
}

type Props = {
  images: PreviewImage[]
  isGenerating: boolean
  /** Expected number of images (for skeleton placeholders) */
  expectedCount?: number
  width: number
  height: number
  /** 'fit' shows the full image; 'fill' shows at 100% (may overflow) */
  fitMode?: 'fit' | 'fill'
  onImageClick?: (img: PreviewImage) => void
}

export function ImagePreview({
  images,
  isGenerating,
  expectedCount = 1,
  width,
  height,
  fitMode = 'fit',
  onImageClick,
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

  if (!hasImages && !isGenerating) {
    // Idle placeholder
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

  // Grid or single layout
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
                className="relative flex items-center justify-center overflow-hidden rounded-md border border-border/30 bg-secondary/15"
                style={{ aspectRatio }}
              >
                {isGenerating && (
                  <LoaderCircleIcon className="size-6 animate-spin text-muted-foreground/30" />
                )}
                <div className="absolute inset-0 animate-pulse bg-gradient-to-r from-transparent via-secondary/20 to-transparent" />
              </div>
            )
          },
        )}
      </div>
    )
  }

  // Single image
  const img = images[0]
  return (
    <div
      className={cn(
        'relative overflow-hidden rounded-lg border border-border/40 bg-secondary/20',
        fitMode === 'fill' && 'overflow-auto',
      )}
    >
      {img ? (
        <div className="group relative" style={fitMode === 'fit' ? { aspectRatio } : undefined}>
          <img
            src={img.url}
            alt="Generated"
            className={cn(
              'cursor-pointer transition-transform hover:scale-[1.01]',
              fitMode === 'fit'
                ? 'h-full w-full object-contain'
                : 'max-w-none',
              isGenerating && 'opacity-50',
            )}
            onClick={() => onImageClick?.(img)}
          />
          {img.seed != null && (
            <span className="absolute bottom-2 right-2 rounded bg-background/80 px-2 py-1 font-mono text-[10px] text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">
              seed: {img.seed}
            </span>
          )}
        </div>
      ) : (
        <div className="flex items-center justify-center" style={{ aspectRatio }}>
          <LoaderCircleIcon className="size-8 animate-spin text-muted-foreground/30" />
        </div>
      )}

      {/* Generation overlay */}
      {isGenerating && img && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/30 backdrop-blur-[2px]">
          <LoaderCircleIcon className="size-8 animate-spin text-primary/70" />
        </div>
      )}
    </div>
  )
}
