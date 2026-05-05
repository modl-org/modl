import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { AlertTriangleIcon, Loader2Icon, XIcon } from 'lucide-react'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import { api, type GeneratedImage } from '../../api'
import { STALE_REALTIME } from '@/lib/query-keys'
import { LazyImage } from '../LazyImage'

// ---------------------------------------------------------------------------
// Session strip — unified queue + output timeline at the bottom of the canvas
// ---------------------------------------------------------------------------

export type SessionItemStatus = 'queued' | 'active' | 'completed' | 'error'

export type SessionItem = {
  id: string
  status: SessionItemStatus
  prompt: string
  model_id: string
  job_type: 'generate' | 'edit'
  batch_count: number
  images?: Array<{ url: string; seed?: number; path?: string }>
  error?: string
  step?: number
  totalSteps?: number
}

type Props = {
  sessionItems: SessionItem[]
  onSessionSelect?: (item: SessionItem) => void
  onHistorySelect?: (image: GeneratedImage) => void
  onCancelQueued?: (id: string) => void
  activePath?: string | null
  activeSessionId?: string | null
}

const THUMB = 48

export function SessionStrip({
  sessionItems,
  onSessionSelect,
  onHistorySelect,
  onCancelQueued,
  activePath,
  activeSessionId,
}: Props) {
  const { data: outputs = [] } = useQuery({
    queryKey: ['outputs'],
    queryFn: api.outputs,
    staleTime: STALE_REALTIME,
    refetchInterval: 15_000,
  })

  const recentImages = useMemo(() => {
    // Collect paths already shown as completed session cards to avoid duplicates
    const sessionPaths = new Set<string>()
    for (const item of sessionItems) {
      if (item.status === 'completed' && item.images) {
        for (const img of item.images) {
          if (img.path) sessionPaths.add(img.path.replace(/^\/files\//, ''))
        }
      }
    }

    const all: GeneratedImage[] = []
    for (const group of outputs) {
      for (const img of group.images) {
        if (!sessionPaths.has(img.path)) all.push(img)
      }
    }
    all.sort((a, b) => b.modified - a.modified)
    return all.slice(0, 30)
  }, [outputs, sessionItems])

  const hasSession = sessionItems.length > 0
  const hasHistory = recentImages.length > 0

  if (!hasSession && !hasHistory) return null

  return (
    <ScrollArea className="w-full">
      <div className="flex items-center gap-1 py-1">
        {/* Session items */}
        {sessionItems.map((item) => (
          <SessionCard
            key={item.id}
            item={item}
            isActive={activeSessionId === item.id}
            activePath={activePath}
            onSelect={() => onSessionSelect?.(item)}
            onCancel={() => onCancelQueued?.(item.id)}
          />
        ))}

        {/* Divider */}
        {hasSession && hasHistory && (
          <div className="mx-1 h-8 w-px shrink-0 bg-border/30" />
        )}

        {/* History label */}
        {hasHistory && (
          <span className="mr-1 shrink-0 rounded bg-secondary/40 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-widest text-muted-foreground/50">
            History
          </span>
        )}

        {/* History thumbnails — grouped by prompt with subtle dividers */}
        {recentImages.map((img, idx) => {
          const isActive = activePath === img.path
          const prevImg = recentImages[idx - 1]
          const showDivider = idx > 0 && prevImg && prevImg.prompt !== img.prompt
          return (
            <div key={img.path} className="flex shrink-0 items-center">
              {showDivider && (
                <div className="mx-0.5 h-6 w-px bg-border/20" />
              )}
              <button
                type="button"
                className={`relative shrink-0 overflow-hidden rounded transition-transform ${
                  isActive
                    ? 'scale-105 ring-2 ring-primary/70'
                    : 'opacity-50 ring-1 ring-border/20 hover:opacity-80 hover:ring-border/40'
                }`}
                style={{ width: THUMB, height: THUMB }}
                onClick={() => onHistorySelect?.(img)}
                title={img.prompt ?? img.filename}
              >
                <LazyImage
                  src={`/files/${img.path}`}
                  alt={img.prompt ?? img.filename}
                  className="h-full w-full"
                  thumbWidth={THUMB * 2}
                />
              </button>
            </div>
          )
        })}
      </div>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  )
}

// ---------------------------------------------------------------------------

function SessionCard({
  item,
  isActive,
  activePath,
  onSelect,
  onCancel,
}: {
  item: SessionItem
  isActive: boolean
  activePath?: string | null
  onSelect: () => void
  onCancel: () => void
}) {
  const progressPct =
    item.step != null && item.totalSteps != null && item.totalSteps > 0
      ? Math.min(100, (item.step / item.totalSteps) * 100)
      : undefined

  // Completed — thumbnail
  if (item.status === 'completed' && item.images && item.images.length > 0) {
    const firstImg = item.images[0]
    const imgPath = firstImg.path?.replace(/^\/files\//, '')
    const active = isActive || activePath === imgPath
    return (
      <button
        type="button"
        className={`group relative shrink-0 overflow-hidden rounded ${
          active
            ? 'ring-2 ring-primary/70'
            : 'ring-1 ring-border/40 hover:ring-border/70'
        }`}
        style={{ width: THUMB, height: THUMB }}
        onClick={onSelect}
        title={item.prompt}
      >
        <LazyImage src={firstImg.url} alt={item.prompt} className="h-full w-full" />
        {item.images.length > 1 && (
          <span className="absolute bottom-0 right-0 rounded-tl bg-black/60 px-1 font-mono text-[8px] text-white/80">
            {item.images.length}
          </span>
        )}
      </button>
    )
  }

  // Active — spinner + progress
  if (item.status === 'active') {
    return (
      <div
        className="relative flex shrink-0 items-center justify-center overflow-hidden rounded ring-1 ring-primary/50 bg-primary/5"
        style={{ width: THUMB, height: THUMB }}
        title={item.prompt}
      >
        {progressPct != null ? (
          <svg className="size-5 -rotate-90" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-border/30" />
            <circle
              cx="10" cy="10" r="8" fill="none" stroke="currentColor" strokeWidth="1.5"
              strokeDasharray={`${(progressPct / 100) * 50.3} 50.3`}
              strokeLinecap="round"
              className="text-primary transition-all duration-300"
            />
          </svg>
        ) : (
          <Loader2Icon className="size-4 animate-spin text-primary/60" />
        )}
        {/* Bottom progress bar */}
        {progressPct != null && (
          <div className="absolute inset-x-0 bottom-0 h-0.5 bg-border/20">
            <div className="h-full bg-primary transition-all duration-300" style={{ width: `${progressPct}%` }} />
          </div>
        )}
      </div>
    )
  }

  // Queued — dim placeholder with cancel
  if (item.status === 'queued') {
    return (
      <div
        className="group relative flex shrink-0 items-center justify-center overflow-hidden rounded ring-1 ring-border/20 bg-secondary/10"
        style={{ width: THUMB, height: THUMB }}
        title={item.prompt}
      >
        <div className="size-1.5 rounded-full bg-muted-foreground/20" />
        <button
          type="button"
          onClick={(e) => { e.stopPropagation(); onCancel() }}
          className="absolute inset-0 flex items-center justify-center bg-background/60 opacity-0 transition-opacity group-hover:opacity-100"
        >
          <XIcon className="size-3 text-muted-foreground hover:text-destructive" />
        </button>
      </div>
    )
  }

  // Error
  if (item.status === 'error') {
    return (
      <button
        type="button"
        className="relative flex shrink-0 items-center justify-center overflow-hidden rounded ring-1 ring-destructive/30 bg-destructive/5"
        style={{ width: THUMB, height: THUMB }}
        onClick={onSelect}
        title={item.error ?? 'Failed'}
      >
        <AlertTriangleIcon className="size-3.5 text-destructive/50" />
      </button>
    )
  }

  return null
}
