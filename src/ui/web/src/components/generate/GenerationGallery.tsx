import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import { api, type GeneratedImage } from '../../api'
import { LazyImage } from '../LazyImage'

type Props = {
  /** Called when a gallery image is clicked */
  onSelect?: (image: GeneratedImage) => void
  /** Currently highlighted image path */
  activePath?: string | null
  /** Max images to show */
  limit?: number
}

export function GenerationGallery({ onSelect, activePath, limit = 30 }: Props) {
  const { data: outputs = [] } = useQuery({
    queryKey: ['outputs'],
    queryFn: api.outputs,
    staleTime: 10_000,
    refetchInterval: 15_000,
  })

  // Flatten and take most recent images
  const recentImages = useMemo(() => {
    const all: GeneratedImage[] = []
    for (const group of outputs) {
      for (const img of group.images) {
        all.push(img)
      }
    }
    // Sort newest first by modified timestamp
    all.sort((a, b) => b.modified - a.modified)
    return all.slice(0, limit)
  }, [outputs, limit])

  if (recentImages.length === 0) {
    return null
  }

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        Recent
      </span>
      <ScrollArea className="w-full">
        <div className="flex gap-1.5 pb-2">
          {recentImages.map((img) => {
            const isActive = activePath === img.path
            const src = `/files/${img.path}`
            return (
              <button
                key={img.path}
                type="button"
                className={`relative shrink-0 overflow-hidden rounded-md border transition-all ${
                  isActive
                    ? 'border-primary ring-1 ring-primary/50'
                    : 'border-border/40 hover:border-border/70'
                }`}
                style={{ width: 64, height: 64 }}
                onClick={() => onSelect?.(img)}
                title={img.prompt ?? img.filename}
              >
                <LazyImage
                  src={src}
                  alt={img.prompt ?? img.filename}
                  className="h-full w-full"
                />
              </button>
            )
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  )
}
