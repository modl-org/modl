import { useCallback, useEffect, type Dispatch, type SetStateAction } from 'react'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import type { GeneratedImage } from '../api'
import { randomSeed, type GenerateFormState } from './generate/generate-state'

type AppTab = 'train' | 'generate' | 'outputs' | 'datasets'

type Props = {
  image: GeneratedImage | null
  onClose: () => void
  setForm?: Dispatch<SetStateAction<GenerateFormState>>
  setActiveTab?: (tab: AppTab) => void
  /** All images in current filtered view, for prev/next navigation */
  allImages?: GeneratedImage[]
  /** Navigate to another image */
  onNavigate?: (image: GeneratedImage) => void
  /** Toggle favourite */
  onToggleFavorite?: (path: string) => void
}

function valueForDisplay(value: unknown): string {
  if (value === undefined || value === null || value === '') return '—'
  return String(value)
}

function metadataRows(image: GeneratedImage): Array<[string, unknown]> {
  return [
    ['Filename', image.filename],
    ['Path', image.path],
    ['Prompt', image.prompt],
    ['Model', image.base_model_id],
    ['LoRA', image.lora_name],
    ['LoRA strength', image.lora_strength],
    ['Seed', image.seed],
    ['Steps', image.steps],
    ['Guidance', image.guidance],
    ['Width', image.width],
    ['Height', image.height],
    ['Generated with', image.generated_with],
    ['Artifact ID', image.artifact_id],
    ['Job ID', image.job_id],
  ]
}

function openAsRecipe(
  image: GeneratedImage,
  setForm: Dispatch<SetStateAction<GenerateFormState>>,
  setActiveTab: (tab: AppTab) => void,
) {
  setForm((prev) => ({
    ...prev,
    prompt: image.prompt ?? '',
    base_model_id: image.base_model_id ?? prev.base_model_id,
    loras: image.lora_name ? [{ id: image.lora_name, name: image.lora_name, strength: image.lora_strength ?? 1.0 }] : [],
    seed: image.seed ?? randomSeed(),
    steps: image.steps ?? 20,
    guidance: image.guidance ?? 3.5,
    width: image.width ?? 1024,
    height: image.height ?? 1024,
  }))
  setActiveTab('generate')
}

export function ImageDetail({ image, onClose, setForm, setActiveTab, allImages, onNavigate, onToggleFavorite }: Props) {
  const currentIndex = image && allImages ? allImages.findIndex((img) => img.path === image.path) : -1
  const prevImage = allImages && currentIndex > 0 ? allImages[currentIndex - 1] : null
  const nextImage = allImages && currentIndex >= 0 && currentIndex < allImages.length - 1 ? allImages[currentIndex + 1] : null

  const goPrev = useCallback(() => {
    if (prevImage && onNavigate) onNavigate(prevImage)
  }, [prevImage, onNavigate])

  const goNext = useCallback(() => {
    if (nextImage && onNavigate) onNavigate(nextImage)
  }, [nextImage, onNavigate])

  useEffect(() => {
    if (!image) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goPrev()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        goNext()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [image, goPrev, goNext])

  return (
    <Dialog open={Boolean(image)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[96vw] gap-0 p-0 sm:max-w-5xl">
        {image ? (
          <div className="grid max-h-[88vh] gap-0 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="relative overflow-auto bg-black/60 p-4">
              {/* Navigation arrows */}
              {prevImage ? (
                <button
                  type="button"
                  onClick={goPrev}
                  className="absolute left-2 top-1/2 z-10 -translate-y-1/2 rounded-full bg-black/50 p-1.5 text-white/80 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
                >
                  <ChevronLeft className="size-5" />
                </button>
              ) : null}
              {nextImage ? (
                <button
                  type="button"
                  onClick={goNext}
                  className="absolute right-2 top-1/2 z-10 -translate-y-1/2 rounded-full bg-black/50 p-1.5 text-white/80 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
                >
                  <ChevronRight className="size-5" />
                </button>
              ) : null}

              <img
                src={`/files/${image.path}`}
                alt={image.filename}
                className="mx-auto max-h-[80vh] rounded-md object-contain"
              />

              {/* Position indicator */}
              {allImages && allImages.length > 1 ? (
                <div className="mt-2 text-center text-xs text-white/50">
                  {currentIndex + 1} / {allImages.length}
                </div>
              ) : null}
            </div>

            <div className="flex min-h-0 flex-col border-l border-border/70 bg-card">
              <DialogHeader className="px-4 pt-4">
                <DialogTitle className="flex items-center gap-2 text-base">
                  Image Metadata
                  {onToggleFavorite ? (
                    <button
                      type="button"
                      className="text-lg leading-none transition-transform hover:scale-110"
                      onClick={() => onToggleFavorite(image.path)}
                      title={image.favorited ? 'Remove from favorites' : 'Add to favorites'}
                    >
                      {image.favorited ? '⭐' : '☆'}
                    </button>
                  ) : null}
                </DialogTitle>
                <DialogDescription>{image.filename}</DialogDescription>
              </DialogHeader>

              <ScrollArea className="min-h-0 flex-1 px-4 py-3">
                <div className="grid grid-cols-[120px_minmax(0,1fr)] gap-x-3 gap-y-2 text-xs">
                  {metadataRows(image).map(([label, value]) => (
                    <div key={label} className="contents">
                      <div className="text-muted-foreground">
                        {label}
                      </div>
                      <div className="break-words font-mono text-foreground">
                        {valueForDisplay(value)}
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>

              <DialogFooter className="gap-2 border-t border-border/70 px-4 py-3 sm:justify-start">
                <div className="flex items-center gap-2">
                  {prevImage ? (
                    <Button type="button" size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={goPrev} title="Previous (←)">
                      <ChevronLeft className="size-4" />
                    </Button>
                  ) : null}
                  {nextImage ? (
                    <Button type="button" size="sm" variant="ghost" className="h-8 w-8 p-0" onClick={goNext} title="Next (→)">
                      <ChevronRight className="size-4" />
                    </Button>
                  ) : null}
                </div>
                <Button
                  type="button"
                  size="sm"
                  disabled={!setForm || !setActiveTab}
                  onClick={() => {
                    if (setForm && setActiveTab) {
                      openAsRecipe(image, setForm, setActiveTab)
                      onClose()
                    }
                  }}
                >
                  Open as recipe
                </Button>
                <Button type="button" size="sm" variant="outline" onClick={onClose}>
                  Close
                </Button>
              </DialogFooter>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
