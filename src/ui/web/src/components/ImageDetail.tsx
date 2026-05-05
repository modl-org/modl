import { useCallback, useEffect, useState } from 'react'
import { ArrowUpIcon, ChevronLeft, ChevronRight, EraserIcon, Info, LoaderCircleIcon, PencilIcon, Play, Trash2, X } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { modelColor } from '@/lib/utils'
import { api, type GeneratedImage } from '../api'
import { useAppNav } from '../contexts/FormContext'

type Props = {
  image: GeneratedImage | null
  onClose: () => void
  allImages?: GeneratedImage[]
  onNavigate?: (image: GeneratedImage) => void
  onToggleFavorite?: (path: string) => void
  onEditImage?: (imageUrl: string, serverPath: string) => void
  onDeleteImage?: (image: GeneratedImage) => void
}

function val(value: unknown): string {
  if (value === undefined || value === null || value === '') return '\u2014'
  return String(value)
}

export function ImageDetail({ image, onClose, allImages, onNavigate, onToggleFavorite, onEditImage, onDeleteImage }: Props) {
  const { useAsRecipe } = useAppNav()
  const [upscaling, setUpscaling] = useState(false)
  const [removingBg, setRemovingBg] = useState(false)
  const [showPanel, setShowPanel] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)

  // Reset load state when image changes
  useEffect(() => {
    setImageLoaded(false)
  }, [image?.path])

  const handleUpscale = useCallback(async () => {
    if (!image || upscaling) return
    setUpscaling(true)
    try {
      const result = await api.upscale(image.path)
      if (result.status === 'completed') {
        toast.success('Upscaled successfully')
      } else {
        toast.error(result.error ?? 'Upscale failed')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Upscale failed')
    } finally {
      setUpscaling(false)
    }
  }, [image, upscaling])

  const handleRemoveBg = useCallback(async () => {
    if (!image || removingBg) return
    setRemovingBg(true)
    try {
      const result = await api.removeBg(image.path)
      if (result.status === 'completed') {
        toast.success('Background removed')
      } else {
        toast.error(result.error ?? 'Background removal failed')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Background removal failed')
    } finally {
      setRemovingBg(false)
    }
  }, [image, removingBg])

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
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        goPrev()
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        goNext()
      } else if (e.key === 'f' || e.key === 'F') {
        e.preventDefault()
        if (onToggleFavorite) onToggleFavorite(image.path)
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault()
        useAsRecipe(image)
        onClose()
      } else if (e.key === 'e' || e.key === 'E') {
        e.preventDefault()
        if (onEditImage) {
          onEditImage(`/files/${image.path}`, image.path)
          onClose()
        }
      } else if (e.key === 'd' || e.key === 'D') {
        e.preventDefault()
        if (onDeleteImage) onDeleteImage(image)
      } else if (e.key === 'i' || e.key === 'I') {
        e.preventDefault()
        setShowPanel((prev) => !prev)
      } else if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [image, goPrev, goNext, onToggleFavorite, onDeleteImage, useAsRecipe, onEditImage, onClose])

  const mColor = modelColor(image?.base_model_id)

  return (
    <Dialog open={Boolean(image)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-[96vw] gap-0 overflow-hidden rounded-xl border border-white/[0.06] bg-[#0c0c12] p-0 shadow-2xl sm:max-w-5xl [&>button]:hidden">
        <DialogTitle className="sr-only">{image?.filename ?? 'Image'}</DialogTitle>
        {image ? (
          <div className="flex max-h-[92vh]">
            {/* ── Image area ─────────────────────────────────────── */}
            <div className="relative flex min-w-0 flex-1 items-center justify-center overflow-hidden transition-all duration-300">
              {/* Navigation arrows */}
              {prevImage ? (
                <button
                  type="button"
                  onClick={goPrev}
                  className="absolute left-2 top-1/2 z-20 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white/70 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
                >
                  <ChevronLeft className="size-5" />
                </button>
              ) : null}
              {nextImage ? (
                <button
                  type="button"
                  onClick={goNext}
                  className="absolute right-2 top-1/2 z-20 -translate-y-1/2 rounded-full bg-black/50 p-2 text-white/70 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
                >
                  <ChevronRight className="size-5" />
                </button>
              ) : null}

              {/* Loading spinner — visible while image loads */}
              {!imageLoaded && (
                <div className="absolute inset-0 z-10 flex items-center justify-center">
                  <LoaderCircleIcon className="size-6 animate-spin text-white/30" />
                </div>
              )}

              <img
                key={image.path}
                src={`/files/${image.path}?w=1920`}
                alt={image.filename}
                onLoad={() => setImageLoaded(true)}
                className={`max-h-[88vh] max-w-full rounded object-contain p-2 transition-opacity duration-200 ${
                  imageLoaded ? 'opacity-100' : 'opacity-0'
                }`}
              />

              {/* Top-right: close only */}
              <button
                type="button"
                onClick={onClose}
                className="absolute top-3 right-3 z-20 rounded-full bg-black/50 p-1.5 text-white/70 backdrop-blur transition-colors hover:bg-black/70 hover:text-white"
                title="Close (Esc)"
              >
                <X className="size-4" />
              </button>

              {/* Position counter */}
              {allImages && allImages.length > 1 ? (
                <div className="absolute bottom-14 left-1/2 z-20 -translate-x-1/2 rounded-full bg-black/40 px-2.5 py-1 text-[10px] tabular-nums text-white/40 backdrop-blur">
                  {currentIndex + 1} / {allImages.length}
                </div>
              ) : null}

              {/* Keyboard hints */}
              <div className="absolute top-3 left-3 z-20 rounded-full bg-black/30 px-2.5 py-1 text-[10px] text-white/20 backdrop-blur">
                ← → F R E D I
              </div>

              {/* ── Bottom toolbar — single control surface ──── */}
              <div className="absolute bottom-4 left-1/2 z-20 flex -translate-x-1/2 items-center gap-0.5 rounded-full bg-black/60 px-2 py-1.5 backdrop-blur-md">
                {/* Star */}
                {onToggleFavorite && (
                  <>
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      className="h-7 w-7 p-0 text-base leading-none hover:bg-white/15"
                      onClick={() => onToggleFavorite(image.path)}
                      title="Favorite (F)"
                    >
                      {image.favorited ? '⭐' : '☆'}
                    </Button>
                    <div className="mx-0.5 h-4 w-px bg-white/15" />
                  </>
                )}

                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2.5 text-xs text-white/90 hover:bg-white/15 hover:text-white"
                  onClick={() => {
                    useAsRecipe(image)
                    onClose()
                  }}
                  title="Open as recipe (R)"
                >
                  <Play className="mr-1 size-3" />
                  Recipe
                </Button>
                {onEditImage && (
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-7 px-2.5 text-xs text-white/90 hover:bg-white/15 hover:text-white"
                    onClick={() => {
                      onEditImage(`/files/${image.path}`, image.path)
                      onClose()
                    }}
                    title="Send to Edit mode"
                  >
                    <PencilIcon className="mr-1 size-3" />
                    Edit
                  </Button>
                )}
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2.5 text-xs text-white/90 hover:bg-white/15 hover:text-white"
                  disabled={upscaling}
                  onClick={handleUpscale}
                  title="Upscale 4x"
                >
                  {upscaling ? <LoaderCircleIcon className="mr-1 size-3 animate-spin" /> : <ArrowUpIcon className="mr-1 size-3" />}
                  4x
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2.5 text-xs text-white/90 hover:bg-white/15 hover:text-white"
                  disabled={removingBg}
                  onClick={handleRemoveBg}
                  title="Remove background"
                >
                  {removingBg ? <LoaderCircleIcon className="mr-1 size-3 animate-spin" /> : <EraserIcon className="mr-1 size-3" />}
                  BG
                </Button>

                <div className="mx-0.5 h-4 w-px bg-white/15" />

                {/* Info toggle — highlighted when panel open */}
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className={`h-7 w-7 p-0 transition-colors hover:bg-white/15 ${
                    showPanel
                      ? 'bg-white/20 text-white'
                      : 'text-white/70 hover:text-white'
                  }`}
                  onClick={() => setShowPanel((prev) => !prev)}
                  title="Toggle details (I)"
                >
                  <Info className="size-3.5" />
                </Button>
                {onDeleteImage && (
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    className="h-7 w-7 p-0 text-white/70 hover:bg-red-500/30 hover:text-red-300"
                    onClick={() => onDeleteImage(image)}
                    title="Delete (D)"
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                )}
              </div>
            </div>

            {/* ── Details panel — slides in within modal bounds ── */}
            <div
              className={`overflow-hidden transition-all duration-300 ease-out ${
                showPanel ? 'w-80 border-l border-white/[0.06]' : 'w-0'
              }`}
            >
              <div className="flex h-full w-80 flex-col bg-[#0e0e16]">
                {/* Panel header with nav */}
                <div className="flex shrink-0 items-center justify-between border-b border-white/[0.06] px-4 py-2.5">
                  <h3 className="text-xs font-semibold text-white/80">Details</h3>
                  {allImages && allImages.length > 1 && (
                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        className="rounded p-0.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white/80 disabled:opacity-20"
                        onClick={goPrev}
                        disabled={!prevImage}
                      >
                        <ChevronLeft className="size-3.5" />
                      </button>
                      <span className="min-w-[3rem] text-center text-[10px] tabular-nums text-white/40">
                        {currentIndex + 1} / {allImages.length}
                      </span>
                      <button
                        type="button"
                        className="rounded p-0.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white/80 disabled:opacity-20"
                        onClick={goNext}
                        disabled={!nextImage}
                      >
                        <ChevronRight className="size-3.5" />
                      </button>
                    </div>
                  )}
                </div>

                <ScrollArea className="flex-1">
                  <div className="space-y-0">
                    {/* Prompt — most prominent */}
                    <div className="border-b border-white/[0.06] px-4 py-3">
                      <p className="mb-1.5 text-[10px] font-medium uppercase tracking-wider text-white/30">Prompt</p>
                      <p className="text-[13px] leading-relaxed text-white/90">
                        {image.prompt || <span className="text-white/20">—</span>}
                      </p>
                    </div>

                    {/* Model + LoRA */}
                    <div className="border-b border-white/[0.06] px-4 py-3">
                      <div className="space-y-2">
                        <div>
                          <p className="text-[10px] text-white/30">Model</p>
                          <p className="flex items-center gap-1.5 font-mono text-xs text-white/80">
                            <span
                              className="inline-block size-2 shrink-0 rounded-full"
                              style={{ backgroundColor: mColor }}
                            />
                            {val(image.base_model_id)}
                          </p>
                        </div>
                        {image.lora_name && (
                          <div>
                            <p className="text-[10px] text-white/30">LoRA</p>
                            <p className="font-mono text-xs text-white/80">
                              {image.lora_name}
                              <span className="ml-1.5 text-white/40">@ {image.lora_strength ?? 1.0}</span>
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Params — compact 2-col grid */}
                    <div className="border-b border-white/[0.06] px-4 py-3">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
                        <div>
                          <p className="text-[10px] text-white/30">Seed</p>
                          <p className="font-mono text-xs text-white/70">{val(image.seed)}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-white/30">Steps</p>
                          <p className="font-mono text-xs text-white/70">{val(image.steps)}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-white/30">Guidance</p>
                          <p className="font-mono text-xs text-white/70">{val(image.guidance)}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-white/30">Size</p>
                          <p className="font-mono text-xs text-white/70">
                            {image.width && image.height ? `${image.width}×${image.height}` : '—'}
                          </p>
                        </div>
                      </div>
                    </div>

                    {/* Technical — collapsed by default */}
                    <details className="group px-4 py-3">
                      <summary className="flex cursor-pointer list-none items-center gap-1 text-[10px] font-medium uppercase tracking-wider text-white/25 transition-colors hover:text-white/40 [&::-webkit-details-marker]:hidden">
                        <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
                        Technical
                      </summary>
                      <div className="mt-2.5 space-y-2">
                        <div>
                          <p className="text-[10px] text-white/25">Filename</p>
                          <p className="break-all font-mono text-[11px] text-white/50">{val(image.filename)}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-white/25">Artifact ID</p>
                          <p className="break-all font-mono text-[11px] text-white/50">{val(image.artifact_id)}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-white/25">Job ID</p>
                          <p className="break-all font-mono text-[11px] text-white/50">{val(image.job_id)}</p>
                        </div>
                      </div>
                    </details>
                  </div>
                </ScrollArea>
              </div>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
