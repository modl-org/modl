import React from 'react'
import { Check, Film, PencilIcon, Play, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { modelColor } from '@/lib/utils'
import type { GeneratedImage } from '../../api'
import { LazyImage } from '../LazyImage'

function isVideoFile(path: string): boolean {
  return path.endsWith('.mp4') || path.endsWith('.webm')
}

/** Shorten a model id for display: keep it under 16 chars */
function modelLabel(id: string): string {
  return id.length > 16 ? id.slice(0, 14) + '...' : id
}

export type ImageGridItemProps = {
  image: GeneratedImage
  /** Whether this image is batch-selected */
  isSelected: boolean
  /** Whether this image has the keyboard cursor focus (select mode) */
  isCursor: boolean
  /** Whether this image is showing the inline delete confirmation */
  isDeleting: boolean
  /** Whether we are in batch select mode */
  selectMode: boolean
  /** Thumbnail width hint passed to LazyImage */
  thumbWidth: number
  /** Whether a delete mutation is in flight */
  deletePending: boolean

  // Callbacks
  onImageClick: (e: React.MouseEvent) => void
  onToggleSelection: (e: React.MouseEvent) => void
  onFavorite: () => void
  onOpenAsRecipe: () => void
  onSendToEdit: () => void
  onOpenDetail: () => void
  onRequestDelete: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
}

export const ImageGridItem = React.memo(function ImageGridItem({
  image,
  isSelected,
  isCursor,
  isDeleting,
  selectMode,
  thumbWidth,
  deletePending,
  onImageClick,
  onToggleSelection,
  onFavorite,
  onOpenAsRecipe,
  onSendToEdit,
  onOpenDetail,
  onRequestDelete,
  onConfirmDelete,
  onCancelDelete,
}: ImageGridItemProps) {
  return (
    <article
      className={`group relative overflow-hidden rounded-lg ${isSelected ? 'ring-2 ring-primary' : ''} ${isCursor && !isSelected ? 'ring-2 ring-primary/40' : ''}`}
    >
      {isVideoFile(image.path) ? (
        <video
          src={`/files/${image.path}`}
          className="aspect-square object-cover cursor-pointer"
          muted
          loop
          playsInline
          onMouseEnter={e => (e.target as HTMLVideoElement).play().catch(() => {})}
          onMouseLeave={e => { const v = e.target as HTMLVideoElement; v.pause(); v.currentTime = 0 }}
          onClick={onImageClick}
        />
      ) : (
        <LazyImage
          src={`/files/${image.path}`}
          alt={image.filename}
          className="aspect-square"
          thumbWidth={thumbWidth}
          onClick={onImageClick}
        />
      )}
      {/* Hover overlay */}
      <div className="pointer-events-none absolute inset-0 bg-black/0 transition-colors group-hover:bg-black/30" />

      {/* Video badge */}
      {isVideoFile(image.path) && (
        <div className="absolute top-1.5 right-1.5 flex items-center gap-0.5 rounded bg-black/60 px-1.5 py-0.5 backdrop-blur-sm pointer-events-none">
          <Film className="size-3 text-white/90" />
          {image.fps && image.num_frames && (
            <span className="text-[9px] font-medium text-white/80">
              {(image.num_frames / image.fps).toFixed(1)}s
            </span>
          )}
        </div>
      )}

      {/* Model color dot */}
      {image.base_model_id && (
        <div
          className="absolute bottom-1.5 left-1.5 flex items-center gap-1 rounded-full bg-black/60 px-2 py-0.5 backdrop-blur-sm"
          title={image.base_model_id}
        >
          <span
            className="inline-block size-2 rounded-full"
            style={{ backgroundColor: modelColor(image.base_model_id) }}
          />
          <span className="text-[10px] font-medium leading-none text-white/90">
            {modelLabel(image.base_model_id)}
          </span>
        </div>
      )}

      {/* Select checkbox (in select mode) */}
      {selectMode && (
        <button
          type="button"
          className="absolute top-1.5 left-1.5 drop-shadow"
          onClick={onToggleSelection}
        >
          <div
            className={`flex size-5 items-center justify-center rounded border-2 transition-colors ${
              isSelected
                ? 'border-primary bg-primary'
                : 'border-white/80 bg-black/40'
            }`}
          >
            {isSelected && <Check className="size-3.5 text-white" strokeWidth={3} />}
          </div>
        </button>
      )}

      {/* Favorite button (when not in select mode) */}
      {!selectMode && (
        <button
          type="button"
          className={`absolute top-1.5 left-1.5 text-base leading-none drop-shadow transition-opacity ${
            image.favorited
              ? 'opacity-100'
              : 'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100'
          }`}
          onClick={(e) => {
            e.stopPropagation()
            onFavorite()
          }}
        >
          {image.favorited ? '\u2B50' : '\u2606'}
        </button>
      )}

      {/* Inline delete confirmation */}
      {isDeleting && (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/70 backdrop-blur-sm">
          <p className="text-xs font-medium text-white">Delete?</p>
          <div className="flex gap-1.5">
            <Button
              type="button"
              size="sm"
              variant="destructive"
              className="h-7 px-3 text-xs"
              disabled={deletePending}
              onClick={onConfirmDelete}
            >
              {deletePending ? 'Deleting...' : 'Delete'}
            </Button>
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="h-7 px-3 text-xs"
              onClick={onCancelDelete}
            >
              Cancel
            </Button>
          </div>
        </div>
      )}

      {/* Hover action buttons (not in select mode) */}
      {!selectMode && !isDeleting && (
        <div className="pointer-events-none absolute top-1.5 right-1.5 flex gap-1 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100">
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="h-6 w-6 p-0"
            title="Open as recipe"
            onClick={(e) => {
              e.stopPropagation()
              onOpenAsRecipe()
            }}
          >
            <Play className="size-3" />
          </Button>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="h-6 w-6 p-0"
            title="Send to Edit"
            onClick={(e) => {
              e.stopPropagation()
              onSendToEdit()
            }}
          >
            <PencilIcon className="size-3" />
          </Button>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            className="h-6 px-2 text-[10px]"
            onClick={onOpenDetail}
          >
            Info
          </Button>
          <Button
            type="button"
            size="sm"
            variant="destructive"
            className="h-6 w-6 p-0"
            onClick={(e) => {
              e.stopPropagation()
              onRequestDelete()
            }}
            disabled={deletePending}
          >
            <Trash2 className="size-3" />
          </Button>
        </div>
      )}
    </article>
  )
})
