import { useCallback, useState } from 'react'
import { ImagePlus, X } from 'lucide-react'
import { cn } from '@/lib/utils'

type Props = {
  files: File[]
  onFilesChange: (files: File[]) => void
  disabled?: boolean
}

export function UploadZone({ files, onFilesChange, disabled }: Props) {
  const [dragActive, setDragActive] = useState(false)

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setDragActive(false)
      if (disabled) return
      const dropped = Array.from(e.dataTransfer.files).filter((f) =>
        f.type.startsWith('image/'),
      )
      if (dropped.length > 0) {
        onFilesChange([...files, ...dropped])
      }
    },
    [files, onFilesChange, disabled],
  )

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled) return
      const selected = Array.from(e.target.files ?? [])
      if (selected.length > 0) {
        onFilesChange([...files, ...selected])
      }
      e.target.value = ''
    },
    [files, onFilesChange, disabled],
  )

  const removeFile = useCallback(
    (index: number) => {
      onFilesChange(files.filter((_, i) => i !== index))
    },
    [files, onFilesChange],
  )

  const previews = files.map((f) => URL.createObjectURL(f))

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <label
        onDragOver={(e) => {
          e.preventDefault()
          setDragActive(true)
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        className={cn(
          'flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-8 cursor-pointer transition-colors',
          dragActive
            ? 'border-primary bg-primary/5'
            : 'border-border/60 hover:border-primary/40 hover:bg-primary/5',
          disabled && 'opacity-50 cursor-not-allowed',
        )}
      >
        <ImagePlus className="h-8 w-8 text-muted-foreground" />
        <p className="text-sm text-muted-foreground">
          Drop photos here or{' '}
          <span className="text-primary font-medium">browse</span>
        </p>
        <p className="text-xs text-muted-foreground/60">JPG, PNG, WebP</p>
        <input
          type="file"
          accept="image/*"
          multiple
          className="hidden"
          onChange={handleFileInput}
          disabled={disabled}
        />
      </label>

      {/* Thumbnails */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, i) => (
            <div
              key={`${file.name}-${i}`}
              className="group relative h-16 w-16 shrink-0 overflow-hidden rounded-lg border border-border/40"
            >
              <img
                src={previews[i]}
                alt={file.name}
                className="h-full w-full object-cover"
                onLoad={() => URL.revokeObjectURL(previews[i])}
              />
              {!disabled && (
                <button
                  onClick={() => removeFile(i)}
                  className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-destructive text-destructive-foreground opacity-0 transition-opacity group-hover:opacity-100"
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
          ))}
          <p className="flex items-center px-2 text-xs text-muted-foreground">
            {files.length} photo{files.length !== 1 ? 's' : ''}
          </p>
        </div>
      )}
    </div>
  )
}
