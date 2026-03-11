import { useCallback, useRef } from 'react'
import { ImagePlusIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { EditImage, GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

export function EditImagesPanel({ form, setForm }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const editImages: EditImage[] = form.edit_images ?? []

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const arr = Array.from(files)
      if (arr.length === 0) return

      for (const file of arr) {
        const reader = new FileReader()
        reader.onloadend = () => {
          setForm((prev) => ({
            ...prev,
            edit_images: [
              ...(prev.edit_images ?? []),
              { type: 'file' as const, preview: reader.result as string, file },
            ],
          }))
        }
        reader.readAsDataURL(file)
      }
    },
    [setForm],
  )

  const removeImage = (index: number) => {
    setForm((prev) => ({
      ...prev,
      edit_images: (prev.edit_images ?? []).filter((_, i) => i !== index),
    }))
  }

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      addFiles(e.dataTransfer.files)
    },
    [addFiles],
  )

  return (
    <div className="space-y-2.5">
      {/* Thumbnail strip */}
      {editImages.length > 0 && (
        <div className="flex gap-2 overflow-x-auto pb-1">
          {editImages.map((img, i) => (
            <div key={i} className="relative flex-shrink-0">
              <img
                src={img.preview}
                alt={`Edit image ${i + 1}`}
                className="h-20 w-20 rounded-md border border-border/60 object-cover bg-secondary/20"
              />
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="absolute -right-1 -top-1 size-5 rounded-full bg-background/90 backdrop-blur"
                onClick={() => removeImage(i)}
              >
                <XIcon className="size-2.5" />
              </Button>
              {img.type === 'server' && (
                <span className="absolute bottom-0.5 left-0.5 rounded bg-black/60 px-1 py-0.5 text-[8px] text-white/70">
                  local
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Drop zone */}
      <div
        className="flex h-20 cursor-pointer flex-col items-center justify-center gap-1.5 rounded-md border border-dashed border-border/60 bg-secondary/10 transition-colors hover:border-primary/40 hover:bg-secondary/20"
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <ImagePlusIcon className="size-5 text-muted-foreground/40" />
        <span className="text-[10px] text-muted-foreground/50">
          {editImages.length === 0
            ? 'Drop image(s) or click to upload'
            : 'Add more images'}
        </span>
      </div>

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) addFiles(e.target.files)
          e.target.value = ''
        }}
      />

      <p className="text-[10px] text-muted-foreground/40">
        Upload source image(s) to edit. Use multiple images to compose elements across them.
      </p>
    </div>
  )
}
