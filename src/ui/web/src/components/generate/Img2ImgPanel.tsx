import { useCallback, useRef } from 'react'
import { ImageIcon, UploadIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

export function Img2ImgPanel({ form, setForm }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return
      const reader = new FileReader()
      reader.onloadend = () => {
        setForm((prev) => ({
          ...prev,
          mode: 'img2img',
          init_image: reader.result as string,
          init_image_file: file,
        }))
      }
      reader.readAsDataURL(file)
    },
    [setForm],
  )

  const clearImage = () => {
    setForm((prev) => ({
      ...prev,
      mode: 'txt2img',
      init_image: null,
      init_image_file: null,
    }))
    if (fileRef.current) fileRef.current.value = ''
  }

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      handleFile(file)
    },
    [handleFile],
  )

  return (
    <div className="space-y-2.5">
      {/* Section label removed — handled by parent CollapsibleSection */}

      {/* Drop zone / preview */}
      {form.init_image ? (
        <div className="relative">
          <img
            src={form.init_image}
            alt="Init image"
            className="h-32 w-full rounded-md border border-border/60 object-contain bg-secondary/20"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1 size-6 bg-background/80 backdrop-blur"
            onClick={clearImage}
          >
            <XIcon className="size-3" />
          </Button>
        </div>
      ) : (
        <div
          className="flex h-24 cursor-pointer flex-col items-center justify-center gap-1.5 rounded-md border border-dashed border-border/60 bg-secondary/10 transition-colors hover:border-primary/40 hover:bg-secondary/20"
          onClick={() => fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <UploadIcon className="size-5 text-muted-foreground/40" />
          <span className="text-[10px] text-muted-foreground/50">
            Drop image or click to upload
          </span>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {/* Denoise strength (only when image is set) */}
      {form.init_image && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground/70">Denoise strength</span>
            <span className="font-mono text-[10px] text-muted-foreground">
              {form.denoise_strength.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[form.denoise_strength]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) =>
              setForm((prev) => ({ ...prev, denoise_strength: v ?? prev.denoise_strength }))
            }
          />
          <p className="text-[10px] text-muted-foreground/40">
            0 = no change, 1 = fully regenerate
          </p>
        </div>
      )}

      {!form.init_image && (
        <p className="flex items-center gap-1.5 text-[10px] text-muted-foreground/40">
          <ImageIcon className="size-3" />
          Upload an image to use img2img mode
        </p>
      )}
    </div>
  )
}
