import { useState } from 'react'
import { Download, X } from 'lucide-react'

type Props = {
  images: string[]
}

export function ResultGallery({ images }: Props) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)

  if (images.length === 0) return null

  return (
    <>
      <div className="space-y-3">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {images.map((image, i) => (
            <button
              key={image}
              onClick={() => setSelectedIndex(i)}
              className="group relative aspect-square overflow-hidden rounded-lg border border-border/40 transition-colors hover:border-primary/40"
            >
              <img
                src={`/files/${image}`}
                alt={`Generated ${i + 1}`}
                className="h-full w-full object-cover"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-black/0 transition-colors group-hover:bg-black/20" />
            </button>
          ))}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <a
            href={images.length === 1 ? `/files/${images[0]}` : '#'}
            download
            className="inline-flex items-center gap-1.5 rounded-md border border-border/60 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <Download className="h-3.5 w-3.5" />
            {images.length === 1 ? 'Download' : 'Download All'}
          </a>
        </div>
      </div>

      {/* Lightbox */}
      {selectedIndex != null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setSelectedIndex(null)}
        >
          <button
            onClick={() => setSelectedIndex(null)}
            className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
          >
            <X className="h-5 w-5" />
          </button>

          <img
            src={`/files/${images[selectedIndex]}`}
            alt={`Generated ${selectedIndex + 1}`}
            className="max-h-[85vh] max-w-[90vw] rounded-lg object-contain"
            onClick={(e) => e.stopPropagation()}
          />

          {/* Nav arrows */}
          {images.length > 1 && (
            <>
              {selectedIndex > 0 && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedIndex(selectedIndex - 1)
                  }}
                  className="absolute left-4 flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
                >
                  &larr;
                </button>
              )}
              {selectedIndex < images.length - 1 && (
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setSelectedIndex(selectedIndex + 1)
                  }}
                  className="absolute right-4 flex h-10 w-10 items-center justify-center rounded-full bg-white/10 text-white transition-colors hover:bg-white/20"
                >
                  &rarr;
                </button>
              )}
            </>
          )}
        </div>
      )}
    </>
  )
}
