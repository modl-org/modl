import { useState } from 'react'
import { cn } from '@/lib/utils'

type Props = {
  src: string
  alt: string
  className?: string
  onClick?: () => void
}

export function LazyImage({ src, alt, className, onClick }: Props) {
  const [loaded, setLoaded] = useState(false)
  const [errored, setErrored] = useState(false)

  return (
    <div className={cn('relative overflow-hidden bg-secondary/40', className)}>
      {/* Skeleton shimmer — visible until image loads */}
      {!loaded && !errored && (
        <div className="absolute inset-0 animate-pulse bg-secondary/60" />
      )}

      {errored ? (
        <div className="absolute inset-0 flex items-center justify-center text-[10px] text-muted-foreground/50">
          ✕
        </div>
      ) : (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          onError={() => setErrored(true)}
          onClick={onClick}
          className={cn(
            'h-full w-full object-cover transition-opacity duration-300',
            loaded ? 'opacity-100' : 'opacity-0',
            onClick && 'cursor-pointer',
          )}
        />
      )}
    </div>
  )
}
