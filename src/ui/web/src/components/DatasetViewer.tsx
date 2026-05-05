import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useSearch } from 'wouter'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { ScrollArea } from '@/components/ui/scroll-area'
import { api, type DatasetImage } from '../api'
import { STALE_MODERATE } from '@/lib/query-keys'
import { LazyImage } from './LazyImage'

const PAGE_SIZE = 50

type LightboxImage = DatasetImage & { datasetName: string }

export function DatasetViewer() {
  const searchString = useSearch()
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [page, setPage] = useState(0)

  // Auto-select dataset from URL param (e.g. ?tab=datasets&dataset=kids-art)
  useEffect(() => {
    const params = new URLSearchParams(searchString)
    const ds = params.get('dataset')
    if (ds) {
      setSelectedDataset(ds)
      setPage(0)
    }
  }, [searchString])
  const [lightbox, setLightbox] = useState<LightboxImage | null>(null)
  const [gridSize, setGridSize] = useState<'s' | 'm' | 'l'>('m')

  const gridClass = {
    s: 'grid-cols-3 gap-1.5 md:grid-cols-5 lg:grid-cols-7 xl:grid-cols-9',
    m: 'grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6',
    l: 'grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  }[gridSize]

  const {
    data: datasets = [],
    error: datasetsError,
    isLoading: datasetsLoading,
  } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.datasets,
    staleTime: STALE_MODERATE,
  })

  const datasetNames = datasets.map((d) => typeof d === 'string' ? d : d.name)

  const current =
    selectedDataset && datasetNames.includes(selectedDataset)
      ? selectedDataset
      : (datasetNames[0] ?? null)

  const {
    data: overview,
    error: overviewError,
    isLoading: overviewLoading,
  } = useQuery({
    queryKey: ['dataset', current, page],
    queryFn: () => api.dataset(current as string, PAGE_SIZE, page * PAGE_SIZE),
    enabled: Boolean(current),
    staleTime: STALE_MODERATE,
    placeholderData: (previousData) => previousData,
  })

  const totalPages = overview ? Math.max(1, Math.ceil(overview.image_count / PAGE_SIZE)) : 1

  return (
    <>
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">Dataset</span>
          {datasetsLoading ? (
            <span className="text-xs text-muted-foreground">Loading…</span>
          ) : datasetsError ? (
            <span className="text-xs text-destructive">Failed to load</span>
          ) : datasetNames.length === 0 ? (
            <span className="text-xs text-muted-foreground">No datasets found</span>
          ) : datasetNames.map((name) => (
            <Button
              key={name}
              type="button"
              size="sm"
              variant={name === current ? 'secondary' : 'ghost'}
              className="h-7 px-2.5 text-xs"
              onClick={() => { setSelectedDataset(name); setPage(0) }}
            >
              {name}
            </Button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          {overview ? (
            <span className="text-xs text-muted-foreground">
              {overview.image_count} images · {Math.round(overview.coverage * 100)}% captioned
            </span>
          ) : null}
          <div className="flex items-center rounded-md border border-border/50">
            {(['s', 'm', 'l'] as const).map((size) => (
              <button
                key={size}
                type="button"
                onClick={() => setGridSize(size)}
                className={`h-7 w-7 text-[10px] font-semibold uppercase transition-colors first:rounded-l-md last:rounded-r-md ${
                  gridSize === size
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {size}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      {overviewError ? (
        <p className="text-sm text-destructive">Failed to load dataset: {String(overviewError)}</p>
      ) : !overview && overviewLoading ? (
        <div className={`grid ${gridClass}`}>
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} className="aspect-square animate-pulse rounded-lg bg-secondary/50" />
          ))}
        </div>
      ) : !overview ? (
        <p className="py-8 text-center text-sm text-muted-foreground">Select a dataset above.</p>
      ) : (
        <>
          <div className={`grid ${gridClass}`}>
            {overview.images.map((image) => (
              <article
                key={image.image_url}
                className="overflow-hidden rounded-lg border border-border/50"
              >
                <LazyImage
                  src={`/files/${image.image_url}?w=400`}
                  alt={image.filename}
                  className="aspect-square"
                  onClick={() => setLightbox({ ...image, datasetName: overview.name })}
                />
                {image.caption ? (
                  <div className="px-2 pt-2 text-xs leading-snug text-muted-foreground">
                    {image.caption}
                  </div>
                ) : null}
                <div className="px-2 pb-2 pt-1 font-mono text-[10px] text-muted-foreground/60">
                  {image.filename}
                </div>
              </article>
            ))}
          </div>

          {totalPages > 1 ? (
            <div className="flex items-center justify-center gap-2 pt-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </Button>
              <span className="text-xs text-muted-foreground">
                Page {page + 1} / {totalPages}
              </span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          ) : null}
        </>
      )}
    </div>

    {/* Lightbox */}
    <Dialog open={Boolean(lightbox)} onOpenChange={(open) => !open && setLightbox(null)}>
      <DialogContent className="max-w-[96vw] gap-0 p-0 sm:max-w-4xl">
        {lightbox ? (
          <div className="grid max-h-[90vh] gap-0 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div className="flex items-center justify-center overflow-auto bg-black/70 p-4">
              <img
                src={`/files/${lightbox.image_url}?w=1920`}
                alt={lightbox.filename}
                className="max-h-[80vh] rounded object-contain"
              />
            </div>
            <div className="flex flex-col border-l border-border/70 bg-card">
              <DialogHeader className="px-4 pt-4">
                <DialogTitle className="text-sm">Image Details</DialogTitle>
                <DialogDescription className="text-xs">{lightbox.datasetName}</DialogDescription>
              </DialogHeader>
              <ScrollArea className="flex-1 px-4 py-3">
                <div className="grid grid-cols-[64px_minmax(0,1fr)] gap-x-3 gap-y-3 text-xs">
                  <div className="text-muted-foreground">File</div>
                  <div className="break-all font-mono text-foreground">{lightbox.filename}</div>
                  {lightbox.caption ? (
                    <>
                      <div className="text-muted-foreground">Caption</div>
                      <div className="leading-relaxed text-foreground">{lightbox.caption}</div>
                    </>
                  ) : (
                    <>
                      <div className="text-muted-foreground">Caption</div>
                      <div className="text-muted-foreground/50">No caption</div>
                    </>
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
    </>
  )
}
