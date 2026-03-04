import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { api } from '../api'

const PAGE_SIZE = 50

export function DatasetViewer() {
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [page, setPage] = useState(0)

  const {
    data: datasets = [],
    error: datasetsError,
    isLoading: datasetsLoading,
  } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.datasets,
    staleTime: 60_000,
  })

  const current = selectedDataset && datasets.includes(selectedDataset)
    ? selectedDataset
    : (datasets[0] ?? null)

  const {
    data: overview,
    error: overviewError,
    isLoading: overviewLoading,
  } = useQuery({
    queryKey: ['dataset', current, page],
    queryFn: () => api.dataset(current as string, PAGE_SIZE, page * PAGE_SIZE),
    enabled: Boolean(current),
    staleTime: 60_000,
    placeholderData: (previousData) => previousData,
  })

  const totalPages = overview ? Math.max(1, Math.ceil(overview.image_count / PAGE_SIZE)) : 1

  return (
    <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Datasets</CardTitle>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-[480px] pr-3">
            <div className="space-y-2">
              {datasets.map((name) => (
                <Button
                  key={name}
                  type="button"
                  className="w-full justify-start"
                  variant={name === current ? 'secondary' : 'ghost'}
                  onClick={() => {
                    setSelectedDataset(name)
                    setPage(0)
                  }}
                >
                  {name}
                </Button>
              ))}

              {!datasetsLoading && datasets.length === 0 && !datasetsError ? (
                <p className="text-sm text-muted-foreground">No datasets found.</p>
              ) : null}

              {datasetsError ? (
                <p className="text-sm text-destructive">Failed to load datasets: {String(datasetsError)}</p>
              ) : null}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <div className="space-y-4">
        {overviewError ? (
          <Card>
            <CardContent className="p-4 text-sm text-destructive">
              Failed to load dataset: {String(overviewError)}
            </CardContent>
          </Card>
        ) : null}

        {!overview && !overviewError ? (
          <Card>
            <CardContent className="p-8 text-center text-sm text-muted-foreground">
              {overviewLoading ? 'Loading dataset...' : 'Select a dataset to view images and captions.'}
            </CardContent>
          </Card>
        ) : null}

        {overview ? (
          <>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{overview.name}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-6">
                  <div>
                    <div className="text-2xl font-semibold text-primary">{overview.image_count}</div>
                    <div className="text-xs text-muted-foreground">Images</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-primary">{overview.captioned_count}</div>
                    <div className="text-xs text-muted-foreground">Captioned</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-primary">
                      {Math.round(overview.coverage * 100)}%
                    </div>
                    <div className="text-xs text-muted-foreground">Coverage</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
              {overview.images.map((image) => (
                <article key={image.image_url} className="overflow-hidden rounded-lg border border-border/70 bg-card">
                  <img
                    src={`/files/${image.image_url}`}
                    loading="lazy"
                    alt={image.filename}
                    className="aspect-square w-full object-cover"
                  />
                  {image.caption ? (
                    <div className="px-3 pt-2 text-xs text-muted-foreground">{image.caption}</div>
                  ) : null}
                  <div className="px-3 pb-2 pt-1 font-mono text-[11px] text-muted-foreground">{image.filename}</div>
                </article>
              ))}
            </div>

            <div className="flex items-center justify-center gap-2">
              <Button type="button" variant="outline" size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                Prev
              </Button>
              <span className="text-sm text-muted-foreground">
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
          </>
        ) : null}
      </div>
    </div>
  )
}
