import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { api, type GeneratedImage, type GeneratedOutput } from '../api'
import type { GenerateFormState } from './generate-form-state'
import { ImageDetail } from './ImageDetail'
import { LazyImage } from './LazyImage'

type Props = {
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  setActiveTab: (tab: 'train' | 'generate' | 'outputs' | 'datasets') => void
}

export function OutputsGallery({ setForm, setActiveTab }: Props) {
  const queryClient = useQueryClient()

  const {
    data: groups = [],
    error,
    isLoading,
    isFetching,
    refetch,
  } = useQuery({
    queryKey: ['outputs'],
    queryFn: api.outputs,
    staleTime: 30_000,
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteOutput,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['outputs'] })
    },
  })

  const [selected, setSelected] = useState<GeneratedImage | null>(null)
  const [modelFilter, setModelFilter] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<string | null>(null)
  const [sortNewestFirst, setSortNewestFirst] = useState(true)

  const modelOptions = useMemo(() => {
    const models = new Set<string>()
    for (const group of groups) {
      for (const image of group.images) {
        if (image.base_model_id) {
          models.add(image.base_model_id)
        }
      }
    }
    return [...models].sort((a, b) => a.localeCompare(b))
  }, [groups])

  const dateOptions = useMemo(() => {
    return groups.map((group) => group.date)
  }, [groups])

  const filteredGroups = useMemo(() => {
    const next: GeneratedOutput[] = []

    for (const group of groups) {
      if (dateFilter && group.date !== dateFilter) {
        continue
      }

      const filteredImages = group.images
        .filter((image) => (modelFilter ? image.base_model_id === modelFilter : true))
        .sort((a, b) => {
          const left = a.modified ?? 0
          const right = b.modified ?? 0
          return sortNewestFirst ? right - left : left - right
        })

      if (filteredImages.length > 0) {
        next.push({ ...group, images: filteredImages })
      }
    }

    next.sort((a, b) => (sortNewestFirst ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date)))
    return next
  }, [groups, dateFilter, modelFilter, sortNewestFirst])

  const onDelete = async (image: GeneratedImage) => {
    if (!window.confirm(`Delete ${image.filename}?`)) {
      return
    }

    await deleteMutation.mutateAsync({ artifact_id: image.artifact_id, path: image.path })
  }

  const combinedError = error ?? deleteMutation.error

  return (
    <div className="space-y-6">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="mr-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">Model</span>
          <Button
            type="button"
            size="sm"
            variant={modelFilter === null ? 'secondary' : 'ghost'}
            className="h-7 px-2.5 text-xs"
            onClick={() => setModelFilter(null)}
          >
            All
          </Button>
          {modelOptions.map((model) => (
            <Button
              key={model}
              type="button"
              size="sm"
              variant={modelFilter === model ? 'secondary' : 'ghost'}
              className="h-7 px-2.5 text-xs"
              onClick={() => setModelFilter(model)}
            >
              {model}
            </Button>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <div className="flex flex-wrap gap-1">
            <Button
              type="button"
              size="sm"
              variant={dateFilter === null ? 'secondary' : 'ghost'}
              className="h-7 px-2.5 text-xs"
              onClick={() => setDateFilter(null)}
            >
              All dates
            </Button>
            {dateOptions.map((date) => (
              <Button
                key={date}
                type="button"
                size="sm"
                variant={dateFilter === date ? 'secondary' : 'ghost'}
                className="h-7 px-2.5 text-xs"
                onClick={() => setDateFilter(date)}
              >
                {date}
              </Button>
            ))}
          </div>
          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={() => setSortNewestFirst((prev) => !prev)}>
            {sortNewestFirst ? '↓ Newest' : '↑ Oldest'}
          </Button>
          <Button type="button" size="sm" variant="ghost" className="h-7 px-2.5 text-xs text-muted-foreground" onClick={() => void refetch()}>
            {isFetching ? 'Refreshing…' : 'Refresh'}
          </Button>
        </div>
      </div>

      {combinedError ? (
        <p className="text-sm text-destructive">Failed to load outputs: {String(combinedError)}</p>
      ) : null}

      {isLoading ? (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
          {Array.from({ length: 10 }).map((_, i) => (
            <div key={i} className="aspect-square animate-pulse rounded-lg bg-secondary/50" />
          ))}
        </div>
      ) : null}

      {!isLoading && filteredGroups.length === 0 && !combinedError ? (
        <p className="py-8 text-center text-sm text-muted-foreground">No generated images for the selected filters.</p>
      ) : null}

      {filteredGroups.map((group) => (
        <section key={group.date} className="space-y-2">
          <div className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{group.date}</div>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6">
            {group.images.map((image) => (
              <article key={image.path} className="group relative overflow-hidden rounded-lg">
                <LazyImage
                  src={`/files/${image.path}`}
                  alt={image.filename}
                  className="aspect-square"
                  onClick={() => setSelected(image)}
                />
                {/* Hover overlay */}
                <div className="pointer-events-none absolute inset-0 bg-black/0 transition-colors group-hover:bg-black/30" />
                <div className="absolute inset-x-0 bottom-0 translate-y-full px-2 pb-2 pt-6 opacity-0 transition-all group-hover:translate-y-0 group-hover:opacity-100" style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }}>
                  <p className="truncate font-mono text-[10px] text-white/70">{image.filename}</p>
                </div>
                <div className="pointer-events-none absolute top-1.5 right-1.5 flex gap-1 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100">
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => setSelected(image)}
                  >
                    Info
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    className="h-6 px-2 text-[10px]"
                    onClick={() => void onDelete(image)}
                    disabled={deleteMutation.isPending}
                  >
                    ✕
                  </Button>
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}

      <ImageDetail
        image={selected}
        onClose={() => setSelected(null)}
        setForm={setForm}
        setActiveTab={setActiveTab}
      />
    </div>
  )
}
