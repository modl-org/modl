import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { api, type GeneratedImage, type GeneratedOutput } from '../api'
import type { GenerateFormState } from './generate'
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

  const favoriteMutation = useMutation<{ favorited: boolean }, Error, string, { prev: GeneratedOutput[] | undefined }>({
    mutationFn: (path: string) => api.favoriteOutput(path),
    onMutate: async (path: string) => {
      await queryClient.cancelQueries({ queryKey: ['outputs'] })
      const prev = queryClient.getQueryData<GeneratedOutput[]>(['outputs'])
      queryClient.setQueryData<GeneratedOutput[]>(['outputs'], (old) =>
        old?.map((g) => ({
          ...g,
          images: g.images.map((img) =>
            img.path === path ? { ...img, favorited: !img.favorited } : img,
          ),
        }))
      )
      return { prev }
    },
    onError: (_err, _path, context) => {
      if (context?.prev) {
        queryClient.setQueryData(['outputs'], context.prev)
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['outputs'] })
    },
  })

  const [selected, setSelected] = useState<GeneratedImage | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<GeneratedImage | null>(null)
  const [modelFilter, setModelFilter] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<string | null>(null)
  const [sortNewestFirst, setSortNewestFirst] = useState(true)
  const [gridSize, setGridSize] = useState<'s' | 'm' | 'l'>('m')
  const [favoriteFilter, setFavoriteFilter] = useState(false)

  const gridClass = {
    s: 'grid-cols-3 gap-1.5 md:grid-cols-5 lg:grid-cols-7 xl:grid-cols-9',
    m: 'grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6',
    l: 'grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  }[gridSize]

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
        .filter(
          (image) =>
            (modelFilter ? image.base_model_id === modelFilter : true) &&
            (!favoriteFilter || image.favorited),
        )
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
  }, [groups, dateFilter, modelFilter, sortNewestFirst, favoriteFilter])

  const allFilteredImages = useMemo(() => {
    return filteredGroups.flatMap((g) => g.images)
  }, [filteredGroups])

  const onDelete = async (image: GeneratedImage) => {
    await deleteMutation.mutateAsync({ artifact_id: image.artifact_id, path: image.path })
    setDeleteTarget(null)
  }

  const combinedError = error ?? deleteMutation.error ?? favoriteMutation.error

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
          <div className="mx-1 h-4 w-px bg-border/50" />
          <Button
            type="button"
            size="sm"
            variant={favoriteFilter ? 'secondary' : 'ghost'}
            className={`h-7 px-2.5 text-xs ${favoriteFilter ? 'text-yellow-400' : ''}`}
            onClick={() => setFavoriteFilter((prev) => !prev)}
          >
            ⭐ Starred
          </Button>
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

      {combinedError ? (
        <p className="text-sm text-destructive">Failed to load outputs: {String(combinedError)}</p>
      ) : null}

      {isLoading ? (
        <div className={`grid ${gridClass}`}>
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
          <div className={`grid ${gridClass}`}>
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
                <button
                  type="button"
                  className={`absolute top-1.5 left-1.5 text-base leading-none drop-shadow transition-opacity ${
                    image.favorited
                      ? 'opacity-100'
                      : 'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100'
                  }`}
                  onClick={(e) => {
                    e.stopPropagation()
                    favoriteMutation.mutate(image.path)
                  }}
                >
                  {image.favorited ? '⭐' : '☆'}
                </button>
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
                    className="h-6 w-6 p-0"
                    onClick={(e) => { e.stopPropagation(); setDeleteTarget(image) }}
                    disabled={deleteMutation.isPending}
                  >
                    <Trash2 className="size-3" />
                  </Button>
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}

      <Dialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}>
        <DialogContent className="max-w-sm" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>Delete image?</DialogTitle>
            <DialogDescription className="break-all">
              {deleteTarget?.filename} will be permanently deleted.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setDeleteTarget(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              disabled={deleteMutation.isPending}
              onClick={() => deleteTarget && void onDelete(deleteTarget)}
            >
              <Trash2 className="mr-1.5 size-4" />
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ImageDetail
        image={selected}
        onClose={() => setSelected(null)}
        setForm={setForm}
        setActiveTab={setActiveTab}
        allImages={allFilteredImages}
        onNavigate={setSelected}
        onToggleFavorite={(path) => favoriteMutation.mutate(path)}
      />
    </div>
  )
}
