import { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { api, type GeneratedImage, type GeneratedOutput, type ModelFamily } from '../api'
import { modelColor } from '@/lib/utils'
import { STALE_FAST, STALE_SLOW, STALE_STATIC } from '@/lib/query-keys'
import { useLocalStorage } from '../hooks/useLocalStorage'
import { useForm, useAppNav } from '../contexts/FormContext'
import { buildSendToEdit } from './generate/generate-state'
import { ImageDetail } from './ImageDetail'
import { GalleryFilterBar, ImageGridItem } from './gallery'

/** Column count matching the Tailwind grid classes below (viewport breakpoints). */
function computeColumns(gridSize: 's' | 'm' | 'l', viewportWidth: number): number {
  if (gridSize === 's') {
    return viewportWidth >= 1280 ? 9 : viewportWidth >= 1024 ? 7 : viewportWidth >= 768 ? 5 : 3
  }
  if (gridSize === 'l') {
    return viewportWidth >= 1280 ? 4 : viewportWidth >= 1024 ? 3 : viewportWidth >= 768 ? 2 : 1
  }
  return viewportWidth >= 1536 ? 6 : viewportWidth >= 1280 ? 5 : viewportWidth >= 1024 ? 4 : viewportWidth >= 768 ? 3 : 2
}

function useGridColumns(gridSize: 's' | 'm' | 'l'): number {
  const [cols, setCols] = useState(() => computeColumns(gridSize, window.innerWidth))
  useEffect(() => {
    const onResize = () => setCols(computeColumns(gridSize, window.innerWidth))
    onResize()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [gridSize])
  return cols
}

/** One virtualized row: a group header or a chunk of `columns` images. */
type GalleryRow =
  | { key: string; type: 'header'; label: string; count: number; collapsed: boolean }
  | { key: string; type: 'images'; images: GeneratedImage[] }

export function OutputsGallery() {
  const { setForm } = useForm()
  const { navigateToTab, useAsRecipe } = useAppNav()
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
    staleTime: STALE_FAST,
  })

  const deleteMutation = useMutation({
    mutationFn: api.deleteOutput,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['outputs'] })
    },
  })

  const batchDeleteMutation = useMutation({
    mutationFn: api.batchDeleteOutputs,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ['outputs'] })
      setSelectedPaths(new Set())
      setSelectMode(false)
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

  // Models + families (cached from other tabs, needed for send-to-edit)
  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: STALE_SLOW,
  })
  const models = modelsResponse?.models ?? []

  const { data: families = [] } = useQuery<ModelFamily[]>({
    queryKey: ['model-families'],
    queryFn: api.modelFamilies,
    staleTime: STALE_STATIC,
  })

  const sendToEdit = useCallback((imageUrl: string, serverPath: string) => {
    setForm(buildSendToEdit(imageUrl, serverPath, models, families))
    navigateToTab('generate')
  }, [models, families, setForm, navigateToTab])

  const [selected, setSelected] = useState<GeneratedImage | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<GeneratedImage | null>(null)
  const [modelFilter, setModelFilter] = useState<string | null>(null)
  const [dateFilter, setDateFilter] = useState<string | null>(null)
  const [sortNewestFirst, setSortNewestFirst] = useLocalStorage('modl:outputs-sort', true)
  const [gridSize, setGridSize] = useLocalStorage<'s' | 'm' | 'l'>('modl:outputs-grid', 'm')
  const [favoriteFilter, setFavoriteFilter] = useState(false)
  const [groupBy, setGroupBy] = useLocalStorage<'date' | 'model' | 'none'>('modl:outputs-group', 'date')
  const [searchQuery, setSearchQuery] = useState('')
  const searchRef = useRef<HTMLInputElement>(null)

  // Batch selection
  const [selectMode, setSelectMode] = useState(false)
  const [selectedPaths, setSelectedPaths] = useState<Set<string>>(new Set())
  const [confirmBatchDelete, setConfirmBatchDelete] = useState(false)

  // Track cursor index for shift+click and shift+arrow range select
  const lastClickedRef = useRef<number>(-1)
  const [cursorIndex, setCursorIndex] = useState(-1)

  // Collapsed groups
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const toggleCollapsed = useCallback((key: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }, [])

  const gridClass = {
    s: 'grid-cols-3 gap-1.5 md:grid-cols-5 lg:grid-cols-7 xl:grid-cols-9',
    m: 'grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6',
    l: 'grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4',
  }[gridSize]

  // Thumbnail width based on grid size
  const thumbWidth = { s: 200, m: 320, l: 480 }[gridSize]

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

  // Defer the heavy filter/regroup while keeping the input itself snappy —
  // with thousands of images the recompute is a per-keystroke long task.
  const deferredSearch = useDeferredValue(searchQuery)
  const searchLower = deferredSearch.toLowerCase().trim()

  const filteredGroups = useMemo(() => {
    // 1. Flatten all images, apply filters + search
    const allImages: (GeneratedImage & { date: string })[] = []
    for (const group of groups) {
      if (dateFilter && group.date !== dateFilter) continue
      for (const image of group.images) {
        if (modelFilter && image.base_model_id !== modelFilter) continue
        if (favoriteFilter && !image.favorited) continue
        if (searchLower && !(
          image.prompt?.toLowerCase().includes(searchLower) ||
          image.filename.toLowerCase().includes(searchLower) ||
          image.base_model_id?.toLowerCase().includes(searchLower) ||
          image.lora_name?.toLowerCase().includes(searchLower)
        )) continue
        allImages.push({ ...image, date: group.date })
      }
    }

    // 2. Sort
    allImages.sort((a, b) => {
      const left = a.modified ?? 0
      const right = b.modified ?? 0
      return sortNewestFirst ? right - left : left - right
    })

    // 3. Group
    const grouped = new Map<string, GeneratedImage[]>()
    for (const img of allImages) {
      const key = groupBy === 'model'
        ? (img.base_model_id ?? 'Unknown model')
        : groupBy === 'date'
          ? img.date
          : 'All images'
      if (!grouped.has(key)) grouped.set(key, [])
      grouped.get(key)!.push(img)
    }

    // 4. Sort groups
    const result: GeneratedOutput[] = []
    for (const [key, images] of grouped) {
      result.push({ date: key, images })
    }
    if (groupBy !== 'none') {
      result.sort((a, b) => sortNewestFirst ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date))
    }
    return result
  }, [groups, dateFilter, modelFilter, sortNewestFirst, favoriteFilter, groupBy, searchLower])

  const allFilteredImages = useMemo(() => {
    return filteredGroups.flatMap((g) => g.images)
  }, [filteredGroups])

  const totalImageCount = useMemo(() => {
    return groups.reduce((sum, g) => sum + g.images.length, 0)
  }, [groups])

  const isFiltered = modelFilter !== null || dateFilter !== null || favoriteFilter || searchLower !== ''

  // Stats bar data
  const [statsOpen, setStatsOpen] = useLocalStorage('modl:outputs-stats', true)

  const stats = useMemo(() => {
    const images = allFilteredImages
    const total = images.length
    const today = new Date().toISOString().slice(0, 10)

    const byModel = new Map<string, { count: number; stars: number }>()
    let todayCount = 0

    for (const img of images) {
      const mid = img.base_model_id ?? 'unknown'
      const entry = byModel.get(mid) ?? { count: 0, stars: 0 }
      entry.count++
      if (img.favorited) entry.stars++
      byModel.set(mid, entry)

      if (img.path?.includes(today)) todayCount++
    }

    const modelChips = [...byModel.entries()]
      .sort((a, b) => b[1].count - a[1].count)

    return { total, todayCount, modelChips }
  }, [allFilteredImages])

  // Global keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault()
        searchRef.current?.focus()
        return
      }
      if (!selectMode || selected) return
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return
      e.preventDefault()
      const total = allFilteredImages.length
      if (total === 0) return
      setCursorIndex((prev) => {
        const cur = prev < 0 ? 0 : prev
        let next = cur
        if (e.key === 'ArrowRight') next = Math.min(cur + 1, total - 1)
        else if (e.key === 'ArrowLeft') next = Math.max(cur - 1, 0)
        else if (e.key === 'ArrowDown') next = Math.min(cur + 4, total - 1)
        else if (e.key === 'ArrowUp') next = Math.max(cur - 4, 0)
        if (e.shiftKey) {
          const anchor = lastClickedRef.current >= 0 ? lastClickedRef.current : 0
          const start = Math.min(anchor, next)
          const end = Math.max(anchor, next)
          setSelectedPaths((prevSet) => {
            const s = new Set(prevSet)
            for (let i = start; i <= end; i++) s.add(allFilteredImages[i].path)
            return s
          })
        }
        return next
      })
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [selectMode, selected, allFilteredImages])

  const togglePathSelection = useCallback((path: string, shiftKey = false) => {
    const clickedIndex = allFilteredImages.findIndex((img) => img.path === path)
    if (shiftKey && lastClickedRef.current >= 0 && clickedIndex >= 0) {
      const start = Math.min(lastClickedRef.current, clickedIndex)
      const end = Math.max(lastClickedRef.current, clickedIndex)
      setSelectedPaths((prev) => {
        const next = new Set(prev)
        for (let i = start; i <= end; i++) {
          next.add(allFilteredImages[i].path)
        }
        return next
      })
    } else {
      setSelectedPaths((prev) => {
        const next = new Set(prev)
        if (next.has(path)) {
          next.delete(path)
        } else {
          next.add(path)
        }
        return next
      })
    }
    lastClickedRef.current = clickedIndex
    setCursorIndex(clickedIndex)
  }, [allFilteredImages])

  const selectAllVisible = useCallback(() => {
    setSelectedPaths(new Set(allFilteredImages.map((img) => img.path)))
  }, [allFilteredImages])

  const deselectAll = useCallback(() => {
    setSelectedPaths(new Set())
  }, [])

  const onDelete = async (image: GeneratedImage) => {
    await deleteMutation.mutateAsync({ artifact_id: image.artifact_id, path: image.path })
    setDeleteTarget(null)
  }

  const onBatchDelete = async () => {
    const items = allFilteredImages
      .filter((img) => selectedPaths.has(img.path))
      .map((img) => ({ artifact_id: img.artifact_id, path: img.path }))
    if (items.length === 0) return
    await batchDeleteMutation.mutateAsync(items)
    setConfirmBatchDelete(false)
  }

  const handleSelectModeChange = useCallback((mode: boolean) => {
    setSelectMode(mode)
    if (!mode) {
      setSelectedPaths(new Set())
      setConfirmBatchDelete(false)
      setCursorIndex(-1)
      lastClickedRef.current = -1
    }
  }, [])

  const combinedError = error ?? deleteMutation.error ?? favoriteMutation.error ?? batchDeleteMutation.error

  // ── Virtualized rows ────────────────────────────────────────────────────
  // 6k+ images can't all live in the DOM at once (~160k nodes, 100ms+ per
  // reconciliation). Chunk each group into rows of `columns` images and only
  // render what's on screen.
  const columns = useGridColumns(gridSize)
  const rows = useMemo(() => {
    const out: GalleryRow[] = []
    for (const group of filteredGroups) {
      const collapsed = collapsedGroups.has(group.date)
      if (groupBy !== 'none') {
        out.push({
          key: `h:${group.date}`,
          type: 'header',
          label: group.date,
          count: group.images.length,
          collapsed,
        })
      }
      if (groupBy === 'none' || !collapsed) {
        for (let i = 0; i < group.images.length; i += columns) {
          out.push({
            key: `r:${group.date}:${group.images[i].path}`,
            type: 'images',
            images: group.images.slice(i, i + columns),
          })
        }
      }
    }
    return out
  }, [filteredGroups, collapsedGroups, groupBy, columns])

  // The scrollable ancestor is App's <main>; found on mount via closest().
  const listRef = useRef<HTMLDivElement | null>(null)
  const scrollElRef = useRef<HTMLElement | null>(null)
  useEffect(() => {
    scrollElRef.current = listRef.current?.closest('main') ?? null
  }, [])

  const estimatedRowHeight = useMemo(() => {
    const width = listRef.current?.clientWidth ?? window.innerWidth - 280
    return Math.ceil(width / columns) + 8
  }, [columns])

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => scrollElRef.current,
    estimateSize: (i) => (rows[i].type === 'header' ? 28 : estimatedRowHeight),
    overscan: 5,
    getItemKey: (i) => rows[i].key,
  })

  return (
    <div className="space-y-4">
      <GalleryFilterBar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        searchRef={searchRef}
        modelFilter={modelFilter}
        onModelFilterChange={setModelFilter}
        modelOptions={modelOptions}
        dateFilter={dateFilter}
        onDateFilterChange={setDateFilter}
        dateOptions={dateOptions}
        favoriteFilter={favoriteFilter}
        onFavoriteFilterChange={setFavoriteFilter}
        sortNewestFirst={sortNewestFirst}
        onSortChange={setSortNewestFirst}
        groupBy={groupBy}
        onGroupByChange={setGroupBy}
        selectMode={selectMode}
        onSelectModeChange={handleSelectModeChange}
        gridSize={gridSize}
        onGridSizeChange={setGridSize}
        isFetching={isFetching}
        onRefresh={() => void refetch()}
        selectedCount={selectedPaths.size}
        filteredCount={allFilteredImages.length}
        onSelectAll={selectAllVisible}
        onDeselectAll={deselectAll}
        confirmBatchDelete={confirmBatchDelete}
        onConfirmBatchDelete={setConfirmBatchDelete}
        onBatchDelete={() => void onBatchDelete()}
        batchDeletePending={batchDeleteMutation.isPending}
        isLoading={isLoading}
        stats={stats}
        statsOpen={statsOpen}
        onStatsOpenChange={setStatsOpen}
        isFiltered={isFiltered}
        totalImageCount={totalImageCount}
      />

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

      <div
        ref={listRef}
        style={{ height: virtualizer.getTotalSize(), position: 'relative' }}
      >
        {virtualizer.getVirtualItems().map((vi) => {
          const row = rows[vi.index]
          return (
            <div
              key={vi.key}
              data-index={vi.index}
              ref={virtualizer.measureElement}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${vi.start}px)`,
              }}
            >
              {row.type === 'header' ? (
                <button
                  type="button"
                  className="flex w-full items-center gap-1.5 py-2 text-left"
                  onClick={() => toggleCollapsed(row.label)}
                >
                  {row.collapsed
                    ? <ChevronRight className="size-3.5 text-muted-foreground/60" />
                    : <ChevronDown className="size-3.5 text-muted-foreground/60" />
                  }
                  {groupBy === 'model' && (
                    <span
                      className="inline-block size-2 rounded-full"
                      style={{ backgroundColor: modelColor(row.label) }}
                    />
                  )}
                  <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{row.label}</span>
                  <span className="text-[10px] text-muted-foreground/50">{row.count}</span>
                </button>
              ) : (
                <div className={`grid ${gridClass} pb-2`}>
                  {row.images.map((image) => {
                    const isSelected = selectedPaths.has(image.path)
                    const isDeleting = deleteTarget?.path === image.path
                    const isCursor = selectMode && cursorIndex >= 0 && allFilteredImages[cursorIndex]?.path === image.path

                    return (
                      <ImageGridItem
                        key={image.path}
                        image={image}
                        isSelected={isSelected}
                        isCursor={isCursor}
                        isDeleting={isDeleting}
                        selectMode={selectMode}
                        thumbWidth={thumbWidth}
                        deletePending={deleteMutation.isPending}
                        onImageClick={(e) => {
                          if (selectMode) {
                            togglePathSelection(image.path, e.shiftKey)
                          } else {
                            setSelected(image)
                          }
                        }}
                        onToggleSelection={(e) => {
                          e.stopPropagation()
                          togglePathSelection(image.path, e.shiftKey)
                        }}
                        onFavorite={() => favoriteMutation.mutate(image.path)}
                        onOpenAsRecipe={() => useAsRecipe(image)}
                        onSendToEdit={() => sendToEdit(`/files/${image.path}`, image.path)}
                        onOpenDetail={() => setSelected(image)}
                        onRequestDelete={() => setDeleteTarget(image)}
                        onConfirmDelete={() => void onDelete(image)}
                        onCancelDelete={() => setDeleteTarget(null)}
                      />
                    )
                  })}
                </div>
              )}
            </div>
          )
        })}
      </div>

      <ImageDetail
        image={selected}
        onClose={() => setSelected(null)}
        allImages={allFilteredImages}
        onNavigate={setSelected}
        onToggleFavorite={(path) => favoriteMutation.mutate(path)}
        onEditImage={sendToEdit}
        onDeleteImage={(img) => {
          setDeleteTarget(img)
          setSelected(null)
        }}
      />
    </div>
  )
}
