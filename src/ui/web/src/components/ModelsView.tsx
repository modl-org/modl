import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Download, Loader2, Search, X } from 'lucide-react'
import { api, type InstalledModel, type FeatureDep, type SearchResult } from '../api'

type TypeGroup = {
  type: string
  label: string
  models: InstalledModel[]
  totalSize: number
}

const TYPE_ORDER: Record<string, number> = {
  checkpoint: 0,
  diffusion_model: 1,
  lora: 2,
  vae: 3,
  text_encoder: 4,
  upscaler: 5,
  controlnet: 6,
  ipadapter: 7,
  embedding: 8,
  llm: 9,
}

const TYPE_LABELS: Record<string, string> = {
  checkpoint: 'Checkpoints',
  diffusion_model: 'Diffusion Models',
  lora: 'LoRAs',
  vae: 'VAEs',
  text_encoder: 'Text Encoders',
  upscaler: 'Upscalers',
  controlnet: 'ControlNets',
  ipadapter: 'IP Adapters',
  embedding: 'Embeddings',
  llm: 'LLMs',
}

function formatSize(bytes: number | null | undefined): string {
  if (!bytes || bytes <= 0) return '—'
  if (bytes >= 1024 * 1024 * 1024) {
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
  }
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`
  }
  return `${(bytes / 1024).toFixed(0)} KB`
}

function sourceLabel(id: string): string {
  if (id.startsWith('train:')) return 'Trained'
  if (id.startsWith('hf:')) return 'HuggingFace'
  if (id.startsWith('local/')) return 'Local'
  return 'Registry'
}

function sourceClass(id: string): string {
  if (id.startsWith('train:')) return 'border-purple-500/50 text-purple-300'
  if (id.startsWith('hf:')) return 'border-yellow-500/50 text-yellow-300'
  if (id.startsWith('local/')) return 'border-blue-500/50 text-blue-300'
  return 'border-border text-muted-foreground'
}

export function ModelsView() {
  const queryClient = useQueryClient()
  const [deleteTarget, setDeleteTarget] = useState<InstalledModel | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchActive, setSearchActive] = useState(false)
  const [installing, setInstalling] = useState<string | null>(null)
  const [installError, setInstallError] = useState<string | null>(null)

  const { data: response, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: 30_000,
  })

  const { data: searchResults = [], isFetching: searching } = useQuery({
    queryKey: ['registry-search', searchQuery],
    queryFn: () => api.searchRegistry(searchQuery),
    enabled: searchQuery.length >= 2,
    staleTime: 60_000,
  })

  const models = response?.models ?? []
  const totalSize = response?.total_size_bytes ?? models.reduce((s, m) => s + (m.size_bytes ?? 0), 0)
  const featureDeps = response?.feature_deps ?? []

  const groups: TypeGroup[] = useMemo(() => {
    const map = new Map<string, InstalledModel[]>()
    for (const m of models) {
      const list = map.get(m.model_type) ?? []
      list.push(m)
      map.set(m.model_type, list)
    }

    return Array.from(map.entries())
      .map(([type, items]) => ({
        type,
        label: TYPE_LABELS[type] ?? type,
        models: items.sort((a, b) => a.name.localeCompare(b.name)),
        totalSize: items.reduce((sum, m) => sum + m.size_bytes, 0),
      }))
      .sort((a, b) => (TYPE_ORDER[a.type] ?? 99) - (TYPE_ORDER[b.type] ?? 99))
  }, [models])

  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteModel(deleteTarget.id)
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      setDeleteTarget(null)
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setDeleting(false)
    }
  }

  const handleInstall = async (result: SearchResult, variant?: string) => {
    setInstalling(result.id)
    setInstallError(null)
    try {
      await api.installModel(result.id, variant)
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      // Refresh search to update installed status
      await queryClient.invalidateQueries({ queryKey: ['registry-search'] })
    } catch (err) {
      setInstallError(err instanceof Error ? err.message : 'Install failed')
    } finally {
      setInstalling(null)
    }
  }

  const hasDependents = (m: InstalledModel) =>
    m.depended_on_by && m.depended_on_by.length > 0

  return (
    <>
      <div className="space-y-6">
        {/* Search bar */}
        <div className="flex items-center gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search registry to install models..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
                if (e.target.value.length >= 2) setSearchActive(true)
              }}
              onFocus={() => { if (searchQuery.length >= 2) setSearchActive(true) }}
              className="h-9 w-full rounded-md border border-border bg-secondary/30 pl-9 pr-8 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
            {searchQuery && (
              <button
                onClick={() => { setSearchQuery(''); setSearchActive(false); setInstallError(null) }}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Search results */}
        {searchActive && searchQuery.length >= 2 && (
          <div className="rounded-lg border border-border/60">
            <div className="border-b border-border/40 px-4 py-2">
              <p className="text-xs font-medium text-muted-foreground">
                {searching ? 'Searching...' : `${searchResults.length} results from registry`}
              </p>
            </div>
            {installError && (
              <div className="border-b border-red-500/20 bg-red-500/5 px-4 py-2 text-xs text-red-400">
                {installError}
              </div>
            )}
            <div className="max-h-80 overflow-y-auto divide-y divide-border/30">
              {searchResults.map((r: SearchResult) => (
                <div
                  key={r.id}
                  className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-secondary/20"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-foreground">{r.name}</span>
                      <Badge variant="outline" className="shrink-0 text-[10px]">{r.model_type}</Badge>
                      {r.requires_auth && (
                        <Badge variant="outline" className="shrink-0 text-[10px] border-amber-500/40 text-amber-400">
                          Auth required
                        </Badge>
                      )}
                    </div>
                    <div className="mt-0.5 flex items-center gap-3 text-xs text-muted-foreground">
                      {r.author && <span>{r.author}</span>}
                      <span className="font-mono">{formatSize(r.size_bytes)}</span>
                      {r.variants.length > 1 && (
                        <span>{r.variants.length} variants</span>
                      )}
                    </div>
                    {r.description && (
                      <p className="mt-1 line-clamp-1 text-xs text-muted-foreground/70">{r.description}</p>
                    )}
                  </div>
                  {r.installed ? (
                    <Badge variant="outline" className="shrink-0 text-[10px] border-emerald-500/40 text-emerald-400">
                      Installed
                    </Badge>
                  ) : (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="shrink-0 gap-1.5 text-xs"
                      disabled={installing === r.id}
                      onClick={() => void handleInstall(r)}
                    >
                      {installing === r.id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Download className="h-3 w-3" />
                      )}
                      {installing === r.id ? 'Installing...' : 'Install'}
                    </Button>
                  )}
                </div>
              ))}
              {!searching && searchResults.length === 0 && (
                <div className="px-4 py-6 text-center text-xs text-muted-foreground">
                  No models found. Try a different search term.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Summary bar */}
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <p className="text-sm font-semibold text-foreground">
              {models.length} models installed
            </p>
            <p className="text-xs text-muted-foreground">
              {formatSize(totalSize)} total on disk
            </p>
          </div>
          <div className="ml-auto flex flex-wrap gap-2">
            {groups.map((g) => (
              <Badge key={g.type} variant="outline" className="text-xs">
                {g.models.length} {g.label.toLowerCase()}
              </Badge>
            ))}
          </div>
        </div>

        {/* Feature dependency warnings */}
        {featureDeps.length > 0 && (
          <div className="space-y-2">
            {featureDeps.map((dep: FeatureDep) => (
              <div
                key={dep.feature}
                className="flex items-center gap-3 rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-3"
              >
                <span className="text-sm text-amber-300">
                  <strong>{dep.feature}</strong>
                  <span className="text-amber-300/70"> — {dep.description}</span>
                </span>
                {dep.install_hint && (
                  <code className="ml-auto shrink-0 rounded bg-secondary px-2 py-1 text-xs text-muted-foreground">
                    {dep.install_hint}
                  </code>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Loading state */}
        {isLoading && (
          <div className="py-12 text-center text-sm text-muted-foreground">Loading models...</div>
        )}

        {/* Model groups */}
        {groups.map((group) => (
          <div key={group.type}>
            <div className="mb-3 flex items-center gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                {group.label}
              </h3>
              <span className="text-xs text-muted-foreground/60">
                {formatSize(group.totalSize)}
              </span>
            </div>

            <div className="divide-y divide-border/40 rounded-lg border border-border/60">
              {group.models.map((m) => (
                <div
                  key={m.id}
                  className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-secondary/20"
                >
                  {/* Info */}
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium text-foreground">
                        {m.name}
                      </span>
                      {m.variant && (
                        <Badge variant="outline" className="shrink-0 text-[10px]">
                          {m.variant}
                        </Badge>
                      )}
                      <Badge
                        variant="outline"
                        className={`shrink-0 text-[10px] ${sourceClass(m.id)}`}
                      >
                        {sourceLabel(m.id)}
                      </Badge>
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
                      <span className="font-mono">{formatSize(m.size_bytes)}</span>
                      <span className="truncate opacity-60" title={m.id}>{m.id}</span>
                      {m.trigger_word && (
                        <span>
                          trigger: <code className="text-foreground/80">{m.trigger_word}</code>
                        </span>
                      )}
                    </div>

                    {/* Dependencies */}
                    {m.depends_on && m.depends_on.length > 0 && (
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground/50">
                          requires:
                        </span>
                        {m.depends_on.map((dep) => (
                          <Badge
                            key={dep.id}
                            variant="outline"
                            className={`text-[10px] ${
                              dep.installed
                                ? 'border-emerald-500/30 text-emerald-400/70'
                                : 'border-red-500/40 text-red-400'
                            }`}
                          >
                            {dep.installed ? '' : '! '}{dep.id}
                          </Badge>
                        ))}
                      </div>
                    )}
                    {hasDependents(m) && (
                      <div className="mt-1 flex flex-wrap gap-1.5">
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground/50">
                          used by:
                        </span>
                        {m.depended_on_by!.map((depId) => (
                          <Badge
                            key={depId}
                            variant="outline"
                            className="text-[10px] border-primary/30 text-primary/70"
                          >
                            {depId}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Delete button */}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="shrink-0 text-xs text-muted-foreground/50 hover:text-red-400"
                    onClick={() => setDeleteTarget(m)}
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          </div>
        ))}

        {!isLoading && models.length === 0 && (
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">No models installed.</p>
            <p className="mt-1 text-xs text-muted-foreground/60">
              Run <code className="rounded bg-secondary px-1.5 py-0.5">modl pull flux-schnell</code> to get started.
            </p>
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      <Dialog open={Boolean(deleteTarget)} onOpenChange={(open) => !open && setDeleteTarget(null)}>
        <DialogContent className="sm:max-w-md">
          {deleteTarget && (
            <>
              <DialogHeader>
                <DialogTitle>Remove {deleteTarget.name}?</DialogTitle>
                <DialogDescription>
                  This will uninstall the model and remove its symlinks.
                  {deleteTarget.model_type !== 'lora' && (
                    <> The store file is kept until you run <code>modl gc</code>.</>
                  )}
                  {deleteTarget.model_type === 'lora' && deleteTarget.id.startsWith('train:') && (
                    <> This will also delete training output (samples, logs, config).</>
                  )}
                </DialogDescription>
              </DialogHeader>

              {hasDependents(deleteTarget) && (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2 text-xs text-amber-300">
                  Warning: this model is required by{' '}
                  <strong>{deleteTarget.depended_on_by!.join(', ')}</strong>.
                  Removing it may break those models.
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <Button variant="ghost" size="sm" onClick={() => setDeleteTarget(null)}>
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  disabled={deleting}
                  onClick={() => void handleDelete()}
                >
                  {deleting ? 'Removing...' : 'Remove'}
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
