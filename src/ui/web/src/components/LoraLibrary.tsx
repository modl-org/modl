import { useMemo, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { DeleteConfirmDialog } from '@/components/ui/DeleteConfirmDialog'
import {
  BookMarked,
  Copy,
  Check,
  Pencil,
  Search,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react'
import { api, type LibraryLora } from '../api'
import { timeAgo, displayModelName, formatBytes } from '@/lib/utils'
import { STALE_FAST } from '@/lib/query-keys'
import { useAppNav } from '../contexts/FormContext'

type SortMode = 'recent' | 'name' | 'base'

export function LoraLibrary() {
  const { navigateToTab, addLoraToForm } = useAppNav()
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortMode>('recent')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editName, setEditName] = useState('')
  const [editTags, setEditTags] = useState('')
  const [editNotes, setEditNotes] = useState('')
  const [deleteTarget, setDeleteTarget] = useState<LibraryLora | null>(null)
  const [deleting, setDeleting] = useState(false)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  const { data: loras = [], isLoading } = useQuery({
    queryKey: ['library-loras'],
    queryFn: api.libraryLoras,
    staleTime: STALE_FAST,
  })

  const totalSize = useMemo(
    () => loras.reduce((sum, l) => sum + (l.size_bytes ?? 0), 0),
    [loras],
  )

  const filtered = useMemo(() => {
    let result = [...loras]

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      result = result.filter(
        (l) =>
          l.name.toLowerCase().includes(q) ||
          (l.trigger_word?.toLowerCase().includes(q) ?? false) ||
          (l.base_model?.toLowerCase().includes(q) ?? false) ||
          (l.tags?.toLowerCase().includes(q) ?? false),
      )
    }

    if (sortBy === 'name') {
      result.sort((a, b) => a.name.localeCompare(b.name))
    } else if (sortBy === 'base') {
      result.sort((a, b) => (a.base_model ?? '').localeCompare(b.base_model ?? ''))
    }
    // 'recent' = default order from API (created_at DESC)

    return result
  }, [loras, searchQuery, sortBy])

  const startEdit = (lora: LibraryLora) => {
    setEditingId(lora.id)
    setEditName(lora.name)
    setEditTags(lora.tags ?? '')
    setEditNotes(lora.notes ?? '')
  }

  const saveEdit = async () => {
    if (!editingId) return
    try {
      await api.updateLibraryLora(editingId, {
        name: editName,
        tags: editTags || undefined,
        notes: editNotes || undefined,
      })
      void queryClient.invalidateQueries({ queryKey: ['library-loras'] })
      setEditingId(null)
    } catch (err) {
      console.error('Update failed:', err)
    }
  }

  const handleDelete = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      await api.deleteLibraryLora(deleteTarget.id)
      void queryClient.invalidateQueries({ queryKey: ['library-loras'] })
      setDeleteTarget(null)
    } catch (err) {
      console.error('Delete failed:', err)
    } finally {
      setDeleting(false)
    }
  }

  const copyTrigger = async (lora: LibraryLora) => {
    if (!lora.trigger_word) return
    try {
      await navigator.clipboard.writeText(lora.trigger_word)
      setCopiedId(lora.id)
      setTimeout(() => setCopiedId((prev) => (prev === lora.id ? null : prev)), 1200)
    } catch {
      // Ignore clipboard failures
    }
  }

  // Empty state
  if (!isLoading && loras.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-secondary/40">
          <BookMarked className="h-7 w-7 text-muted-foreground/40" />
        </div>
        <p className="text-sm font-medium text-foreground">No LoRAs in your library yet</p>
        <p className="mt-1 max-w-sm text-xs text-muted-foreground">
          Train a LoRA and promote it from the Training page, or browse community LoRAs on the Models page.
        </p>
        <div className="mt-4 flex gap-2">
          <Button
            size="sm"
            variant="outline"
            className="text-xs"
            onClick={() => navigateToTab('train')}
          >
            Go to Training
          </Button>
        </div>
      </div>
    )
  }

  return (
    <>
      <div className="space-y-4">
        {/* Header bar */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="shrink-0">
            <span className="text-sm font-semibold text-foreground">
              {loras.length} LoRA{loras.length !== 1 ? 's' : ''}
            </span>
            <span className="ml-2 text-xs text-muted-foreground">
              {formatBytes(totalSize)}
            </span>
          </div>

          {/* Search */}
          <div className="relative ml-auto min-w-[200px] max-w-xs flex-1">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search by name, trigger, model..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 w-full rounded-md border border-border bg-secondary/30 pl-8 pr-7 text-xs text-foreground placeholder:text-muted-foreground/60 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>

          {/* Sort */}
          <div className="flex items-center gap-1">
            {(['recent', 'name', 'base'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`rounded px-2 py-1 text-[10px] transition-colors ${
                  sortBy === mode
                    ? 'bg-secondary/60 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setSortBy(mode)}
              >
                {mode === 'recent' ? 'Recent' : mode === 'name' ? 'Name' : 'Base Model'}
              </button>
            ))}
          </div>
        </div>

        {/* Loading */}
        {isLoading && (
          <div className="py-12 text-center text-sm text-muted-foreground">Loading library...</div>
        )}

        {/* Grid */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((lora) => {
            const isEditing = editingId === lora.id
            return (
              <div
                key={lora.id}
                className="group rounded-lg border border-border/60 bg-card/50 transition-colors hover:border-border"
              >
                {/* Thumbnail */}
                {lora.thumbnail ? (
                  <div className="aspect-video w-full overflow-hidden rounded-t-lg border-b border-border/40">
                    <img
                      src={`/files/${lora.thumbnail}?w=400`}
                      alt={lora.name}
                      className="h-full w-full object-cover"
                    />
                  </div>
                ) : (
                  <div
                    className="flex aspect-video w-full items-center justify-center rounded-t-lg border-b border-border/40"
                    style={{
                      background: `linear-gradient(135deg, hsl(${Math.abs(lora.name.charCodeAt(0) * 7) % 360}, 30%, 15%), hsl(${Math.abs(lora.name.charCodeAt(0) * 7 + 60) % 360}, 20%, 10%))`,
                    }}
                  >
                    <BookMarked className="h-8 w-8 text-muted-foreground/20" />
                  </div>
                )}

                {/* Content */}
                <div className="space-y-2 p-3">
                  {/* Name */}
                  {isEditing ? (
                    <input
                      type="text"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      className="w-full rounded border border-primary/50 bg-secondary/40 px-2 py-1 text-sm font-medium text-foreground focus:outline-none focus:ring-1 focus:ring-primary/30"
                      autoFocus
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') void saveEdit()
                        if (e.key === 'Escape') setEditingId(null)
                      }}
                    />
                  ) : (
                    <div className="flex items-center gap-2">
                      <span className="min-w-0 truncate text-sm font-medium text-foreground">
                        {lora.name}
                      </span>
                      <button
                        type="button"
                        className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                        onClick={() => startEdit(lora)}
                        title="Edit"
                      >
                        <Pencil className="h-3 w-3 text-muted-foreground hover:text-foreground" />
                      </button>
                    </div>
                  )}

                  {/* Trigger word */}
                  {lora.trigger_word && (
                    <div className="flex items-center gap-1.5">
                      <code className="rounded bg-secondary/50 px-1.5 py-0.5 text-[11px] text-primary">
                        {lora.trigger_word}
                      </code>
                      <button
                        type="button"
                        className="text-muted-foreground hover:text-foreground"
                        onClick={() => void copyTrigger(lora)}
                        title="Copy trigger word"
                      >
                        {copiedId === lora.id ? (
                          <Check className="h-3 w-3 text-emerald-400" />
                        ) : (
                          <Copy className="h-3 w-3" />
                        )}
                      </button>
                    </div>
                  )}

                  {/* Metadata line */}
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[10px] text-muted-foreground">
                    {lora.base_model && (
                      <span>{displayModelName(lora.base_model)}</span>
                    )}
                    {lora.step && (
                      <Badge variant="outline" className="text-[10px] px-1 py-0">
                        step {lora.step.toLocaleString()}
                      </Badge>
                    )}
                    <span>{formatBytes(lora.size_bytes)}</span>
                    <span>{timeAgo(lora.created_at)}</span>
                  </div>

                  {/* Tags (editing mode) */}
                  {isEditing && (
                    <div className="space-y-1.5">
                      <input
                        type="text"
                        value={editTags}
                        onChange={(e) => setEditTags(e.target.value)}
                        placeholder="Tags (comma-separated)"
                        className="w-full rounded border border-border bg-secondary/40 px-2 py-1 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
                      />
                      <textarea
                        value={editNotes}
                        onChange={(e) => setEditNotes(e.target.value)}
                        placeholder="Notes..."
                        rows={2}
                        className="w-full resize-none rounded border border-border bg-secondary/40 px-2 py-1 text-xs text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
                      />
                      <div className="flex gap-1">
                        <Button size="sm" className="h-6 px-2 text-[10px]" onClick={() => void saveEdit()}>
                          Save
                        </Button>
                        <Button size="sm" variant="ghost" className="h-6 px-2 text-[10px]" onClick={() => setEditingId(null)}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  )}

                  {/* Tags display */}
                  {!isEditing && lora.tags && (
                    <div className="flex flex-wrap gap-1">
                      {lora.tags.split(',').map((tag) => tag.trim()).filter(Boolean).map((tag) => (
                        <Badge key={tag} variant="outline" className="text-[10px] px-1 py-0">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {/* Notes display */}
                  {!isEditing && lora.notes && (
                    <p className="line-clamp-2 text-[11px] text-muted-foreground/80">
                      {lora.notes}
                    </p>
                  )}

                  {/* Actions */}
                  {!isEditing && (
                    <div className="flex items-center gap-1 pt-1">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-7 gap-1.5 px-2 text-xs"
                        onClick={() => addLoraToForm(lora)}
                        title="Use this LoRA in Generate"
                      >
                        <Sparkles className="h-3 w-3" />
                        Use
                      </Button>
                      {lora.training_run && (
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          className="h-7 px-2 text-xs text-muted-foreground"
                          onClick={() => navigateToTab('train')}
                          title="View training run"
                        >
                          View Run
                        </Button>
                      )}
                      <div className="flex-1" />
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-xs text-muted-foreground hover:text-red-400"
                        onClick={() => setDeleteTarget(lora)}
                        title="Delete from library"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        {!isLoading && filtered.length === 0 && searchQuery && (
          <div className="py-12 text-center text-sm text-muted-foreground">
            No LoRAs matching &ldquo;{searchQuery}&rdquo;
          </div>
        )}
      </div>

      {/* Delete confirmation */}
      <DeleteConfirmDialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        title={`Remove ${deleteTarget?.name}?`}
        description="This will remove the LoRA from your library. The training run and files will not be deleted."
        loading={deleting}
        onConfirm={() => void handleDelete()}
        onCancel={() => setDeleteTarget(null)}
      />
    </>
  )
}
