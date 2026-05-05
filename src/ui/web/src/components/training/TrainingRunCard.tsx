import { Fragment, useMemo, useState, memo } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Play,
  Pause,
  Trash2,
  SlidersHorizontal,
} from 'lucide-react'
import {
  api,
  type TrainingRun,
  type TrainingStatusRun,
} from '../../api'
import { LazyImage } from '../LazyImage'
import { displayModelName, formatBytes } from '@/lib/utils'
import { STALE_REALTIME, STALE_SLOW } from '@/lib/query-keys'
import { LossChart } from './LossChart'
import { PromotionDialog } from './PromotionDialog'
import type { SampleLightboxImage } from './SampleLightbox'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ProcessConfig = {
  model?: {
    name_or_path?: string
    arch?: string
  }
  trigger_word?: string
  train?: {
    steps?: number
    start_step?: number
    lr?: number
    batch_size?: number
    optimizer?: string
  }
  network?: {
    linear?: number
    linear_alpha?: number
  }
  sample?: {
    sample_every?: number
    prompts?: unknown
  }
}

export type DetailField = {
  label: string
  value: string | number
}

export type TrainingRunCardProps = {
  runName: string
  isRunning: boolean
  gpuBusy: boolean
  currentStatus: TrainingStatusRun | undefined
  onSetLightboxImage: (image: SampleLightboxImage) => void
  onSelectRun: (name: string | null) => void
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractProcessConfig(config?: Record<string, unknown>): ProcessConfig | null {
  const candidates: unknown[] = [
    (config as { config?: { process?: unknown[] } } | undefined)?.config?.process?.[0],
    (config as { process?: unknown[] } | undefined)?.process?.[0],
    config,
  ]
  for (const candidate of candidates) {
    if (candidate && typeof candidate === 'object') {
      return candidate as ProcessConfig
    }
  }
  return null
}

function extractSamplePrompts(config?: Record<string, unknown>): string[] {
  const processCfg = extractProcessConfig(config)
  const candidate = processCfg?.sample?.prompts
  if (!Array.isArray(candidate)) return []
  return candidate
    .filter((value): value is string => typeof value === 'string')
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
}

function inferPromptCount(samples: TrainingRun['samples']): number {
  const counts = samples.map((s) => s.images.length).filter((c) => c > 0)
  if (counts.length === 0) return 0
  const frequencies = new Map<number, number>()
  for (const count of counts) {
    frequencies.set(count, (frequencies.get(count) ?? 0) + 1)
  }
  let modeCount = counts[0]
  let modeFrequency = frequencies.get(modeCount) ?? 0
  for (const [count, frequency] of frequencies) {
    if (frequency > modeFrequency || (frequency === modeFrequency && count > modeCount)) {
      modeCount = count
      modeFrequency = frequency
    }
  }
  return modeCount
}

function shortenPrompt(prompt: string): string {
  return prompt.length <= 180 ? prompt : `${prompt.slice(0, 177)}...`
}

function deriveRunStatus(run?: TrainingRun, isLiveRunning?: boolean): string {
  if (isLiveRunning) return 'running'

  // If a final LoRA file exists on disk, training completed regardless
  // of what stale DB job records say.
  if (run?.lora_path) return 'completed'

  const statuses = run?.lineage?.jobs?.map((job) => job.status.toLowerCase()) ?? []
  if (statuses.length === 0) {
    return 'unknown'
  }
  // Prefer "completed" if any job completed (handles stale "running" jobs)
  if (statuses.includes('completed')) return 'completed'
  const latest = statuses[statuses.length - 1]
  if (latest === 'running') {
    return 'interrupted'
  }
  return latest
}

const STATUS_INFO: Record<string, { label: string; dot: string; badge: string }> = {
  running:     { label: 'Running',     dot: 'bg-emerald-400', badge: 'border-emerald-500/50 text-emerald-300' },
  completed:   { label: 'Completed',   dot: 'bg-primary',     badge: 'border-primary/50 text-primary' },
  interrupted: { label: 'Interrupted', dot: 'bg-amber-400',   badge: 'border-amber-500/50 text-amber-300' },
  failed:      { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  error:       { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  cancelled:   { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  unknown:     { label: 'Unknown',     dot: 'bg-muted-foreground/40', badge: 'border-border text-muted-foreground' },
}

function getStatusInfo(status: string) {
  return STATUS_INFO[status] ?? STATUS_INFO.unknown
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const TrainingRunCard = memo(function TrainingRunCard({
  runName,
  isRunning,
  gpuBusy,
  currentStatus,
  onSetLightboxImage,
  onSelectRun,
}: TrainingRunCardProps) {
  const queryClient = useQueryClient()

  // Local UI state
  const [detailsOpen, setDetailsOpen] = useState(false)
  const [copiedRun, setCopiedRun] = useState<string | null>(null)
  const [resuming, setResuming] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const {
    data: runDetail,
    error: detailError,
    isLoading: detailLoading,
  } = useQuery({
    queryKey: ['run', runName],
    queryFn: () => api.run(runName),
    enabled: Boolean(runName),
    staleTime: isRunning ? STALE_REALTIME : STALE_SLOW,
    refetchInterval: isRunning ? 15_000 : false,
  })

  const { data: lossPoints = [] } = useQuery({
    queryKey: ['loss', runName],
    queryFn: () => api.lossHistory(runName),
    enabled: Boolean(runName),
    staleTime: STALE_REALTIME,
    refetchInterval: isRunning ? 5000 : false,
  })

  // -----------------------------------------------------------------------
  // Derived state
  // -----------------------------------------------------------------------

  const processCfg = useMemo(() => extractProcessConfig(runDetail?.config), [runDetail?.config])
  const samplePrompts = useMemo(() => extractSamplePrompts(runDetail?.config), [runDetail?.config])

  const baseModelRaw = runDetail?.lineage?.base_model ?? processCfg?.model?.name_or_path
  const baseModelName = displayModelName(baseModelRaw)
  const triggerWord = processCfg?.trigger_word
  const runStatus = deriveRunStatus(runDetail, isRunning)
  const latestCheckpoint = currentStatus?.latest_checkpoint
  const canResume = !isRunning && latestCheckpoint && runStatus === 'interrupted'

  const promptCount = useMemo(() => {
    if (!runDetail) return 0
    return Math.max(samplePrompts.length, inferPromptCount(runDetail.samples))
  }, [runDetail, samplePrompts])

  const expectedColumns = useMemo(() => {
    if (!runDetail) return 0
    const sampleEvery = runDetail.sample_every ?? processCfg?.sample?.sample_every
    const totalSteps = runDetail.total_steps
    if (totalSteps && sampleEvery && sampleEvery > 0) {
      return Math.ceil(totalSteps / sampleEvery)
    }
    if (sampleEvery && sampleEvery > 0) {
      const phaseSteps = processCfg?.train?.steps
      if (phaseSteps && phaseSteps > 0) return Math.ceil(phaseSteps / sampleEvery)
    }
    return runDetail.samples.length
  }, [runDetail, processCfg])

  const totalColumns = Math.max(expectedColumns, runDetail?.samples.length ?? 0)

  const promptRows = useMemo(() => {
    if (!runDetail || promptCount === 0) return []
    return Array.from({ length: promptCount }, (_, idx) => ({
      id: idx,
      label: samplePrompts[idx] ?? `Prompt ${idx + 1}`,
      imagesByStep: Array.from({ length: totalColumns }, (_, col) =>
        runDetail.samples[col]?.images[idx] ?? null,
      ),
    }))
  }, [runDetail, promptCount, samplePrompts, totalColumns])

  const detailFields: DetailField[] = useMemo(() => {
    if (!runDetail || !processCfg) return []
    const fields: Array<DetailField | null> = [
      processCfg.model?.arch ? { label: 'Architecture', value: processCfg.model.arch } : null,
      processCfg.train?.steps ? { label: 'Train Steps', value: processCfg.train.steps } : null,
      processCfg.sample?.sample_every
        ? { label: 'Sample Every', value: processCfg.sample.sample_every }
        : null,
      processCfg.train?.lr ? { label: 'Learning Rate', value: processCfg.train.lr } : null,
      processCfg.train?.batch_size ? { label: 'Batch Size', value: processCfg.train.batch_size } : null,
      processCfg.train?.optimizer ? { label: 'Optimizer', value: processCfg.train.optimizer } : null,
      processCfg.network?.linear ? { label: 'LoRA Rank', value: processCfg.network.linear } : null,
      processCfg.network?.linear_alpha ? { label: 'LoRA Alpha', value: processCfg.network.linear_alpha } : null,
      runDetail.lineage?.dataset_name
        ? { label: 'Dataset', value: runDetail.lineage.dataset_name }
        : null,
      runDetail.lineage?.dataset_image_count
        ? { label: 'Dataset Images', value: runDetail.lineage.dataset_image_count }
        : null,
    ]
    return fields.filter((f): f is DetailField => f !== null)
  }, [processCfg, runDetail])

  // -----------------------------------------------------------------------
  // Actions
  // -----------------------------------------------------------------------

  const copyTrigger = async () => {
    if (!runDetail || !triggerWord) return
    try {
      await navigator.clipboard.writeText(triggerWord)
      setCopiedRun(runDetail.name)
      window.setTimeout(() => {
        setCopiedRun((prev) => (prev === runDetail.name ? null : prev))
      }, 1200)
    } catch {
      // Ignore clipboard failures.
    }
  }

  // -----------------------------------------------------------------------
  // Render guards
  // -----------------------------------------------------------------------

  if (detailError) {
    return (
      <div className="p-6 text-sm text-destructive">
        Failed to load run details: {String(detailError)}
      </div>
    )
  }

  if (!runDetail && !detailError) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
        {detailLoading ? (
          <p className="text-sm text-muted-foreground">Loading...</p>
        ) : (
          <>
            <SlidersHorizontal className="h-8 w-8 text-muted-foreground/30" />
            <p className="text-sm text-muted-foreground">
              Select a training run to view details
            </p>
          </>
        )}
      </div>
    )
  }

  if (!runDetail) return null

  const statusInfo = getStatusInfo(runStatus)

  return (
    <div className="flex flex-1 flex-col overflow-y-auto">
      {/* ======= Detail Header ======= */}
      <div className="border-b border-border px-4 py-3 space-y-2">
        {/* Row 1: Name + Status */}
        <div className="flex items-center gap-3">
          <h2 className="min-w-0 truncate text-base font-semibold text-foreground">
            {runDetail.name}
          </h2>
          <Badge variant="outline" className={`shrink-0 ${statusInfo.badge}`}>
            {statusInfo.label}
          </Badge>
          {isRunning && currentStatus?.is_sampling ? (
            <span className="shrink-0 text-xs text-amber-400">
              generating samples…
            </span>
          ) : isRunning && currentStatus?.current_step != null && currentStatus?.total_steps ? (
            <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
              {currentStatus.current_step.toLocaleString()}/{currentStatus.total_steps.toLocaleString()}
              {' '}
              <span className="text-emerald-400">
                ({Math.round((currentStatus.current_step / currentStatus.total_steps) * 100)}%)
              </span>
            </span>
          ) : null}
        </div>

        {/* Row 2: Metadata */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span title={baseModelRaw ?? '\u2014'}>{baseModelName}</span>
          {triggerWord && (
            <>
              <span className="text-border">&middot;</span>
              <button
                type="button"
                className="font-mono text-foreground hover:text-primary"
                title="Click to copy trigger word"
                onClick={() => void copyTrigger()}
              >
                {copiedRun === runDetail.name ? 'Copied!' : triggerWord}
              </button>
            </>
          )}
          {runDetail.lora_size ? (
            <>
              <span className="text-border">&middot;</span>
              <span>{formatBytes(runDetail.lora_size)}</span>
            </>
          ) : null}
          {runDetail.lineage?.dataset_name ? (
            <>
              <span className="text-border">&middot;</span>
              <a
                href={`/?tab=datasets&dataset=${encodeURIComponent(runDetail.lineage.dataset_name)}`}
                className="text-primary hover:underline"
              >
                {runDetail.lineage.dataset_name}
              </a>
            </>
          ) : null}
          {isRunning && currentStatus?.speed ? (
            <>
              <span className="text-border">&middot;</span>
              <span>{currentStatus.speed.toFixed(2)} it/s</span>
            </>
          ) : null}
          {isRunning && currentStatus?.eta ? (
            <>
              <span className="text-border">&middot;</span>
              <span>ETA {currentStatus.eta}</span>
            </>
          ) : null}
        </div>

        {/* Row 3: Actions */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Promote / Save to Library */}
          {!isRunning && (
            <PromotionDialog
              runDetail={runDetail}
              triggerWord={triggerWord}
              baseModelRaw={baseModelRaw}
              currentRunKey={runName}
            />
          )}

          {/* Pause */}
          {isRunning && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1.5 px-2 text-xs border-red-500/50 text-red-400 hover:bg-red-500/10"
              disabled={cancelling}
              title="Pause training -- you can resume later"
              onClick={async () => {
                setCancelling(true)
                try {
                  await api.cancelTraining(runDetail.name)
                  void queryClient.invalidateQueries({ queryKey: ['status'] })
                  void queryClient.invalidateQueries({ queryKey: ['run', runName] })
                  void queryClient.invalidateQueries({ queryKey: ['runs'] })
                } catch (err) {
                  console.error('Pause failed:', err)
                } finally {
                  setCancelling(false)
                }
              }}
            >
              <Pause className="h-3 w-3" />
              {cancelling ? 'Pausing...' : 'Pause'}
            </Button>
          )}

          {/* Resume */}
          {canResume && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 gap-1.5 px-2 text-xs"
              disabled={resuming || gpuBusy}
              title={gpuBusy ? 'GPU is busy with another training run' : 'Resume training from last checkpoint'}
              onClick={async () => {
                setResuming(true)
                try {
                  await api.resumeTraining(runDetail.name, latestCheckpoint!)
                } catch (err) {
                  console.error('Resume failed:', err)
                } finally {
                  setResuming(false)
                }
              }}
            >
              <Play className="h-3 w-3" />
              {resuming ? 'Resuming...' : gpuBusy ? 'GPU busy' : 'Resume'}
            </Button>
          )}

          <div className="flex-1" />

          {/* Details toggle */}
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
            onClick={() => setDetailsOpen((prev) => !prev)}
          >
            {detailsOpen ? 'Hide details' : 'Details'}
          </Button>

          {/* Delete */}
          {!isRunning &&
            (confirmDelete === runDetail.name ? (
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs border-red-500/50 text-red-400 hover:bg-red-500/10"
                  disabled={deleting}
                  onClick={async () => {
                    setDeleting(true)
                    try {
                      await api.deleteRun(runDetail.name)
                      setConfirmDelete(null)
                      onSelectRun(null)
                      void queryClient.invalidateQueries({ queryKey: ['runs'] })
                    } catch (err) {
                      console.error('Delete failed:', err)
                      alert(`Delete failed: ${err instanceof Error ? err.message : err}`)
                    } finally {
                      setDeleting(false)
                    }
                  }}
                >
                  {deleting ? 'Deleting...' : 'Confirm'}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2 text-xs"
                  onClick={() => setConfirmDelete(null)}
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="h-7 gap-1.5 px-2 text-xs text-muted-foreground hover:text-red-400"
                title="Delete training run"
                onClick={() => setConfirmDelete(runDetail.name)}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            ))}
        </div>
      </div>

      {/* ======= Expandable details ======= */}
      {detailsOpen && (
        <div className="border-b border-border px-4 py-3 space-y-4">
          {detailFields.length > 0 && (
            <div>
              <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                Training config
              </p>
              <dl className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-3 lg:grid-cols-4">
                {detailFields.map((field) => (
                  <div key={field.label}>
                    <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">
                      {field.label}
                    </dt>
                    <dd className="mt-0.5 text-sm text-foreground">{field.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
          {lossPoints.length > 1 && (
            <div>
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                Training Loss
              </p>
              <LossChart points={lossPoints} />
            </div>
          )}
        </div>
      )}

      {/* ======= Sample Evolution ======= */}
      <div className="flex-1 px-4 py-3">
        <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          Sample Evolution
        </p>
        {runDetail.samples.length === 0 || promptRows.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            {isRunning ? (
              <>
                <div className="mb-3 h-8 w-8 animate-pulse rounded-full bg-emerald-400/20" />
                <p className="text-sm text-muted-foreground">
                  Generating first sample
                  {processCfg?.sample?.sample_every
                    ? ` at step ${processCfg.sample.sample_every.toLocaleString()}`
                    : ''}
                  ...
                </p>
                {currentStatus?.current_step != null && currentStatus?.total_steps ? (
                  <p className="mt-1 text-xs text-muted-foreground/60">
                    Currently at step {currentStatus.current_step.toLocaleString()}
                  </p>
                ) : null}
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No sample images were generated during this run.
              </p>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `repeat(${totalColumns}, minmax(120px, 1fr))`,
              }}
            >
              {/* Step headers */}
              {Array.from({ length: totalColumns }, (_, idx) => {
                const sample = runDetail.samples[idx]
                const sampleEvery = processCfg?.sample?.sample_every
                const stepLabel = sample
                  ? sample.step.toLocaleString()
                  : sampleEvery
                    ? ((idx + 1) * sampleEvery).toLocaleString()
                    : '\u2014'
                return (
                  <div
                    key={idx}
                    className={`rounded px-2 py-1.5 text-center text-[10px] font-medium uppercase tracking-wide ${
                      sample
                        ? 'bg-secondary/60 text-muted-foreground'
                        : 'bg-secondary/20 text-muted-foreground/40'
                    }`}
                  >
                    Step {stepLabel}
                  </div>
                )
              })}

              {/* Prompt rows */}
              {promptRows.map((row) => (
                <Fragment key={row.id}>
                  <div className="col-span-full rounded border border-border/60 bg-secondary/20 px-3 py-2">
                    <div className="mb-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                      Prompt {row.id + 1}
                    </div>
                    <div
                      className="text-xs leading-relaxed text-foreground"
                      title={row.label}
                    >
                      {shortenPrompt(row.label)}
                    </div>
                  </div>

                  {row.imagesByStep.map((image, idx) => (
                    <div
                      key={`${row.id}-${idx}`}
                      className="overflow-hidden rounded border border-border/50"
                    >
                      {image ? (
                        <LazyImage
                          src={`/files/${image}`}
                          alt={`Prompt ${row.id + 1} - step ${runDetail.samples[idx]?.step}`}
                          className="aspect-square"
                          thumbWidth={300}
                          onClick={() =>
                            onSetLightboxImage({
                              src: `/files/${image}?w=1920`,
                              step: runDetail.samples[idx]?.step ?? 0,
                              promptIndex: row.id,
                              prompt: row.label,
                              runName: runDetail.name,
                            })
                          }
                        />
                      ) : (
                        <div className="aspect-square w-full rounded bg-secondary/20 border border-dashed border-border/30" />
                      )}
                    </div>
                  ))}
                </Fragment>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
})
