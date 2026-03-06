import { Fragment, useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { api, type TrainingRun } from '../api'
import { LazyImage } from './LazyImage'

type ProcessConfig = {
  model?: {
    name_or_path?: string
    arch?: string
  }
  trigger_word?: string
  train?: {
    steps?: number
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

type DetailField = {
  label: string
  value: string | number
}

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
  if (!Array.isArray(candidate)) {
    return []
  }

  return candidate
    .filter((value): value is string => typeof value === 'string')
    .map((value) => value.trim())
    .filter((value) => value.length > 0)
}

function inferPromptCount(samples: TrainingRun['samples']): number {
  const counts = samples.map((sample) => sample.images.length).filter((count) => count > 0)
  if (counts.length === 0) {
    return 0
  }

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
  if (prompt.length <= 180) {
    return prompt
  }
  return `${prompt.slice(0, 177)}...`
}

function formatMegabytes(bytes?: number): string {
  if (!bytes || bytes <= 0) {
    return '—'
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function displayModelName(model?: string): string {
  if (!model) return '—'
  return model.split('/').pop() ?? model
}

function deriveRunStatus(run?: TrainingRun): { label: string; className: string } | null {
  const statuses = run?.lineage?.jobs?.map((job) => job.status.toLowerCase()) ?? []

  if (statuses.includes('running')) {
    return { label: 'Running', className: 'border-emerald-500/50 text-emerald-300' }
  }
  if (statuses.includes('interrupted')) {
    return { label: 'Interrupted', className: 'border-amber-500/50 text-amber-300' }
  }
  if (statuses.length > 0 && statuses.every((status) => status === 'completed')) {
    return { label: 'Completed', className: 'border-primary/50 text-primary' }
  }
  if (statuses.length > 0) {
    return { label: 'Attention', className: 'border-amber-500/50 text-amber-300' }
  }

  return null
}

type SampleLightboxImage = {
  src: string
  step: number
  promptIndex: number
  prompt: string
  runName: string
}

export function TrainingRuns() {
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [detailsOpenByRun, setDetailsOpenByRun] = useState<Record<string, boolean>>({})
  const [copiedRun, setCopiedRun] = useState<string | null>(null)
  const [lightboxImage, setLightboxImage] = useState<SampleLightboxImage | null>(null)

  const {
    data: runs = [],
    error: runsError,
    isLoading: runsLoading,
  } = useQuery({
    queryKey: ['runs'],
    queryFn: api.runs,
    staleTime: 5 * 60_000,
  })

  const { data: statusRuns = [] } = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 3000,
  })

  const runningNames = useMemo(
    () => new Set(statusRuns.filter((r) => r.is_running).map((r) => r.name)),
    [statusRuns],
  )

  const currentRun = selectedRun && runs.includes(selectedRun) ? selectedRun : (runs.at(-1) ?? null)

  const {
    data: runDetail,
    error: detailError,
    isLoading: detailLoading,
  } = useQuery({
    queryKey: ['run', currentRun],
    queryFn: () => api.run(currentRun as string),
    enabled: Boolean(currentRun),
    staleTime: 5 * 60_000,
  })

  const detailsOpen = runDetail ? (detailsOpenByRun[runDetail.name] ?? false) : false
  const processCfg = useMemo(() => extractProcessConfig(runDetail?.config), [runDetail?.config])
  const samplePrompts = useMemo(() => extractSamplePrompts(runDetail?.config), [runDetail?.config])

  const baseModelRaw = runDetail?.lineage?.base_model ?? processCfg?.model?.name_or_path
  const baseModelName = displayModelName(baseModelRaw)
  const triggerWord = processCfg?.trigger_word
  const runStatus = deriveRunStatus(runDetail)

  const promptCount = useMemo(() => {
    if (!runDetail) return 0
    return Math.max(samplePrompts.length, inferPromptCount(runDetail.samples))
  }, [runDetail, samplePrompts])

  // Calculate expected total sample columns from config (steps / sample_every)
  const expectedColumns = useMemo(() => {
    if (!runDetail) return 0
    const totalSteps = processCfg?.train?.steps
    const sampleEvery = processCfg?.sample?.sample_every
    if (totalSteps && sampleEvery && sampleEvery > 0) {
      return Math.ceil(totalSteps / sampleEvery)
    }
    return runDetail.samples.length
  }, [runDetail, processCfg])

  const totalColumns = Math.max(expectedColumns, runDetail?.samples.length ?? 0)

  const promptRows = useMemo(() => {
    if (!runDetail || promptCount === 0) {
      return []
    }

    // Pad imagesByStep to totalColumns so future steps show placeholders
    return Array.from({ length: promptCount }, (_, idx) => ({
      id: idx,
      label: samplePrompts[idx] ?? `Prompt ${idx + 1}`,
      imagesByStep: Array.from({ length: totalColumns }, (_, col) =>
        runDetail.samples[col]?.images[idx] ?? null,
      ),
    }))
  }, [runDetail, promptCount, samplePrompts, totalColumns])

  const detailFields: DetailField[] = useMemo(() => {
    if (!runDetail || !processCfg) {
      return []
    }

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

    return fields.filter((field): field is DetailField => field !== null)
  }, [processCfg, runDetail])

  const copyTrigger = async () => {
    if (!runDetail || !triggerWord) {
      return
    }

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

  return (
    <>
    <div className="flex min-h-[520px] overflow-hidden">
      {/* Left: runs list */}
      <div className="flex w-48 shrink-0 flex-col border-r border-border/60">
        <div className="border-b border-border px-4 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
            Training Runs
          </p>
        </div>
        <div className="flex-1 overflow-y-auto py-2">
          {runs.map((name) => {
            const isRunning = runningNames.has(name)
            return (
              <button
                key={name}
                type="button"
                onClick={() => setSelectedRun(name)}
                className={`flex w-full items-center gap-2 px-4 py-2 text-left text-sm transition-colors ${
                  name === currentRun
                    ? 'bg-primary/10 font-medium text-primary'
                    : 'text-muted-foreground hover:bg-accent hover:text-foreground'
                }`}
              >
                {isRunning ? (
                  <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-emerald-400" />
                ) : null}
                <span className="truncate">{name}</span>
              </button>
            )
          })}
          {!runsLoading && runs.length === 0 && !runsError ? (
            <p className="px-4 py-3 text-xs text-muted-foreground">No training runs found.</p>
          ) : null}
          {runsError ? (
            <p className="px-4 py-3 text-xs text-destructive">Failed to load runs.</p>
          ) : null}
        </div>
      </div>

      {/* Right: detail */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {detailError ? (
          <div className="p-6 text-sm text-destructive">Failed to load run details: {String(detailError)}</div>
        ) : !runDetail && !detailError ? (
          <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
            {detailLoading ? 'Loading…' : 'Select a training run to view details.'}
          </div>
        ) : runDetail ? (
          <div className="flex flex-1 flex-col overflow-y-auto">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div className="min-w-0">
                <h2 className="truncate text-sm font-semibold text-foreground">{runDetail.name}</h2>
                {runDetail.lineage?.dataset_name ? (
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Dataset:{' '}
                    <a
                      href={`/?tab=datasets&dataset=${encodeURIComponent(runDetail.lineage.dataset_name)}`}
                      className="text-primary hover:underline"
                    >
                      {runDetail.lineage.dataset_name}
                    </a>
                  </p>
                ) : null}
              </div>
              <div className="flex items-center gap-2">
                {runStatus ? (
                  <Badge variant="outline" className={runStatus.className}>
                    {runStatus.label}
                  </Badge>
                ) : null}
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                  onClick={() =>
                    setDetailsOpenByRun((prev) => ({
                      ...prev,
                      [runDetail.name]: !detailsOpen,
                    }))
                  }
                >
                  {detailsOpen ? 'Hide details' : 'Details'}
                </Button>
              </div>
            </div>

            {/* Info rows */}
            <div className="grid grid-cols-2 divide-x divide-border border-b border-border">
              <div className="px-4 py-3">
                <p className="mb-1 text-[10px] uppercase tracking-widest text-muted-foreground">Base model</p>
                <p className="truncate text-sm font-medium text-foreground" title={baseModelRaw ?? '—'}>
                  {baseModelName}
                </p>
                {baseModelRaw && baseModelRaw !== baseModelName ? (
                  <p className="mt-0.5 truncate text-xs text-muted-foreground" title={baseModelRaw}>
                    {baseModelRaw}
                  </p>
                ) : null}
              </div>
              <div className="px-4 py-3">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <p className="text-[10px] uppercase tracking-widest text-muted-foreground">Trigger word</p>
                  <Badge variant="outline" className="text-[10px]">
                    {formatMegabytes(runDetail.lora_size)}
                  </Badge>
                </div>
                <div className="flex items-center gap-2">
                  <span className="truncate text-sm font-medium text-foreground" title={triggerWord ?? '—'}>
                    {triggerWord ? `"${triggerWord}"` : '—'}
                  </span>
                  {triggerWord ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="ghost"
                      className="ml-auto h-6 shrink-0 px-2 text-xs text-muted-foreground"
                      onClick={() => void copyTrigger()}
                    >
                      {copiedRun === runDetail.name ? 'Copied' : 'Copy'}
                    </Button>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Expandable details */}
            {detailsOpen && detailFields.length > 0 ? (
              <div className="border-b border-border px-4 py-3">
                <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                  Training config
                </p>
                <dl className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-3 lg:grid-cols-4">
                  {detailFields.map((field) => (
                    <div key={field.label}>
                      <dt className="text-[10px] uppercase tracking-wide text-muted-foreground">{field.label}</dt>
                      <dd className="mt-0.5 text-sm text-foreground">{field.value}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ) : null}

            {/* Sample evolution */}
            <div className="flex-1 px-4 py-4">
              <p className="mb-3 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                Sample Evolution
              </p>
              {runDetail.samples.length === 0 || promptRows.length === 0 ? (
                <p className="text-sm text-muted-foreground">No sample images generated yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <div
                    className="grid gap-2"
                    style={{
                      gridTemplateColumns: `repeat(${totalColumns}, minmax(120px, 1fr))`,
                    }}
                  >
                    {Array.from({ length: totalColumns }, (_, idx) => {
                      const sample = runDetail.samples[idx]
                      const sampleEvery = processCfg?.sample?.sample_every
                      const stepLabel = sample
                        ? sample.step.toLocaleString()
                        : sampleEvery
                          ? ((idx + 1) * sampleEvery).toLocaleString()
                          : '—'
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

                    {promptRows.map((row) => (
                      <Fragment key={row.id}>
                        <div className="col-span-full rounded border border-border/60 bg-secondary/20 px-3 py-2">
                          <div className="mb-0.5 text-[10px] uppercase tracking-wide text-muted-foreground">
                            Prompt {row.id + 1}
                          </div>
                          <div className="text-xs leading-relaxed text-foreground" title={row.label}>
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
                                onClick={() =>
                                  setLightboxImage({
                                    src: `/files/${image}`,
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
        ) : null}
      </div>
    </div>

    {/* Sample image lightbox */}
    <Dialog open={Boolean(lightboxImage)} onOpenChange={(open) => !open && setLightboxImage(null)}>
      <DialogContent className="max-w-[96vw] gap-0 p-0 sm:max-w-4xl">
        {lightboxImage ? (
          <div className="grid max-h-[90vh] gap-0 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div className="flex items-center justify-center overflow-auto bg-black/70 p-4">
              <img
                src={lightboxImage.src}
                alt={`Step ${lightboxImage.step} — Prompt ${lightboxImage.promptIndex + 1}`}
                className="max-h-[80vh] rounded object-contain"
              />
            </div>
            <div className="flex flex-col border-l border-border/70 bg-card">
              <DialogHeader className="px-4 pt-4">
                <DialogTitle className="text-sm">Sample Details</DialogTitle>
                <DialogDescription className="text-xs">{lightboxImage.runName}</DialogDescription>
              </DialogHeader>
              <div className="flex-1 overflow-auto px-4 py-3">
                <div className="grid grid-cols-[80px_minmax(0,1fr)] gap-x-3 gap-y-3 text-xs">
                  <div className="text-muted-foreground">Step</div>
                  <div className="font-mono text-foreground">
                    {lightboxImage.step.toLocaleString()}
                  </div>
                  <div className="text-muted-foreground">Prompt&nbsp;#</div>
                  <div className="font-mono text-foreground">{lightboxImage.promptIndex + 1}</div>
                  <div className="text-muted-foreground">Prompt</div>
                  <div className="break-words leading-relaxed text-foreground">
                    {lightboxImage.prompt}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
    </>
  )
}
