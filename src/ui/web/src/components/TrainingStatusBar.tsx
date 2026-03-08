import { useQuery } from '@tanstack/react-query'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { api } from '../api'

function num(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toLocaleString()
}

function fixed(value?: number | null, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

export function TrainingStatusBar() {
  const { data = [] } = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 1000,
  })

  const runs = data.filter((run) => run.is_running)

  if (runs.length === 0) {
    return null
  }

  return (
    <div className="space-y-3">
      {runs.map((run) => {
        const pct = run.percent ?? 0
        const arch = [run.arch, run.trigger_word].filter(Boolean).join(' · ')

        return (
          <Card key={run.name} className="border-emerald-500/50 bg-emerald-500/5">
            <CardContent className="space-y-3 p-4">
              <div className="flex flex-wrap items-center gap-2">
                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                <span className="text-sm font-semibold">{run.name}</span>
                {arch ? <Badge variant="outline">{arch}</Badge> : null}
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-secondary">
                <div className="h-full bg-emerald-400 transition-all duration-700" style={{ width: `${pct}%` }} />
              </div>

              <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground">
                <span>
                  <strong className="text-foreground">{num(run.current_step)}</strong> / {num(run.total_steps)}
                </span>
                <span>
                  <strong className="text-foreground">{fixed(run.speed, 2)}</strong> it/s
                </span>
                <span>
                  elapsed <strong className="text-foreground">{run.elapsed ?? '—'}</strong>
                </span>
                <span>
                  eta <strong className="text-foreground">{run.eta ?? '—'}</strong>
                </span>
                <span>
                  loss <strong className="font-mono text-amber-300">{fixed(run.loss, 4)}</strong>
                </span>
                <span>
                  lr <strong className="text-foreground">{run.lr ?? '—'}</strong>
                </span>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
