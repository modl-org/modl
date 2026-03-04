import { AlertTriangleIcon, CheckCircle2Icon, LoaderCircleIcon, RadioIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'

export type GenerateState =
  | { status: 'idle' }
  | { status: 'submitting' }
  | { status: 'streaming'; lines: string[] }
  | { status: 'done'; count: number }
  | { status: 'error'; message: string }

type Props = {
  state: GenerateState
  onViewOutputs: () => void
  onReset: () => void
}

export function GenerateProgress({ state, onViewOutputs, onReset }: Props) {
  if (state.status === 'idle') {
    return null
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {state.status === 'submitting' ? <LoaderCircleIcon className="size-4 animate-spin" /> : null}
          {state.status === 'streaming' ? <RadioIcon className="size-4 text-emerald-400" /> : null}
          {state.status === 'done' ? <CheckCircle2Icon className="size-4 text-emerald-400" /> : null}
          {state.status === 'error' ? <AlertTriangleIcon className="size-4 text-destructive" /> : null}
          Generate Progress
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {state.status === 'submitting' ? <p className="text-sm text-muted-foreground">Submitting job...</p> : null}

        {state.status === 'streaming' ? (
          <ScrollArea className="h-56 rounded-md border border-border/80 bg-background p-3">
            <pre className="whitespace-pre-wrap font-mono text-xs text-muted-foreground">
              {state.lines.join('\n') || 'Waiting for stream...'}
            </pre>
          </ScrollArea>
        ) : null}

        {state.status === 'done' ? (
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm text-emerald-300">Generated {state.count} image(s).</p>
            <Button type="button" size="sm" onClick={onViewOutputs}>
              View in Outputs
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={onReset}>
              Dismiss
            </Button>
          </div>
        ) : null}

        {state.status === 'error' ? (
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm text-destructive">{state.message}</p>
            <Button type="button" size="sm" variant="ghost" onClick={onReset}>
              Dismiss
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  )
}
