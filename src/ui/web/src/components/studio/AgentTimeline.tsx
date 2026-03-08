import type { AgentEvent } from '../../api'
import { TimelineCard } from './TimelineCard'

type Props = {
  events: AgentEvent[]
}

/** Merge raw events into a deduplicated step list for display. */
function mergeSteps(events: AgentEvent[]) {
  const steps: { event: AgentEvent; isCompleted: boolean; key: string }[] = []
  const completedTools = new Set<string>()

  for (const ev of events) {
    if (ev.type === 'tool_complete' && ev.tool) {
      completedTools.add(ev.tool)
      // Update existing step
      const existing = steps.find((s) => s.event.tool === ev.tool && !s.isCompleted)
      if (existing) {
        existing.isCompleted = true
        existing.event = { ...existing.event, result: ev.result }
        continue
      }
    }

    if (ev.type === 'tool_progress') {
      // Update existing active step with progress
      const existing = steps.find((s) => s.event.tool === ev.tool && !s.isCompleted)
      if (existing) {
        existing.event = { ...existing.event, progress: ev.progress, detail: ev.detail }
        continue
      }
    }

    if (ev.type === 'thinking') {
      // Only show thinking if it's the first or contains meaningful content
      if (steps.length === 0 || (ev.message && ev.message.length > 30)) {
        steps.push({
          event: ev,
          isCompleted: true,
          key: `thinking-${steps.length}`,
        })
      }
      continue
    }

    if (ev.type === 'tool_start') {
      steps.push({
        event: ev,
        isCompleted: completedTools.has(ev.tool ?? ''),
        key: `${ev.tool}-${steps.length}`,
      })
      continue
    }

    if (ev.type === 'output_ready' || ev.type === 'error') {
      steps.push({
        event: ev,
        isCompleted: ev.type === 'output_ready',
        key: `${ev.type}-${steps.length}`,
      })
    }
  }

  return steps
}

export function AgentTimeline({ events }: Props) {
  const steps = mergeSteps(events)

  if (steps.length === 0) {
    return null
  }

  // The last non-completed step is active
  let lastActiveIndex = -1
  for (let i = steps.length - 1; i >= 0; i--) {
    if (!steps[i].isCompleted) {
      lastActiveIndex = i
      break
    }
  }

  return (
    <div className="space-y-2">
      {steps.map((step, i) => (
        <TimelineCard
          key={step.key}
          event={step.event}
          isActive={i === lastActiveIndex}
          isCompleted={step.isCompleted}
        />
      ))}
    </div>
  )
}
