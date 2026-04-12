import { Film } from 'lucide-react'
import { Slider } from '@/components/ui/slider'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

/** Valid frame counts must be 8*N+1 (LTX requirement) */
const FRAME_PRESETS = [25, 49, 73, 97, 121, 145] as const
const FPS_OPTIONS = [24, 25, 30] as const

function nearestValidFrames(n: number): number {
  // Snap to nearest 8*N+1
  const k = Math.max(0, Math.round((n - 1) / 8))
  return k * 8 + 1
}

export function VideoPanel({ form, setForm }: Props) {
  const duration = form.fps > 0 ? form.num_frames / form.fps : 0

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        <Film className="size-3.5 text-muted-foreground" />
        <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide">
          Video
        </span>
      </div>

      {/* Frames */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">Frames</span>
          <span className="font-mono text-[11px] text-foreground/70">{form.num_frames}</span>
        </div>
        <Slider
          value={[form.num_frames]}
          min={9}
          max={201}
          step={8}
          onValueChange={([v]) => {
            const frames = nearestValidFrames(v ?? 121)
            setForm((prev) => ({ ...prev, num_frames: frames }))
          }}
        />
        <div className="flex gap-1 pt-0.5">
          {FRAME_PRESETS.map((f) => (
            <button
              key={f}
              type="button"
              className={`rounded px-1.5 py-0.5 text-[10px] transition-colors ${
                form.num_frames === f
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
              onClick={() => setForm((prev) => ({ ...prev, num_frames: f }))}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* FPS */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium text-muted-foreground">FPS</span>
          <span className="font-mono text-[11px] text-foreground/70">{form.fps}</span>
        </div>
        <div className="flex gap-1">
          {FPS_OPTIONS.map((f) => (
            <button
              key={f}
              type="button"
              className={`rounded px-2 py-1 text-[11px] font-medium transition-colors ${
                form.fps === f
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
              onClick={() => setForm((prev) => ({ ...prev, fps: f }))}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Duration readout */}
      <div className="flex items-center justify-between rounded bg-muted/50 px-2 py-1">
        <span className="text-[11px] text-muted-foreground">Duration</span>
        <span className="font-mono text-[11px] font-medium text-foreground/80">
          {duration.toFixed(1)}s
        </span>
      </div>
    </div>
  )
}
