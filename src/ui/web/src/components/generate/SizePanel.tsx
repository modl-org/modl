import { useState, useRef, useEffect } from 'react'
import { SIZE_PRESETS, detectSizePreset, type GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
}

/** Snap a value to the nearest multiple of 8, clamped to [128, 2048] */
function snapTo8(val: number): number {
  const clamped = Math.max(128, Math.min(2048, val))
  return Math.round(clamped / 8) * 8
}

/** Inline editable size value — click to type, blur/Enter to commit */
function EditableSize({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(String(value))
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) {
      setDraft(String(value))
      requestAnimationFrame(() => inputRef.current?.select())
    }
  }, [editing, value])

  const commit = () => {
    setEditing(false)
    const n = Number(draft)
    if (Number.isFinite(n) && n > 0) onChange(n)
  }

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="w-12 bg-transparent text-center text-xs text-foreground outline-none border-b border-primary/50"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setEditing(false) }}
      />
    )
  }

  return (
    <button
      type="button"
      className="cursor-text rounded px-1.5 py-0.5 text-xs tabular-nums text-foreground/80 hover:bg-secondary/50"
      onClick={() => setEditing(true)}
    >
      {value}
    </button>
  )
}

/** Tiny aspect ratio preview rectangle */
function AspectPreview({ w, h, active }: { w: number; h: number; active: boolean }) {
  const scale = 14
  return (
    <div
      className={`rounded-[2px] border-[1.5px] transition-colors ${active ? 'border-primary-foreground/70' : 'border-muted-foreground/30'}`}
      style={{
        width: Math.round(scale * Math.min(w / h, 1.4)),
        height: Math.round(scale * Math.min(h / w, 1.4)),
      }}
    />
  )
}

export function SizePanel({ form, setForm }: Props) {
  const activePreset = detectSizePreset(form.width, form.height)

  const applyPreset = (width: number, height: number) => {
    setForm((prev) => ({ ...prev, width, height }))
  }

  return (
    <div className="space-y-2.5">
      {/* Preset buttons with aspect ratio previews */}
      <div className="flex gap-1.5">
        {SIZE_PRESETS.map((preset) => {
          const isActive = activePreset === preset.label
          return (
            <button
              key={preset.label}
              type="button"
              className={`flex flex-1 flex-col items-center gap-1.5 rounded-lg border px-2 py-2.5 transition-colors ${
                isActive
                  ? 'border-primary bg-primary text-primary-foreground shadow-sm'
                  : 'border-border/40 bg-secondary/10 text-muted-foreground hover:bg-secondary/30'
              }`}
              onClick={() => applyPreset(preset.width, preset.height)}
            >
              <AspectPreview w={preset.width} h={preset.height} active={isActive} />
              <span className={`text-[10px] ${isActive ? 'font-semibold' : ''}`}>
                {preset.label}
              </span>
            </button>
          )
        })}
      </div>

      {/* Editable dimensions — tied below presets */}
      <div className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
        <EditableSize
          value={form.width}
          onChange={(v) => setForm((prev) => ({ ...prev, width: snapTo8(v) }))}
        />
        <span className="text-muted-foreground/40">&times;</span>
        <EditableSize
          value={form.height}
          onChange={(v) => setForm((prev) => ({ ...prev, height: snapTo8(v) }))}
        />
        <span className="text-[10px] text-muted-foreground/40">px</span>
      </div>
    </div>
  )
}
