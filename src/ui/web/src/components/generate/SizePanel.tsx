import { useState, useRef, useEffect } from 'react'
import { SquareIcon, RectangleHorizontalIcon, RectangleVerticalIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
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

/** Map preset aspect label to an icon */
function presetIcon(label: string) {
  if (label === '1:1') return <SquareIcon className="size-3" />
  if (label === '3:4' || label === '9:16') return <RectangleVerticalIcon className="size-3" />
  return <RectangleHorizontalIcon className="size-3" />
}

export function SizePanel({ form, setForm }: Props) {
  const activePreset = detectSizePreset(form.width, form.height)

  const applyPreset = (width: number, height: number) => {
    setForm((prev) => ({ ...prev, width, height }))
  }

  return (
    <div className="space-y-2.5">
      {/* Preset buttons with icons */}
      <div className="flex gap-1">
        {SIZE_PRESETS.map((preset) => (
          <Button
            key={preset.label}
            type="button"
            size="sm"
            variant={activePreset === preset.label ? 'secondary' : 'outline'}
            className="h-7 flex-1 gap-1 px-1 text-[11px]"
            onClick={() => applyPreset(preset.width, preset.height)}
          >
            {presetIcon(preset.label)}
            {preset.label}
          </Button>
        ))}
      </div>

      {/* Dimensions display — click to edit */}
      <div className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground">
        <EditableSize
          value={form.width}
          onChange={(v) => setForm((prev) => ({ ...prev, width: snapTo8(v) }))}
        />
        <span className="text-muted-foreground/40">×</span>
        <EditableSize
          value={form.height}
          onChange={(v) => setForm((prev) => ({ ...prev, height: snapTo8(v) }))}
        />
        <span className="text-[10px] text-muted-foreground/40">px</span>
      </div>
    </div>
  )
}
