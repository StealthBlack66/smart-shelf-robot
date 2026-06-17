import type { ReactNode } from 'react'

export function Panel({ title, right, children, className }: {
  title: string; right?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`panel ${className ?? ''}`}>
      <header>
        {title}
        {right && <span className="badge">{right}</span>}
      </header>
      <div className="body">{children}</div>
    </section>
  )
}

export function Pill({ ok, label, warn }: { ok: boolean; label: string; warn?: boolean }) {
  const cls = ok ? 'ok' : warn ? 'warn' : 'err'
  return (
    <span className={`pill ${cls}`}>
      <span className={`dot ${cls}`} />
      {label}
    </span>
  )
}

export function KV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="kv">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  )
}

export function Stat({ label, value, tone }: {
  label: string; value: ReactNode; tone?: 'ok' | 'warn' | 'err'
}) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className={`num ${tone ?? ''}`}>{value}</div>
    </div>
  )
}

export function Gauge({ label, value, max, unit, color }: {
  label: string; value: number; max: number; unit: string; color: string
}) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100))
  return (
    <div className="gauge">
      <div className="head">
        <span>{label}</span>
        <span style={{ fontFamily: 'var(--mono)' }}>
          {value} / {max} {unit}
        </span>
      </div>
      <div className="track">
        <div className="fill" style={{ width: `${pct}%`, background: color }} />
      </div>
    </div>
  )
}
