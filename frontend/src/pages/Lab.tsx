import { Activity, AlertTriangle, CheckCircle2, Clock, Download, Gauge, Layers } from 'lucide-react'
import { mockRobustness, mockRuns, type RunStatus } from '@/data/mock/lab'
import { cn } from '@/lib/utils'

function statusChip(status: RunStatus) {
  switch (status) {
    case 'complete':
      return { label: 'COMPLETE', icon: <CheckCircle2 size={14} />, className: 'text-cyan' }
    case 'running':
      return { label: 'RUNNING', icon: <Activity size={14} />, className: 'text-amber' }
    case 'queued':
      return { label: 'QUEUED', icon: <Clock size={14} />, className: 'text-muted' }
    case 'failed':
      return { label: 'FAILED', icon: <AlertTriangle size={14} />, className: 'text-danger' }
  }
}

function Metric(props: { label: string; value: string; accent?: 'amber' | 'cyan' }) {
  const accent = props.accent === 'amber' ? 'text-amber' : props.accent === 'cyan' ? 'text-cyan' : 'text-fg0'
  return (
    <div className="rounded-2xl bg-white/5 px-4 py-3">
      <div className="ds-display text-[11px] tracking-[0.28em] text-muted">{props.label}</div>
      <div className={cn('mt-2 ds-display text-lg tracking-wide', accent)}>{props.value}</div>
    </div>
  )
}

export default function Lab() {
  return (
    <div className="mx-auto h-full max-w-6xl px-4 pb-16 pt-8">
      <div className="flex flex-col gap-2">
        <div className="ds-display text-xs tracking-[0.28em] text-muted">LAB CONSOLE</div>
        <h1 className="ds-display text-4xl tracking-wide md:text-5xl">Control Room</h1>
        <p className="max-w-[78ch] text-fg1">
          This is mock data by design. The UI is built first; later we’ll wire it to real runs/evals.
        </p>
      </div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
        <section className="ds-surface p-6 md:p-8">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-fg1">
              <Layers size={16} />
              <div className="ds-display text-xs tracking-[0.28em] text-muted">RUNS</div>
            </div>
            <button className="inline-flex items-center gap-2 rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15">
              <Download size={16} />
              Export (Mock)
            </button>
          </div>

          <div className="mt-6 grid gap-4">
            {mockRuns.map((r) => {
              const chip = statusChip(r.status)
              const showMetrics = r.status !== 'failed'
              return (
                <div key={r.id} className="rounded-2xl bg-white/5 p-5 ring-1 ring-white/10">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="ds-display text-xs tracking-[0.28em] text-muted">
                        {r.dataset} · {r.variant.toUpperCase()}
                      </div>
                      <div className="mt-2 truncate text-sm text-fg1">{r.name}</div>
                      <div className="mt-3 text-[12px] text-muted">Updated {r.updatedAt}</div>
                    </div>
                    <div className={cn('inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-2 text-xs tracking-wide', chip.className)}>
                      {chip.icon}
                      <span className="ds-display tracking-[0.22em]">{chip.label}</span>
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 sm:grid-cols-4">
                    <Metric label="mAP@0.5" value={showMetrics ? r.metrics.map50.toFixed(2) : '—'} accent="cyan" />
                    <Metric label="mAP@0.5:0.95" value={showMetrics ? r.metrics.map5095.toFixed(2) : '—'} />
                    <Metric label="Params (M)" value={showMetrics ? r.metrics.paramsM.toFixed(2) : '—'} />
                    <Metric label="FPS" value={showMetrics ? String(r.metrics.fps) : '—'} accent="amber" />
                  </div>
                </div>
              )
            })}
          </div>
        </section>

        <section className="ds-surface p-6 md:p-8">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 text-fg1">
              <Gauge size={16} />
              <div className="ds-display text-xs tracking-[0.28em] text-muted">ROBUSTNESS</div>
            </div>
            <div className="ds-display text-xs tracking-[0.28em] text-muted">4×3</div>
          </div>

          <div className="mt-6 grid gap-4">
            {mockRobustness.perturbations.map((p) => {
              const vals = mockRobustness.values[p.key as keyof typeof mockRobustness.values]
              return (
                <div key={p.key} className="rounded-2xl bg-white/5 p-4 ring-1 ring-white/10">
                  <div className="ds-display text-sm tracking-wide">{p.label}</div>
                  <div className="mt-3 grid grid-cols-3 gap-2">
                    {p.levels.map((lvl, i) => {
                      const v = vals[i] ?? 0
                      const tint =
                        v > 0.74
                          ? {
                              background: 'oklch(82% 0.12 205 / 0.10)',
                              color: 'var(--cyan)',
                              boxShadow: 'inset 0 0 0 1px oklch(82% 0.12 205 / 0.22)',
                            }
                          : v > 0.66
                            ? {
                                background: 'oklch(78% 0.18 65 / 0.10)',
                                color: 'var(--amber)',
                                boxShadow: 'inset 0 0 0 1px oklch(78% 0.18 65 / 0.22)',
                              }
                            : {
                                background: 'oklch(100% 0 0 / 0.04)',
                                color: 'var(--fg-1)',
                                boxShadow: 'inset 0 0 0 1px oklch(100% 0 0 / 0.10)',
                              }
                      return (
                        <div key={lvl} className="rounded-xl px-3 py-2 text-center" style={tint}>
                          <div className="ds-display text-[11px] tracking-[0.22em] opacity-80">{lvl}</div>
                          <div className="mt-1 ds-display text-sm tracking-wide">{v.toFixed(2)}</div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>

          <div className="mt-6 rounded-2xl bg-white/5 p-4 ring-1 ring-white/10">
            <div className="ds-display text-xs tracking-[0.28em] text-muted">NOTE</div>
            <p className="mt-3 text-sm text-fg1">
              Once training scripts exist, this panel will read real CSV/JSON outputs from runs and evals.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
