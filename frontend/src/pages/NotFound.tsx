import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="mx-auto flex h-full max-w-5xl items-center px-4">
      <div className="ds-surface ds-glow-amber w-full p-8 md:p-10">
        <div className="ds-display text-xs tracking-[0.28em] text-muted">NAVIGATION ERROR</div>
        <h1 className="ds-display mt-3 text-3xl tracking-wide md:text-4xl">Section Not Found</h1>
        <p className="mt-3 max-w-[68ch] text-fg1">
          This corridor doesn’t exist in the factory layout.
        </p>
        <div className="mt-6 flex items-center gap-3">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-full bg-white/10 px-5 py-2 text-sm text-fg0 transition hover:bg-white/15"
          >
            Return to Floor
          </Link>
          <Link
            to="/wiki"
            className="inline-flex items-center justify-center rounded-full px-5 py-2 text-sm text-fg1 transition hover:bg-white/5 hover:text-fg0"
          >
            Open Wiki
          </Link>
        </div>
      </div>
    </div>
  )
}
