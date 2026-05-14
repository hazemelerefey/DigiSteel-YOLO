import { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Link, useParams } from 'react-router-dom'
import { extractToc, getWikiPage, getWikiPages, slugifyHeading } from '@/lib/wiki'
import { cn } from '@/lib/utils'

function WikiNav(props: { activeSlug: string }) {
  const pages = useMemo(() => getWikiPages(), [])
  return (
    <aside className="hidden w-64 shrink-0 md:block">
      <div className="ds-surface p-4">
        <div className="ds-display text-xs tracking-[0.28em] text-muted">WIKI</div>
        <div className="mt-4 flex flex-col gap-1">
          {pages.map((p) => {
            const active = p.slug.toLowerCase() === props.activeSlug.toLowerCase()
            return (
              <Link
                key={p.slug}
                to={p.slug === 'Home' ? '/wiki' : `/wiki/${p.slug}`}
                className={cn(
                  'rounded-xl px-3 py-2 text-sm transition',
                  active ? 'bg-white/10 text-fg0' : 'text-fg1 hover:bg-white/5 hover:text-fg0',
                )}
              >
                {p.title}
              </Link>
            )
          })}
        </div>
      </div>
    </aside>
  )
}

function Toc(props: { markdown: string }) {
  const items = useMemo(() => extractToc(props.markdown), [props.markdown])
  return (
    <aside className="hidden w-56 shrink-0 xl:block">
      <div className="ds-surface p-4">
        <div className="ds-display text-xs tracking-[0.28em] text-muted">ON THIS PAGE</div>
        <div className="mt-4 flex flex-col gap-2">
          {items.length === 0 ? (
            <div className="text-sm text-fg1">No headings</div>
          ) : (
            items.map((it) => (
              <a
                key={it.id}
                href={`#${it.id}`}
                className={cn(
                  'text-sm text-fg1 transition hover:text-fg0',
                  it.depth === 3 ? 'pl-3 opacity-90' : 'pl-0',
                )}
              >
                {it.text}
              </a>
            ))
          )}
        </div>
      </div>
    </aside>
  )
}

export default function Wiki() {
  const params = useParams()
  const page = useMemo(() => getWikiPage(params.page), [params.page])

  if (!page) return null

  const tocSource = page.content
  const repoPath = `file:///workspace/wiki/${page.slug}.md`

  return (
    <div className="mx-auto flex h-full max-w-6xl gap-6 px-4 pb-16 pt-8">
      <WikiNav activeSlug={page.slug} />
      <div className="min-w-0 flex-1">
        <div className="ds-surface p-6 md:p-8">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="ds-display text-xs tracking-[0.28em] text-muted">DOC</div>
              <h1 className="ds-display mt-2 text-3xl tracking-wide md:text-4xl">{page.title}</h1>
            </div>
            <a
              href={repoPath}
              className="rounded-full bg-white/10 px-4 py-2 text-sm text-fg0 transition hover:bg-white/15"
            >
              Open Source File
            </a>
          </div>

          <article className="prose prose-invert mt-8 max-w-none prose-headings:ds-display prose-headings:tracking-wide prose-a:text-cyan prose-a:no-underline hover:prose-a:underline prose-code:text-amber prose-code:before:content-none prose-code:after:content-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h2: ({ children }) => {
                  const text = Array.isArray(children) ? children.join('') : String(children)
                  const id = slugifyHeading(text)
                  return (
                    <h2 id={id} className="scroll-mt-24">
                      {children}
                    </h2>
                  )
                },
                h3: ({ children }) => {
                  const text = Array.isArray(children) ? children.join('') : String(children)
                  const id = slugifyHeading(text)
                  return (
                    <h3 id={id} className="scroll-mt-24">
                      {children}
                    </h3>
                  )
                },
              }}
            >
              {page.content}
            </ReactMarkdown>
          </article>
        </div>
      </div>
      <Toc markdown={tocSource} />
    </div>
  )
}
