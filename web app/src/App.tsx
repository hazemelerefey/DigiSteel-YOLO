import { marked } from 'marked'
import { useEffect, useMemo, useState } from 'react'

type WikiPage = {
  id: string
  title: string
  content: string
}

const slugify = (value: string) =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .replace(/\s+/g, '-')

const wikiPagesRaw = import.meta.glob('../../wiki/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>

const buildPages = (): WikiPage[] => {
  const pages = Object.entries(wikiPagesRaw).map(([path, content]) => {
    const file = path.split('/').pop() ?? path
    const id = file.replace(/\.md$/i, '')
    return { id, title: id, content }
  })
  pages.sort((a, b) => {
    if (a.id === 'Home') return -1
    if (b.id === 'Home') return 1
    return a.id.localeCompare(b.id)
  })
  return pages
}

const buildToc = (markdown: string) => {
  const lines = markdown.split('\n')
  const out: Array<{ level: number; label: string; id: string }> = []
  for (const line of lines) {
    const match = /^(#{2,4})\s+(.+)$/.exec(line.trim())
    if (!match) continue
    const level = match[1].length
    const label = match[2].replace(/\s+#+\s*$/, '').trim()
    out.push({ level, label, id: slugify(label) })
  }
  return out
}

function App() {
  const pages = useMemo(() => buildPages(), [])
  const [activeId, setActiveId] = useState(() => {
    const hash = window.location.hash.replace(/^#/, '')
    const [page] = hash.split(':')
    return page || 'Home'
  })
  const [pendingAnchor, setPendingAnchor] = useState<string | null>(() => {
    const hash = window.location.hash.replace(/^#/, '')
    const [, anchor] = hash.split(':')
    return anchor || null
  })
  const active = pages.find((p) => p.id === activeId) ?? pages[0]

  useEffect(() => {
    const onHash = () => {
      const hash = window.location.hash.replace(/^#/, '')
      const [page, anchor] = hash.split(':')
      setActiveId(page || 'Home')
      setPendingAnchor(anchor || null)
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const renderer = useMemo(() => {
    const r = new marked.Renderer()
    r.heading = ({ text, depth }) => {
      const id = slugify(text)
      return `<h${depth} id="${id}">${text}</h${depth}>`
    }
    r.link = ({ href, title, text }) => {
      const t = title ? ` title="${title}"` : ''
      const safeHref = href ?? ''
      const isExternal = /^https?:\/\//.test(safeHref)
      const target = isExternal ? ' target="_blank" rel="noreferrer"' : ''
      return `<a href="${safeHref}"${t}${target}>${text}</a>`
    }
    return r
  }, [])

  const html = useMemo(() => marked.parse(active.content, { renderer }), [active.content, renderer])
  const toc = useMemo(() => buildToc(active.content), [active.content])

  useEffect(() => {
    if (!pendingAnchor) return
    const t = window.setTimeout(() => {
      const el = document.getElementById(pendingAnchor)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      setPendingAnchor(null)
    }, 0)
    return () => window.clearTimeout(t)
  }, [pendingAnchor, active.id])

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="mark" aria-hidden="true" />
          <div className="brandText">
            <div className="brandTitle">DigiSteel-YOLO</div>
            <div className="brandSub">Code Wiki</div>
          </div>
        </div>
        <a className="repoLink" href="https://github.com/hazemelerefey/DigiSteel-YOLO" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </header>

      <div className="frame">
        <aside className="nav">
          <div className="navTitle">Pages</div>
          <nav className="navList">
            {pages.map((p) => (
              <a
                key={p.id}
                className={p.id === active.id ? 'navItem active' : 'navItem'}
                href={`#${p.id}`}
              >
                {p.title}
              </a>
            ))}
          </nav>

          <div className="navFooter">
            <a
              className="navMini"
              href="https://github.com/hazemelerefey/DigiSteel-YOLO/blob/main/CODE_WIKI.md"
              target="_blank"
              rel="noreferrer"
            >
              Open in GitHub
            </a>
          </div>
        </aside>

        <main className="content">
          <article className="doc" dangerouslySetInnerHTML={{ __html: html }} />
        </main>

        <aside className="toc">
          <div className="tocTitle">On this page</div>
          <div className="tocList">
            {toc.map((h) => (
              <a key={`${h.level}-${h.id}`} className={`tocItem l${h.level}`} href={`#${active.id}:${h.id}`}>
                {h.label}
              </a>
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}

export default App
