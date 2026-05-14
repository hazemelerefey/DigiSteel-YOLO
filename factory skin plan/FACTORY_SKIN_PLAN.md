# DigiSteel-YOLO — Factory Skin Plan (Session Capture)

## What You Mean (Confirmed)

You are not asking for “a separate website next to the repo”.

Your goal is: **the repository itself should feel like a steel factory product**.

That means:

- The **GitHub repo experience** (README + folder structure + docs + visuals + badges + navigation) must look/feel like a “steel factory skin”.
- The repo should have a **high-end cinematic identity** (industrial/steel/graphite/hazard accents) so the project feels unique and memorable.
- Any web UI (3D factory world) should be **integrated into the repo’s presentation**, e.g.:
  - GitHub Pages deployment (so the repo has an official interactive “factory world” front door)
  - README “Launch Factory” section linking to that experience
  - Wiki/docs inside repo and rendered in a themed viewer

Yes — I understood you: **the repo itself must be the factory skin**.

## What We Already Produced in This Workspace (And Pushed To GitHub)

### 1) Code Wiki (Markdown)

Purpose: structured internal documentation that matches the current code snapshot.

Repo paths:

- `CODE_WIKI.md` (entry)
- `wiki/` folder with pages:
  - `Home.md`
  - `Architecture.md`
  - `Modules.md`
  - `API.md`
  - `Dependencies.md`
  - `Running.md`

### 2) Wiki Viewer Web App (React/Vite)

Purpose: a lightweight local site that renders `wiki/*.md` with navigation.

Repo path:

- `web app/`

Notes:

- It’s currently a practical viewer, not yet the final “factory world”.
- It is a stepping stone: we can evolve this into the full “Factory World” by adding 3D, motion, and branded UX.

## The Factory Skin Strategy (How To Make The Repo Itself Feel Like a Steel Factory)

### A) README becomes the “Factory Control Panel”

Convert `README.md` from a standard ML repo README into an **industrial product brochure + control console**.

The README should:

- Read like a “factory manual”: sections as stations/lines (A2 Station, A3 Station, Robustness Bay, Export Dock).
- Use repo branding assets (`assets/banner.png`, `assets/logo.png`) consistently.
- Have a strong “start here” flow:
  - Quick start
  - Demo / Launch Factory (link to GitHub Pages)
  - Documentation (CODE_WIKI + wiki viewer)
  - A2/A3 explanations (with deep links to code and wiki pages)

### B) Repo Navigation as “Factory Map”

Treat folders like locations:

- `digisteel/modules/` = Innovation Bay
- `configs/` = Production Recipes
- `tests/` = QA Lab
- `wiki/` = Factory Manual
- `web app/` = Visitor Center / Interactive Tour

Add a README “map” section linking to these with short narratives.

### C) GitHub Pages as the “Factory Entrance”

Make the “Factory World” a real deployed page:

- Deploy the `web app/` via GitHub Pages (or move it to `docs/` if you want the canonical GH Pages pattern).
- Add a big README CTA: **Launch Factory World**.

This makes the repo feel like a product, not just code.

### D) Visual Identity Rules (Factory Skin Look)

Core aesthetic:

- Materials: steel / graphite / oil-slick sheen
- Lighting: cinematic low-light, crisp highlights
- Accents: molten amber + cold cyan
- Motion: “machines powering on” (stateful motion, not decoration)

Anti-patterns to avoid:

- generic template hero sections
- purple gradients
- random “AI looking” card grids
- overused fonts / generic UI kits

## Factory World (Web UI) — Scope Definition

The 3D site is not just “a website” — it is a **visual layer representing the repo**.

The first version should be:

- A 3D factory scene as navigation (stations you click)
- Each station opens a panel with:
  - explanation
  - links to wiki pages
  - links to code

Pages/stations:

- Home: Factory floor + guided tour
- A2 Station: GhostConv / weight sharing explanation
- A3 Station: Inner-WIoU explanation
- Code Wiki: embedded markdown reader
- Lab Console (mock): future training/eval dashboard preview

## Concrete Next Steps (Recommended Execution Order)

### Phase 1 — Repo Skin (GitHub-first)

1) Redesign `README.md` into “Factory Control Panel”
2) Add a `FACTORY_SKIN.md` (short style guide) for contributors
3) Ensure wiki entry points are linked clearly from README
4) Make repo navigation feel intentional (links + icons + naming)

Acceptance criteria:

- Opening GitHub repo gives a unique industrial vibe immediately
- README has strong hierarchy, clean scanability, and “factory station” structure

### Phase 2 — Deploy the Wiki Viewer (Pages)

1) Decide GitHub Pages approach:
   - Deploy from `web app/` (build output) to `gh-pages`
   - Or move to `docs/` if you want that convention
2) Add GitHub Actions workflow for Pages
3) Add README “Launch Factory World” link

Acceptance criteria:

- Public URL exists
- README links to it

### Phase 3 — Upgrade to “Factory World” (3D + Motion)

1) Add 3D layer (react-three-fiber or three.js)
2) Add station hotspots + camera transitions
3) Keep markdown/wiki as an in-world panel
4) Respect performance + prefers-reduced-motion

Acceptance criteria:

- 3D scene feels like a steel factory interior
- Navigation is clear and not gimmicky

## Decisions We Already Made In This Session

- You want a **3D interactive factory world** as the main UI layer.
- Use **mock data** for “Lab Console” for now (until real runs/evals exist).
- The goal is **extreme UI/UX**, industrial cinematic style, not generic template UI.

## How To Continue Later (Portable Checklist)

When you return:

1) Start from the repo root:
   - confirm README direction
   - confirm GitHub Pages plan
2) Decide if “web app/” should be renamed to `factory-world/` or kept as-is.
3) Implement README redesign first (fastest way to make repo feel like a product).
4) Only after README looks right: expand web app into full 3D.

