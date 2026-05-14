import { create } from 'zustand'

export type StationKey = 'a2' | 'a3' | 'wiki' | 'lab'

export type Station = {
  key: StationKey
  label: string
  route: string
  anchor: [number, number, number]
  camera: [number, number, number]
  lookAt: [number, number, number]
}

export const stations: Station[] = [
  {
    key: 'a2',
    label: 'A2 GhostConv',
    route: '/innovations',
    anchor: [-2.2, 0.6, -1.4],
    camera: [-3.9, 2.1, 2.6],
    lookAt: [-2.2, 0.9, -1.3],
  },
  {
    key: 'a3',
    label: 'A3 Inner-WIoU',
    route: '/innovations',
    anchor: [2.3, 0.6, -1.2],
    camera: [3.9, 2.2, 2.4],
    lookAt: [2.3, 0.9, -1.2],
  },
  {
    key: 'wiki',
    label: 'Code Wiki',
    route: '/wiki',
    anchor: [-1.6, 0.6, 2.2],
    camera: [-2.7, 2.0, 5.4],
    lookAt: [-1.6, 0.9, 2.2],
  },
  {
    key: 'lab',
    label: 'Lab Console',
    route: '/lab',
    anchor: [1.9, 0.6, 2.1],
    camera: [2.9, 2.0, 5.2],
    lookAt: [1.9, 0.9, 2.1],
  },
]

type FactoryState = {
  hovered: StationKey | null
  focused: StationKey | null
  tour: boolean
  tourIndex: number
  setHovered: (key: StationKey | null) => void
  focus: (key: StationKey | null) => void
  startTour: () => void
  stopTour: () => void
  nextStop: () => void
}

export const useFactoryStore = create<FactoryState>((set, get) => ({
  hovered: null,
  focused: null,
  tour: false,
  tourIndex: 0,
  setHovered: (key) => set({ hovered: key }),
  focus: (key) => set({ focused: key }),
  startTour: () => set({ tour: true, tourIndex: 0, focused: stations[0]?.key ?? null }),
  stopTour: () => set({ tour: false, focused: null }),
  nextStop: () => {
    const idx = get().tourIndex
    const next = (idx + 1) % stations.length
    set({ tourIndex: next, focused: stations[next]?.key ?? null })
  },
}))

