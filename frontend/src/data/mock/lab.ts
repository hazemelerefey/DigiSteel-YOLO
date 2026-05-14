export type RunStatus = 'complete' | 'running' | 'queued' | 'failed'

export type RunSummary = {
  id: string
  name: string
  dataset: 'NEU-DET' | 'GC10-DET'
  variant: 'baseline' | 'a2' | 'a3' | 'a2+a3'
  status: RunStatus
  metrics: {
    map50: number
    map5095: number
    paramsM: number
    fps: number
  }
  updatedAt: string
}

export const mockRuns: RunSummary[] = [
  {
    id: 'run_2026_05_01_120100',
    name: 'yolov11n_a2_a3_NEU_seed42_20260501_120100',
    dataset: 'NEU-DET',
    variant: 'a2+a3',
    status: 'running',
    metrics: { map50: 0.82, map5095: 0.44, paramsM: 1.95, fps: 168 },
    updatedAt: '2026-05-14 15:40',
  },
  {
    id: 'run_2026_05_01_111900',
    name: 'yolov11n_baseline_NEU_seed42_20260501_111900',
    dataset: 'NEU-DET',
    variant: 'baseline',
    status: 'complete',
    metrics: { map50: 0.79, map5095: 0.41, paramsM: 2.64, fps: 154 },
    updatedAt: '2026-05-12 09:10',
  },
  {
    id: 'run_2026_05_09_204200',
    name: 'yolov11n_a2_a3_GC10_seed42_20260509_204200',
    dataset: 'GC10-DET',
    variant: 'a2+a3',
    status: 'queued',
    metrics: { map50: 0.76, map5095: 0.38, paramsM: 1.95, fps: 168 },
    updatedAt: '2026-05-14 15:39',
  },
  {
    id: 'run_2026_05_08_083000',
    name: 'yolov11n_baseline_GC10_seed42_20260508_083000',
    dataset: 'GC10-DET',
    variant: 'baseline',
    status: 'failed',
    metrics: { map50: 0.0, map5095: 0.0, paramsM: 2.64, fps: 0 },
    updatedAt: '2026-05-08 10:22',
  },
]

export const mockRobustness = {
  perturbations: [
    { key: 'blur', label: 'Gaussian Blur', levels: ['σ=1', 'σ=3', 'σ=5'] },
    { key: 'noise', label: 'Gaussian Noise', levels: ['σ=0.05', 'σ=0.10', 'σ=0.20'] },
    { key: 'brightness', label: 'Brightness Drift', levels: ['Δ=-50', 'Δ=+20', 'Δ=+50'] },
    { key: 'jpeg', label: 'JPEG Compression', levels: ['Q=80', 'Q=50', 'Q=30'] },
  ],
  values: {
    blur: [0.78, 0.72, 0.64],
    noise: [0.76, 0.69, 0.58],
    brightness: [0.75, 0.77, 0.71],
    jpeg: [0.77, 0.73, 0.67],
  },
}

