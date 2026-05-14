import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { useMemo, useRef } from 'react'
import { Color, Group, MathUtils, Vector3 } from 'three'
import { stations, useFactoryStore, type Station, type StationKey } from '@/store/factory'

function SceneRig() {
  const camera = useThree((s) => s.camera)
  const focused = useFactoryStore((s) => s.focused)
  const hovered = useFactoryStore((s) => s.hovered)

  const vPos = useMemo(() => new Vector3(), [])
  const vLook = useMemo(() => new Vector3(), [])

  useFrame((_, dt) => {
    const s = stations.find((x) => x.key === focused) ?? stations.find((x) => x.key === hovered)
    if (!s) return

    vPos.set(s.camera[0], s.camera[1], s.camera[2])
    camera.position.lerp(vPos, 1 - Math.pow(0.0009, dt))

    vLook.set(s.lookAt[0], s.lookAt[1], s.lookAt[2])
    const cur = new Vector3()
    camera.getWorldDirection(cur)
    const lookNow = new Vector3().copy(camera.position).add(cur)
    const blended = lookNow.lerp(vLook, 1 - Math.pow(0.0009, dt))
    camera.lookAt(blended)
  })

  return null
}

function Machine(props: { station: Station; onActivate: (key: StationKey) => void }) {
  const groupRef = useRef<Group>(null)
  const hovered = useFactoryStore((s) => s.hovered)
  const setHovered = useFactoryStore((s) => s.setHovered)

  const isHot = hovered === props.station.key

  useFrame((_, dt) => {
    if (!groupRef.current) return
    groupRef.current.rotation.y = MathUtils.damp(groupRef.current.rotation.y, isHot ? 0.12 : 0, 6, dt)
  })

  const emissive = isHot ? new Color('white').lerp(new Color('#2de2ff'), 0.65) : new Color('#0b1118')

  return (
    <group
      ref={groupRef}
      position={props.station.anchor}
      onPointerEnter={() => setHovered(props.station.key)}
      onPointerLeave={() => setHovered(null)}
      onClick={() => props.onActivate(props.station.key)}
    >
      <mesh castShadow receiveShadow>
        <boxGeometry args={[1.4, 0.9, 1.0]} />
        <meshStandardMaterial color={'#1c262f'} roughness={0.35} metalness={0.55} emissive={emissive} emissiveIntensity={0.4} />
      </mesh>
      <mesh position={[0, 0.62, 0]} castShadow>
        <boxGeometry args={[1.1, 0.18, 0.72]} />
        <meshStandardMaterial color={'#0e151c'} roughness={0.2} metalness={0.65} emissive={isHot ? '#ffaa33' : '#070a0f'} emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[0.64, 0.18, -0.42]} castShadow>
        <cylinderGeometry args={[0.08, 0.08, 0.9, 18]} />
        <meshStandardMaterial color={'#26323a'} roughness={0.55} metalness={0.4} />
      </mesh>
      <mesh position={[-0.6, 0.12, 0.42]} castShadow>
        <cylinderGeometry args={[0.06, 0.06, 0.8, 18]} />
        <meshStandardMaterial color={'#26323a'} roughness={0.55} metalness={0.4} />
      </mesh>
      <mesh position={[0, 0.95, 0.6]} castShadow>
        <boxGeometry args={[1.1, 0.18, 0.08]} />
        <meshStandardMaterial color={'#0a0f14'} emissive={isHot ? '#2de2ff' : '#0a0f14'} emissiveIntensity={1.2} />
      </mesh>
    </group>
  )
}

function FactoryScene(props: { onActivate: (key: StationKey) => void }) {
  return (
    <>
      <color attach="background" args={['#070a0f']} />
      <fog attach="fog" args={['#070a0f', 7, 18]} />

      <ambientLight intensity={0.22} />
      <directionalLight
        position={[6, 8, 6]}
        intensity={1.2}
        color={'#ffe6bf'}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-5, 3, -4]} intensity={1.0} color={'#2de2ff'} distance={14} />
      <pointLight position={[5, 3, -4]} intensity={1.0} color={'#ffaa33'} distance={14} />

      <group position={[0, 0, 0]}>
        <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[18, 18, 1, 1]} />
          <meshStandardMaterial color={'#0b1118'} roughness={0.95} metalness={0.15} />
        </mesh>

        <mesh position={[0, 1.8, -7]} receiveShadow>
          <boxGeometry args={[18, 3.6, 0.4]} />
          <meshStandardMaterial color={'#0a0f14'} roughness={0.85} metalness={0.2} />
        </mesh>
      </group>

      {stations.map((s) => (
        <Machine key={s.key} station={s} onActivate={props.onActivate} />
      ))}

      <EffectComposer multisampling={0}>
        <Bloom intensity={0.85} luminanceThreshold={0.58} luminanceSmoothing={0.2} />
      </EffectComposer>

      <SceneRig />
      <OrbitControls enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI * 0.48} minDistance={4.5} maxDistance={11} />
    </>
  )
}

export default function FactoryWorld(props: { onActivate: (key: StationKey) => void }) {
  return (
    <Canvas
      shadows
      camera={{ position: [0.2, 2.2, 8.4], fov: 44, near: 0.1, far: 100 }}
      dpr={[1, 1.8]}
    >
      <FactoryScene onActivate={props.onActivate} />
    </Canvas>
  )
}

