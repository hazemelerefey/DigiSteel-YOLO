import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Shell from '@/components/Shell'
import About from '@/pages/About'
import Home from '@/pages/Home'
import Innovations from '@/pages/Innovations'
import Lab from '@/pages/Lab'
import NotFound from '@/pages/NotFound'
import Wiki from '@/pages/Wiki'

export default function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<Home />} />
          <Route path="/innovations" element={<Innovations />} />
          <Route path="/wiki" element={<Wiki />} />
          <Route path="/wiki/:page" element={<Wiki />} />
          <Route path="/lab" element={<Lab />} />
          <Route path="/about" element={<About />} />
        </Route>
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  )
}
