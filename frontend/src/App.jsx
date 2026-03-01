import { BrowserRouter, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import Overview from './pages/Overview';
import SeriesExplorer from './pages/SeriesExplorer';
import Findings from './pages/Findings';
import Speakers from './pages/Speakers';

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        <NavBar />
        <main className="content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/series" element={<SeriesExplorer />} />
            <Route path="/findings" element={<Findings />} />
            <Route path="/speakers" element={<Speakers />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
