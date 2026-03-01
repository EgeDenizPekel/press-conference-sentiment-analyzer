const BASE = 'http://localhost:8000';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const fetchSummary = () => get('/analysis/summary');
export const fetchTrajectory = () => get('/analysis/trajectory');
export const fetchSeriesPosition = () => get('/analysis/series-position');
export const fetchSeriesList = () => get('/series');
export const fetchSeries = (id) => get(`/series/${encodeURIComponent(id)}`);
export const fetchSpeakers = (minTurns = 50) => get(`/speakers?min_turns=${minTurns}`);
