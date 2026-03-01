import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { fetchSpeakers } from '../api';

function SpeakerTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div style={{ color: '#f1f5f9', fontWeight: 600 }}>{d.speaker}</div>
      <div>Mean sentiment: {d.mean_sentiment.toFixed(3)}</div>
      <div style={{ color: '#94a3b8' }}>
        {(d.pct_positive * 100).toFixed(0)}% positive
      </div>
      <div style={{ color: '#64748b' }}>{d.n_turns} turns</div>
    </div>
  );
}

export default function Speakers() {
  const [minTurns, setMinTurns] = useState(50);
  const [sliderVal, setSliderVal] = useState(50);
  const [speakers, setSpeakers] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setSpeakers(null);
    fetchSpeakers(minTurns)
      .then(setSpeakers)
      .catch((e) => setError(e.message));
  }, [minTurns]);

  // Debounce slider -> minTurns
  useEffect(() => {
    const t = setTimeout(() => setMinTurns(sliderVal), 300);
    return () => clearTimeout(t);
  }, [sliderVal]);

  if (error) return <div className="error-msg">Failed to load speakers: {error}</div>;

  // Recharts horizontal bar needs data sorted ascending (bottom to top)
  const chartData = speakers ? [...speakers].reverse() : [];

  return (
    <div>
      <h1>Speaker Sentiment</h1>
      <p className="page-subtitle">
        Top 20 speakers by mean sentiment score across matched press conferences. Filtered by minimum turn count.
      </p>

      <div className="slider-row">
        <label htmlFor="min-turns">Minimum turns:</label>
        <input
          id="min-turns"
          type="range"
          min={10}
          max={200}
          value={sliderVal}
          onChange={(e) => setSliderVal(Number(e.target.value))}
        />
        <span className="slider-value">{sliderVal}</span>
      </div>

      {!speakers && <div className="loading">Loading...</div>}

      {speakers && speakers.length === 0 && (
        <div className="empty-state">No speakers with {minTurns}+ turns. Try lowering the slider.</div>
      )}

      {speakers && speakers.length > 0 && (
        <div className="card">
          <h3>Mean Sentiment Score (top {speakers.length} speakers)</h3>
          <ResponsiveContainer width="100%" height={Math.max(300, chartData.length * 32)}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 40, left: 120, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
              <XAxis
                type="number"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                domain={[-0.2, 1]}
                tickFormatter={(v) => v.toFixed(1)}
              />
              <YAxis
                type="category"
                dataKey="speaker"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                width={115}
              />
              <ReferenceLine x={0} stroke="#475569" strokeDasharray="4 2" />
              <Tooltip content={<SpeakerTooltip />} />
              <Bar dataKey="mean_sentiment" radius={[0, 4, 4, 0]}>
                {chartData.map((entry) => (
                  <Cell
                    key={entry.speaker}
                    fill={entry.mean_sentiment >= 0 ? '#2563eb' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
