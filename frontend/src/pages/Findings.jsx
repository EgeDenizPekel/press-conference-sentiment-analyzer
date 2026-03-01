import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from 'recharts';
import { fetchSummary, fetchTrajectory, fetchSeriesPosition } from '../api';

function TrajectoryTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div className="custom-tooltip">
      <div className="tt-label">Game {label}</div>
      <div>Mean: {d?.mean_sentiment?.toFixed(3)}</div>
      <div style={{ color: '#94a3b8' }}>Std: {d?.std_sentiment?.toFixed(3)}</div>
      <div style={{ color: '#64748b' }}>{d?.n_turns} turns</div>
    </div>
  );
}

function PositionTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="tt-label">Game {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {p.value?.toFixed(3)}
        </div>
      ))}
    </div>
  );
}

export default function Findings() {
  const [trajectory, setTrajectory] = useState(null);
  const [position, setPosition] = useState(null);
  const [pearson, setPearson] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([fetchTrajectory(), fetchSeriesPosition(), fetchSummary()])
      .then(([traj, pos, summary]) => {
        // Build CI bands: mean +/- std/sqrt(n)
        const trajWithCI = traj.map((d) => {
          const se = d.std_sentiment / Math.sqrt(d.n_turns);
          return {
            ...d,
            upper: d.mean_sentiment + se,
            lower: d.mean_sentiment - se,
          };
        });
        setTrajectory(trajWithCI);

        // Pivot series-position data into per-game objects
        const byGame = {};
        pos.forEach((row) => {
          if (!byGame[row.game_num]) byGame[row.game_num] = { game_num: row.game_num };
          byGame[row.game_num][row.home_leads ? 'home_leads' : 'home_trails'] = row.mean_sentiment;
        });
        setPosition(Object.values(byGame).sort((a, b) => a.game_num - b.game_num));

        setPearson(summary.pearson);
      })
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error-msg">Failed to load findings: {error}</div>;
  if (!trajectory || !position || !pearson) return <div className="loading">Loading...</div>;

  return (
    <div>
      <h1>Research Findings</h1>
      <p className="page-subtitle">
        Sentiment patterns across 30 playoff series, 141 games, 10,881 matched speaker turns.
      </p>

      {/* Charts side by side */}
      <div className="charts-grid">
        <div className="card">
          <h3>Sentiment Trajectory Across a Series</h3>
          <p style={{ color: '#64748b', fontSize: '12px', marginBottom: '12px' }}>
            Mean sentiment by game position (shaded: +-1 SE)
          </p>
          <ComposedChart data={trajectory} margin={{ top: 8, right: 16, left: 0, bottom: 8 }} width={500} height={260}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="game_num" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Game', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 }} />
            <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={(v) => v.toFixed(2)} />
            <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
            <Tooltip content={<TrajectoryTooltip />} />
            <Area type="monotone" dataKey="upper" stroke="none" fill="#2563eb" fillOpacity={0.12} legendType="none" />
            <Area type="monotone" dataKey="lower" stroke="none" fill="#0f172a" fillOpacity={1} legendType="none" />
            <Line type="monotone" dataKey="mean_sentiment" name="Mean Sentiment" stroke="#2563eb" strokeWidth={2} dot={{ r: 4, fill: '#2563eb', strokeWidth: 0 }} activeDot={{ r: 6 }} />
          </ComposedChart>
          <ResponsiveContainer width="100%" height={0} />
        </div>

        <div className="card">
          <h3>Sentiment by Series Position</h3>
          <p style={{ color: '#64748b', fontSize: '12px', marginBottom: '12px' }}>
            Home team leads vs trails heading into each game
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={position} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="game_num" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} label={{ value: 'Game', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 }} />
              <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 12 }} tickFormatter={(v) => v.toFixed(2)} />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
              <Tooltip content={<PositionTooltip />} />
              <Legend wrapperStyle={{ fontSize: '12px', color: '#94a3b8', paddingTop: '8px' }} />
              <Line type="monotone" dataKey="home_leads" name="Home leads series" stroke="#22c55e" strokeWidth={2} dot={{ r: 4, fill: '#22c55e', strokeWidth: 0 }} connectNulls />
              <Line type="monotone" dataKey="home_trails" name="Home trails series" stroke="#ef4444" strokeWidth={2} dot={{ r: 4, fill: '#ef4444', strokeWidth: 0 }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Pearson table */}
      <div className="card">
        <h2>Correlation Analysis</h2>
        <p style={{ color: '#94a3b8', fontSize: '13px', marginBottom: '16px' }}>
          Pearson r between post-game press conference sentiment and game outcomes. n=141 games.
        </p>
        <table className="pearson-table">
          <thead>
            <tr>
              <th>Analysis</th>
              <th>r</th>
              <th>p-value</th>
              <th>n</th>
              <th>Significant</th>
            </tr>
          </thead>
          <tbody>
            {pearson.map((row) => (
              <tr key={row.label}>
                <td>{row.label}</td>
                <td>{row.r.toFixed(3)}</td>
                <td>{row.p.toFixed(3)}</td>
                <td>{row.n}</td>
                <td>
                  {row.p < 0.05
                    ? <span className="badge-sig">Yes</span>
                    : <span className="badge-ns">No (p&gt;0.05)</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ color: '#64748b', fontSize: '12px', marginTop: '12px' }}>
          Neither same-game nor next-game point differentials show a significant linear relationship with post-game sentiment.
          Press conference framing appears independent of game outcomes.
        </p>
      </div>
    </div>
  );
}
