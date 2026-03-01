import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Dot,
} from 'recharts';
import { fetchSeriesList, fetchSeries } from '../api';

function GameDot(props) {
  const { cx, cy, payload } = props;
  const color = payload.home_win === 1 ? '#22c55e' : '#ef4444';
  return <circle cx={cx} cy={cy} r={6} fill={color} stroke="#0f172a" strokeWidth={2} />;
}

function SeriesLineTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="custom-tooltip">
      <div className="tt-label">Game {label} - {d.date}</div>
      <div>{d.home_team} vs {d.away_team}</div>
      <div style={{ color: '#94a3b8' }}>
        Point diff: {d.point_diff > 0 ? '+' : ''}{d.point_diff}
      </div>
      <div>Sentiment: {d.mean_sentiment.toFixed(3)}</div>
      <div style={{ color: '#94a3b8' }}>
        {d.pct_positive !== undefined
          ? `${(d.pct_positive * 100).toFixed(0)}% pos / ${(d.pct_negative * 100).toFixed(0)}% neg`
          : ''}
      </div>
      <div style={{ color: '#64748b' }}>{d.n_turns} turns</div>
    </div>
  );
}

export default function SeriesExplorer() {
  const [seriesList, setSeriesList] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState(null);
  const [listError, setListError] = useState(null);
  const [detailError, setDetailError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSeriesList()
      .then((list) => {
        setSeriesList(list);
        if (list.length > 0) setSelectedId(list[0].series_id);
      })
      .catch((e) => setListError(e.message));
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    setLoading(true);
    setDetailError(null);
    fetchSeries(selectedId)
      .then((d) => { setDetail(d); setLoading(false); })
      .catch((e) => { setDetailError(e.message); setLoading(false); });
  }, [selectedId]);

  if (listError) return <div className="error-msg">Failed to load series list: {listError}</div>;

  return (
    <div>
      <h1>Series Explorer</h1>
      <p className="page-subtitle">
        Game-by-game sentiment arc for each playoff series. Dots are green (home win) or red (home loss).
      </p>

      <div className="series-controls">
        <select
          className="series-select"
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
        >
          {seriesList.map((s) => (
            <option key={s.series_id} value={s.series_id}>
              {s.label} - {s.round} ({s.n_games} games)
            </option>
          ))}
        </select>
      </div>

      {loading && <div className="loading">Loading series...</div>}
      {detailError && <div className="error-msg">Failed to load series: {detailError}</div>}

      {!loading && detail && (
        <div className="card">
          <h2>{detail.label} - {detail.round}</h2>

          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={detail.games} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="game_num"
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                label={{ value: 'Game', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 }}
              />
              <YAxis
                stroke="#64748b"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
              <Tooltip content={<SeriesLineTooltip />} />
              <Line
                type="monotone"
                dataKey="mean_sentiment"
                name="Mean Sentiment"
                stroke="#2563eb"
                strokeWidth={2}
                dot={<GameDot />}
                activeDot={{ r: 8 }}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Win/loss chips */}
          <div className="game-results">
            {detail.games.map((g) => (
              <div key={g.game_num} className={`game-chip ${g.home_win === 1 ? 'win' : 'loss'}`}>
                <span>G{g.game_num}</span>
                <span>{g.home_win === 1 ? 'W' : 'L'}</span>
                <span className="chip-label">{g.home_team}</span>
              </div>
            ))}
          </div>

          {/* Series score */}
          {detail.games.length > 0 && (() => {
            const last = detail.games[detail.games.length - 1];
            return (
              <div style={{ marginTop: '16px', color: '#94a3b8', fontSize: '13px' }}>
                Final: {last.home_team} {last.home_series_wins} - {last.away_series_wins} {last.away_team}
              </div>
            );
          })()}
        </div>
      )}

      {!loading && !detail && !detailError && (
        <div className="empty-state">Select a series above to view its sentiment arc.</div>
      )}
    </div>
  );
}
