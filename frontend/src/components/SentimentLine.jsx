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
} from 'recharts';

function DefaultTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <div className="tt-label">Game {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </div>
      ))}
    </div>
  );
}

/**
 * Reusable Recharts line chart wrapper.
 *
 * Props:
 *   data        - array of objects
 *   xKey        - key for x-axis
 *   lines       - [{ key, name, color }]
 *   xLabel      - string label for x-axis
 *   yLabel      - string label for y-axis
 *   yDomain     - [min, max] or 'auto'
 *   tooltipContent - optional custom tooltip component
 */
export default function SentimentLine({
  data,
  xKey,
  lines,
  xLabel,
  yDomain = ['auto', 'auto'],
  tooltipContent,
}) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey={xKey}
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 } : undefined}
        />
        <YAxis
          stroke="#64748b"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          domain={yDomain}
          tickFormatter={(v) => v.toFixed(2)}
        />
        <ReferenceLine y={0} stroke="#475569" strokeDasharray="4 2" />
        <Tooltip content={tooltipContent ?? <DefaultTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: '12px', color: '#94a3b8', paddingTop: '8px' }}
        />
        {lines.map((l) => (
          <Line
            key={l.key}
            type="monotone"
            dataKey={l.key}
            name={l.name}
            stroke={l.color}
            strokeWidth={2}
            dot={{ r: 4, fill: l.color, strokeWidth: 0 }}
            activeDot={{ r: 6 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
