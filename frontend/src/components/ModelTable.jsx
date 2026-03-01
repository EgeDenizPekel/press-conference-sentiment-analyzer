export default function ModelTable({ models }) {
  return (
    <table className="model-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Seed Accuracy</th>
          <th>Macro F1</th>
        </tr>
      </thead>
      <tbody>
        {models.map((m) => (
          <tr key={m.model} className={m.ours ? 'ours' : ''}>
            <td>{m.model}</td>
            <td>{(m.accuracy * 100).toFixed(0)}%</td>
            <td>{m.macro_f1.toFixed(3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
