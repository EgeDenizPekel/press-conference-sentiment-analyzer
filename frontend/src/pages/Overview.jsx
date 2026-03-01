import { useEffect, useState } from 'react';
import { fetchSummary } from '../api';
import StatCard from '../components/StatCard';
import ModelTable from '../components/ModelTable';

export default function Overview() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchSummary()
      .then(setData)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error-msg">Failed to load: {error}</div>;
  if (!data) return <div className="loading">Loading...</div>;

  const { corpus, sentiment_dist, pearson, model_comparison } = data;

  return (
    <div>
      <h1>Press Conference Sentiment Analyzer</h1>
      <p className="page-subtitle">
        NBA playoff press conferences 2013-2022 - sentiment analysis of coach and player speech.
      </p>

      {/* Corpus stats */}
      <div className="stat-row">
        <StatCard value={corpus.n_transcripts.toLocaleString()} label="Transcripts scraped" />
        <StatCard value={corpus.n_turns.toLocaleString()} label="Speaker turns analyzed" />
        <StatCard value={corpus.n_games} label="Games matched" />
        <StatCard value={corpus.n_series} label="Series covered" />
      </div>

      {/* Model comparison */}
      <div className="card">
        <h2>Model Comparison</h2>
        <p style={{ color: '#94a3b8', fontSize: '13px', marginBottom: '16px' }}>
          Fine-tuned RoBERTa on 2,000 GPT-4o-mini labeled turns. Evaluated on 50 hand-labeled seed turns.
        </p>
        <ModelTable models={model_comparison} />
      </div>

      {/* Sentiment distribution */}
      <div className="card">
        <h2>Corpus Sentiment Distribution</h2>
        <div style={{ display: 'flex', gap: '32px', marginBottom: '16px' }}>
          <div>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#22c55e' }}>
              {(sentiment_dist.positive * 100).toFixed(1)}%
            </div>
            <div style={{ color: '#94a3b8', fontSize: '12px' }}>Positive</div>
          </div>
          <div>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#94a3b8' }}>
              {(sentiment_dist.neutral * 100).toFixed(1)}%
            </div>
            <div style={{ color: '#94a3b8', fontSize: '12px' }}>Neutral</div>
          </div>
          <div>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#ef4444' }}>
              {(sentiment_dist.negative * 100).toFixed(1)}%
            </div>
            <div style={{ color: '#94a3b8', fontSize: '12px' }}>Negative</div>
          </div>
          <div>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#2563eb' }}>
              {sentiment_dist.mean_numeric.toFixed(3)}
            </div>
            <div style={{ color: '#94a3b8', fontSize: '12px' }}>Mean numeric score</div>
          </div>
        </div>

        {/* Stacked bar */}
        <div style={{ display: 'flex', height: '12px', borderRadius: '6px', overflow: 'hidden', gap: '2px' }}>
          <div style={{ flex: sentiment_dist.positive, background: '#22c55e' }} />
          <div style={{ flex: sentiment_dist.neutral, background: '#475569' }} />
          <div style={{ flex: sentiment_dist.negative, background: '#ef4444' }} />
        </div>
      </div>

      {/* Key findings */}
      <div className="card">
        <h2>Key Findings</h2>
        <div className="findings-grid">
          <div className="finding-item">
            <strong>No significant sentiment-outcome correlation</strong>
            Post-game press conference sentiment does not predict game outcomes. The Pearson correlation
            between mean sentiment and point differential was r={pearson[0].r} (p={pearson[0].p}),
            well above the p&lt;0.05 significance threshold. Athletes and coaches maintain consistent
            public framing regardless of win/loss.
          </div>
          <div className="finding-item">
            <strong>Sentiment rises over a series</strong>
            Across all 30 series, speaker sentiment increases from game 1 to game 6-7.
            Late-series games - which are often clinching or elimination contexts - produce
            more emotionally charged, positive language. This may reflect elevated stakes
            driving more expressive speech rather than actual mood differences.
          </div>
          <div className="finding-item">
            <strong>Sports language requires domain-specific models</strong>
            General-purpose sentiment models (DistilBERT, FinBERT) performed poorly on
            NBA press conference transcripts. Fine-tuning on 2,000 domain-labeled turns
            improved accuracy from 54% to 92% on the seed set - a +38 percentage point gain.
          </div>
          <div className="finding-item">
            <strong>Press conferences skew positive</strong>
            {(sentiment_dist.positive * 100).toFixed(0)}% of speaker turns are classified
            as positive. Athletes and coaches use diplomatically positive framing by default -
            praising opponents, emphasizing team effort, and projecting confidence. Negative
            sentiment ({(sentiment_dist.negative * 100).toFixed(0)}%) appears mainly after
            decisive losses.
          </div>
        </div>
      </div>
    </div>
  );
}
