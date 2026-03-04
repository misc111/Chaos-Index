import Link from "next/link";

export default function HomePage() {
  return (
    <div className="grid two">
      <div className="card">
        <h2 className="title">Pipeline</h2>
        <p className="small">
          Daily workflow: fetch {"->"} features {"->"} train/update {"->"} predict {"->"} ingest results {"->"} score {"->"} performance/validation artifacts.
        </p>
        <p className="small">Use Make targets from repo root to run each stage.</p>
      </div>
      <div className="card">
        <h2 className="title">Quick Links</h2>
        <p>
          <Link href="/predictions">Upcoming forecasts</Link>
        </p>
        <p>
          <Link href="/leaderboard">Model leaderboard</Link>
        </p>
        <p>
          <Link href="/validation">Validation suite</Link>
        </p>
      </div>
    </div>
  );
}
