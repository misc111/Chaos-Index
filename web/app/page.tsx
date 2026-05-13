/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import styles from "./overview.module.css";
import { frontPageData, type FrontPageTournamentRow } from "@/lib/front-page-data";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

const navItems = [
  { href: "/", label: "Front page", icon: "home", active: true },
  { href: "/games-today?league=MLB", label: "Games today", icon: "calendar" },
  { href: "/intra-family-tournament?league=MLB", label: "Intra-Family Tournament", icon: "trophy" },
  { href: "/inter-family-tournament?league=MLB", label: "Inter-Family Tournament", icon: "trophy" },
  { href: "/ensemble-summary?league=MLB", label: "Ensemble Summary Table", icon: "table" },
] as const;

// Chaos Index is a pregame attention score: model-family disagreement, market movement,
// lineup/pitching uncertainty, price sensitivity, and edge dispersion rolled into one read.
const { kpis, games, upcomingStarters, intraFamilies, interFamilies, ensembleRows, modelStamp } = frontPageData;

type IconName = (typeof navItems)[number]["icon"];

function Icon({ name, className }: { name: IconName | "clock"; className?: string }) {
  if (name === "home") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
        <path d="M3.5 11.2 12 3.8l8.5 7.4" />
        <path d="M6.4 10.5v9.1h4.1v-5.8h3v5.8h4.1v-9.1" />
      </svg>
    );
  }

  if (name === "calendar") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
        <path d="M5 4.8h14v15H5z" />
        <path d="M5 9h14" />
        <path d="M8 3.2v3.1M16 3.2v3.1" />
        <path d="M8.2 12.2h2.2M13.6 12.2h2.2M8.2 15.8h2.2M13.6 15.8h2.2" />
      </svg>
    );
  }

  if (name === "trophy") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
        <path d="M8 4.5h8v4.9c0 3.1-1.6 5-4 5s-4-1.9-4-5z" />
        <path d="M8 6.4H4.9v2.2c0 2.1 1.2 3.4 3.3 3.5M16 6.4h3.1v2.2c0 2.1-1.2 3.4-3.3 3.5" />
        <path d="M12 14.4v3.4M8.6 20h6.8M10 17.8h4" />
      </svg>
    );
  }

  if (name === "table") {
    return (
      <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
        <path d="M4.5 5.3h15v13.4h-15z" />
        <path d="M4.5 9.8h15M9.4 5.3v13.4M14.6 5.3v13.4" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 6v6l4.2 2.4" />
    </svg>
  );
}

function BaseballLogo() {
  return (
    <svg viewBox="0 0 80 80" className={styles.baseballLogo} aria-hidden="true">
      <circle cx="40" cy="40" r="34" />
      <path d="M27 11c-8 7-11.9 16.7-11.9 29S19 62 27 69" />
      <path d="M53 11c8 7 11.9 16.7 11.9 29S61 62 53 69" />
      {[-19, -12, -5, 2, 9, 16].map((offset) => (
        <g key={offset}>
          <path d={`M${24 + offset / 12} ${40 + offset}l7 3`} />
          <path d={`M${56 - offset / 12} ${40 + offset}l-7 3`} />
        </g>
      ))}
    </svg>
  );
}

function TeamMark({ code }: { code: string }) {
  const fileCode = code === "CHW" ? "cws" : code.toLowerCase();

  return (
    <span className={styles.teamMark}>
      <img src={`${BASE_PATH}/team-icons/mlb/${fileCode}.png`} alt="" />
    </span>
  );
}

function Chevron() {
  return (
    <svg viewBox="0 0 16 24" className={styles.chevron} aria-hidden="true">
      <path d="m3 3 9 9-9 9" />
    </svg>
  );
}

function Sparkline() {
  return (
    <svg viewBox="0 0 400 82" className={styles.sparkline} aria-hidden="true">
      <defs>
        <linearGradient id="spark-orange" x1="0" x2="1">
          <stop offset="0" stopColor="#f58b00" />
          <stop offset="1" stopColor="#ffb020" />
        </linearGradient>
        <filter id="spark-glow" x="-10%" y="-80%" width="120%" height="260%">
          <feGaussianBlur stdDeviation="2.3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <path className={styles.sparkGrid} d="M0 41H400M0 3H400M0 79H400M80 0V82M160 0V82M240 0V82M320 0V82" />
      <path
        filter="url(#spark-glow)"
        d="M2 58 18 54 34 50 50 47 66 58 82 68 98 63 114 55 130 49 146 54 162 45 178 25 194 23 210 38 226 58 242 50 258 33 274 28 290 39 306 61 322 52 338 45 354 42 370 30 386 17 398 15"
      />
    </svg>
  );
}

function TournamentCard({
  title,
  rows,
  tone,
  href,
}: {
  title: string;
  rows: readonly FrontPageTournamentRow[];
  tone: "orange" | "teal";
  href: string;
}) {
  return (
    <section className={styles.tournamentCard}>
      <div className={styles.cardTitleRow}>
        <h3 className={tone === "teal" ? styles.tealHeading : undefined}>{title}</h3>
        <span>Top 3 Families</span>
      </div>
      <div className={styles.familyHeader}>
        <span />
        <span />
        <span>Score</span>
        <span>W-L</span>
        <span>Win %</span>
      </div>
      {rows.length === 0 ? (
        <div className={styles.familyRow}>
          <span className={styles.rank}>-</span>
          <span>No rows</span>
          <strong>N/A</strong>
          <span>research-only</span>
          <span>N/A</span>
        </div>
      ) : rows.map(([rank, family, score, record, win]) => (
        <div className={styles.familyRow} key={`${title}-${rank}`}>
          <span className={styles.rank}>{rank}</span>
          <span>{family}</span>
          <strong>{score}</strong>
          <span>{record}</span>
          <span>{win}</span>
        </div>
      ))}
      <Link href={href} className={tone === "teal" ? styles.panelLinkTeal : styles.panelLink}>
        View full {tone === "teal" ? "inter-family" : "intra-family"} <span aria-hidden="true">→</span>
      </Link>
    </section>
  );
}

export default function HomePage() {
  return (
    <div className={styles.page}>
      <header className={styles.topChrome}>
        <Link href="/" className={styles.brand} aria-label="Chaos Index front page">
          <BaseballLogo />
          <span className={styles.wordmark}>
            <span>
              Chaos <strong>Index</strong>
            </span>
            <small>MLB Modeling</small>
          </span>
        </Link>

        <nav className={styles.nav} aria-label="Primary navigation">
          {navItems.map((item) => (
            <Link
              href={item.href}
              className={`${styles.navItem} ${item.href === "/" ? styles.activeNavItem : ""}`}
              aria-current={item.href === "/" ? "page" : undefined}
              key={item.href}
            >
              <Icon name={item.icon} className={styles.navIcon} />
              <span>{item.label}</span>
            </Link>
          ))}
        </nav>

        <div className={styles.modelStamp}>
          <span>Model as of</span>
          <strong>{modelStamp}</strong>
        </div>
      </header>

      <section className={styles.kpiStrip} aria-label="Slate summary">
        {kpis.map((kpi) => (
          <article className={styles.kpi} key={kpi.label}>
            <span>{kpi.label}</span>
            <strong>{kpi.value}</strong>
            <small className={kpi.noteTone === "orange" ? styles.orangeText : styles.tealText}>{kpi.note}</small>
          </article>
        ))}
        <article className={styles.edgeChart}>
          <span>Edge Over Time (7d)</span>
          <div className={styles.chartWrap}>
            <Sparkline />
            <div className={styles.axisLabels} aria-hidden="true">
              <span>2%</span>
              <span>0%</span>
              <span>-2%</span>
            </div>
          </div>
        </article>
      </section>

      <main className={styles.dashboardGrid}>
        <section className={styles.gamesPanel} aria-labelledby="games-title">
          <div className={styles.panelHeader}>
            <Icon name="calendar" className={styles.sectionIcon} />
            <h2 id="games-title">Games Today</h2>
          </div>
          <div className={styles.gamesTable} role="table" aria-label="Games today">
            <div className={`${styles.gameRow} ${styles.tableHead}`} role="row">
              <span>Time ET</span>
              <span>Matchup</span>
              <span>Spread</span>
              <span>Total</span>
              <span>Chaos Index</span>
              <span>Edge</span>
              <span>Best Bet</span>
              <span />
            </div>
            {games.map((game) => (
              <div className={styles.gameRow} role="row" key={`${game.away}-${game.home}`}>
                <span className={styles.gameTime}>{game.time}</span>
                <span className={styles.matchup}>
                  <TeamMark code={game.away} />
                  <strong>{game.away}</strong>
                  <span>@</span>
                  <TeamMark code={game.home} />
                  <strong>{game.home}</strong>
                </span>
                <span className={styles.marketCell}>
                  <strong>{game.spread[0]}</strong>
                  <small>{game.spread[1]}</small>
                </span>
                <span className={styles.marketCell}>
                  <strong>{game.total[0]}</strong>
                  <small>{game.total[1]}</small>
                </span>
                <span className={styles.chaosCell} title="Pregame attention score from model disagreement, market movement, uncertainty, price sensitivity, and edge dispersion.">
                  <strong>{game.chaos}</strong>
                  <span className={styles.chaosTrack}>
                    <span style={{ width: `${game.chaos}%` }} />
                  </span>
                </span>
                <strong className={styles.edgeValue}>{game.edge}</strong>
                <strong className={`${styles.bestBet} ${styles[`${game.betTone}Bet`]}`}>{game.bestBet}</strong>
                <Chevron />
              </div>
            ))}
          </div>
          <Link href="/games-today?league=MLB" className={styles.footerLink}>
            View all games today <span aria-hidden="true">→</span>
          </Link>
        </section>

        <section className={styles.upcomingPanel} aria-labelledby="upcoming-title">
          <div className={styles.upcomingHeader}>
            <Icon name="clock" className={styles.sectionIcon} />
            <h2 id="upcoming-title">Upcoming Starters</h2>
            <span>Next 3 Games</span>
          </div>
          <div className={styles.starterGrid}>
            {upcomingStarters.map((starter) => (
              <article className={styles.starterCard} key={`${starter.away}-${starter.home}`}>
                <time>{starter.time}</time>
                <TeamMark code={starter.away} />
                <span>vs</span>
                <TeamMark code={starter.home} />
                <Link href="/games-today?league=MLB">More info <span aria-hidden="true">→</span></Link>
              </article>
            ))}
          </div>
        </section>

        <aside className={styles.sideColumn}>
          <section className={styles.panelBlock} aria-labelledby="tournament-title">
            <div className={styles.panelHeader}>
              <Icon name="trophy" className={styles.sectionIconOrange} />
              <h2 id="tournament-title">Tournament Summary</h2>
            </div>
            <TournamentCard
              title="Intra-Family Tournament"
              rows={intraFamilies}
              tone="orange"
              href="/intra-family-tournament?league=MLB"
            />
            <TournamentCard
              title="Inter-Family Tournament"
              rows={interFamilies}
              tone="teal"
              href="/inter-family-tournament?league=MLB"
            />
          </section>

          <section className={styles.ensemblePanel} aria-labelledby="ensemble-title">
            <div className={styles.panelHeader}>
              <Icon name="table" className={styles.sectionIcon} />
              <h2 id="ensemble-title">Ensemble Edge Snapshot</h2>
            </div>
            <div className={styles.ensembleTable} role="table" aria-label="Ensemble edge snapshot">
              <div className={styles.ensembleHead} role="row">
                <span>Model</span>
                <span>Sample</span>
                <span>Log Loss</span>
                <span>Accuracy</span>
              </div>
              {ensembleRows.map(([model, weight, roi, edge]) => (
                <div className={styles.ensembleRow} role="row" key={model}>
                  <span>{model}</span>
                  <span>{weight}</span>
                  <strong>{roi}</strong>
                  <strong>{edge}</strong>
                </div>
              ))}
            </div>
            <Link href="/ensemble-summary?league=MLB" className={styles.footerLink}>
              View full ensemble table <span aria-hidden="true">→</span>
            </Link>
          </section>
        </aside>
      </main>
    </div>
  );
}
