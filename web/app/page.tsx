/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import styles from "./overview.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

const iconCodes = ["sf", "lad", "nyy", "bos", "sea", "ari", "chc", "hou", "atl", "tor"];

const pageCards = [
  {
    href: "/games-today?league=MLB",
    title: "Games today",
    copy: "The slate, the probabilities, and the little bet/no-bet note.",
  },
  {
    href: "/intra-family-tournament?league=MLB",
    title: "Intra-Family Tournament",
    copy: "Each model family picks its own best baseball brain.",
  },
  {
    href: "/inter-family-tournament?league=MLB",
    title: "Inter-Family Tournament",
    copy: "The family winners meet on the final evidence board.",
  },
  {
    href: "/ensemble-summary?league=MLB",
    title: "Ensemble Summary Table",
    copy: "A compact roster of probabilities and component models.",
  },
];

export default function HomePage() {
  return (
    <div className={styles.page}>
      <section className={styles.hero} aria-labelledby="home-title">
        <div className={styles.heroCopy}>
          <p className={styles.kicker}>Chaos Index</p>
          <h1 id="home-title">Baseball &amp; Insurance Premiums</h1>
          <p>
            A small MLB notebook for modeling today’s games, keeping the tournament evidence honest, and making the
            probabilities feel friendly enough to actually read.
          </p>
          <div className={styles.heroActions}>
            <Link href="/games-today?league=MLB" className={styles.primaryLink}>
              See games today
            </Link>
            <Link href="/ensemble-summary?league=MLB" className={styles.secondaryLink}>
              View ensemble table
            </Link>
          </div>
        </div>
        <div className={styles.logoConstellation} aria-hidden="true">
          {iconCodes.map((code) => (
            <img key={code} src={`${BASE_PATH}/team-icons/mlb/${code}.png`} alt="" />
          ))}
        </div>
      </section>

      <section className={styles.pageGrid} aria-label="Main pages">
        {pageCards.map((card) => (
          <Link href={card.href} className={styles.pageCard} key={card.href}>
            <span>{card.title}</span>
            <p>{card.copy}</p>
          </Link>
        ))}
      </section>
    </div>
  );
}
