/* eslint-disable @next/next/no-img-element */
import Link from "next/link";
import styles from "./overview.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

const iconCodes = ["sf", "lad", "nyy", "bos", "sea", "ari", "chc", "hou", "atl", "tor"];

const pageCards = [
  {
    href: "/games-today?league=MLB",
    number: "01",
    title: "Games today",
    copy: "Pregame slate, model probabilities, market context, and the bet/no-bet note.",
    action: "Open slate",
  },
  {
    href: "/intra-family-tournament?league=MLB",
    number: "02",
    title: "Intra-Family Tournament",
    copy: "Model-family evidence before the bracket promotes a representative.",
    action: "Review families",
  },
  {
    href: "/inter-family-tournament?league=MLB",
    number: "03",
    title: "Inter-Family Tournament",
    copy: "Champion comparison across families with governance labels kept visible.",
    action: "Compare winners",
  },
  {
    href: "/ensemble-summary?league=MLB",
    number: "04",
    title: "Ensemble Summary Table",
    copy: "Compact probability roster, component model weights, and ensemble readout.",
    action: "View table",
  },
];

export default function HomePage() {
  return (
    <div className={styles.page}>
      <section className={styles.hero} aria-labelledby="home-title">
        <div className={styles.heroCopy}>
          <h1 id="home-title">Chaos Index</h1>
          <p className={styles.deck}>MLB probability desk for the pregame slate.</p>
          <p>
            Pregame probabilities, market context, and model-family evidence stay in one MLB-focused notebook.
          </p>
        </div>
        <div className={styles.heroPanel} aria-hidden="true">
          <div className={styles.heroPanelTop}>
            <span>MLB</span>
            <span>Pregame only</span>
          </div>
          <div className={styles.logoConstellation}>
            {iconCodes.map((code) => (
              <img key={code} src={`${BASE_PATH}/team-icons/mlb/${code}.png`} alt="" />
            ))}
          </div>
        </div>
      </section>

      <section className={styles.pageGrid} aria-label="Main pages">
        {pageCards.map((card) => (
          <Link href={card.href} className={styles.pageCard} key={card.href}>
            <span className={styles.pageCardNumber}>{card.number}</span>
            <span className={styles.pageCardBody}>
              <span className={styles.pageCardTitle}>{card.title}</span>
              <span className={styles.pageCardCopy}>{card.copy}</span>
            </span>
            <span className={styles.pageCardAction}>
              {card.action}
              <span aria-hidden="true">&rarr;</span>
            </span>
          </Link>
        ))}
      </section>
    </div>
  );
}
