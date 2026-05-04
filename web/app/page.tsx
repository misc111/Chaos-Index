/* eslint-disable @next/next/no-img-element */
import styles from "./overview.module.css";

const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH || "";

const navLinks = [
  { href: "/market-board?league=MLB", label: "Market Board" },
  { href: "/research-desk?league=MLB", label: "Research Desk" },
  { href: "/performance?league=MLB", label: "Performance" },
];

const principles = [
  {
    label: "Actuarial spine",
    title: "Credibility before conviction",
    body:
      "GLM-family forecasts, penalized structure, and governance labels keep the model evidence separate from the betting overlay.",
  },
  {
    label: "Baseball texture",
    title: "Pregame context only",
    body:
      "The ledger is built around frozen pregame probabilities, first-pitch timing, and MLB-specific feature discipline.",
  },
  {
    label: "Market pressure",
    title: "Price is a comparison",
    body:
      "Vig-free market transforms are treated as explicit reference points, not as permission to copy the board.",
  },
];

export default function HomePage() {
  const heroImage = `${BASE_PATH}/images/chaos-index-hero.png`;
  const credibilityImage = `${BASE_PATH}/images/chaos-index-credibility.png`;

  return (
    <div className={`${styles.page} beauty-home`}>
      <section className={styles.hero} aria-labelledby="home-hero-title">
        <img className={styles.heroImage} src={heroImage} alt="" aria-hidden="true" />
        <div className={styles.heroShade} aria-hidden="true" />
        <div className={styles.heroContent}>
          <p className={styles.eyebrow}>MLB actuarial betting model</p>
          <h1 className={styles.headline} id="home-hero-title">
            Chaos Index
          </h1>
          <p className={styles.heroCopy}>
            Independent baseball probabilities shaped by actuarial discipline, market-aware pricing, and pregame-only
            evidence.
          </p>
          <div className={styles.heroActions} aria-label="Primary destinations">
            <a className={styles.primaryAction} href={`${BASE_PATH}/market-board?league=MLB`}>
              Open Market Board
            </a>
            <a className={styles.secondaryAction} href={`${BASE_PATH}/research-desk?league=MLB`}>
              Review Evidence
            </a>
          </div>
        </div>
        <div className={styles.signalStrip} aria-label="Model posture">
          <span>Pregame ledger</span>
          <span>GLM family</span>
          <span>Vig-free market lens</span>
        </div>
      </section>

      <section className={styles.intro} aria-label="Model introduction">
        <p className={styles.kicker}>Built by David Iruegas, ACAS</p>
        <h2>Actuarial science, under stadium lights.</h2>
        <p>
          Chaos Index treats MLB betting like an evidence problem: quantify uncertainty, protect the timing of every
          forecast, and let the market become a disciplined benchmark rather than the answer key.
        </p>
        <a
          className={styles.textLink}
          href="https://www.linkedin.com/in/david-iruegas/"
          target="_blank"
          rel="noreferrer"
        >
          Connect with David
        </a>
      </section>

      <section className={styles.principleGrid} aria-label="Model principles">
        {principles.map((principle) => (
          <article className={styles.principleCard} key={principle.label}>
            <p>{principle.label}</p>
            <h3>{principle.title}</h3>
            <span>{principle.body}</span>
          </article>
        ))}
      </section>

      <section className={styles.featureBand} aria-labelledby="credibility-title">
        <div className={styles.featureImageWrap}>
          <img
            className={styles.featureImage}
            src={credibilityImage}
            alt="Actuarial tables, a baseball, and probability ribbons flowing into a strike-zone visualization."
          />
        </div>
        <div className={styles.featureCopy}>
          <p className={styles.eyebrow}>Theory lane</p>
          <h2 id="credibility-title">Credibility is the house style.</h2>
          <p>
            The model program is rooted in GLMs, penalization, offset logic, diagnostics, and written promotion evidence.
            The betting product can be sharp only if the statistical story stays traceable.
          </p>
          <div className={styles.metricRow} aria-label="Model commitments">
            <span>
              <strong>01</strong>
              Frozen forecasts
            </span>
            <span>
              <strong>02</strong>
              Governance lanes
            </span>
            <span>
              <strong>03</strong>
              Promotion evidence
            </span>
          </div>
        </div>
      </section>

      <section className={styles.linkBand} aria-label="Dashboard destinations">
        <div>
          <p className={styles.kicker}>From beauty to board</p>
          <h2>Step into the live model surface.</h2>
        </div>
        <div className={styles.destinationGrid}>
          {navLinks.map((link) => (
            <a className={styles.destinationLink} href={`${BASE_PATH}${link.href}`} key={link.href}>
              {link.label}
            </a>
          ))}
        </div>
      </section>
    </div>
  );
}
