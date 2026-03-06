import styles from "./overview.module.css";

export default function HomePage() {
  return (
    <div className={styles.page}>
      <section className={styles.card}>
        <p className={styles.eyebrow}>Overview</p>
        <h1 className={styles.headline}>Betting with a Brain</h1>
        <p className={styles.body}>
          Most betting apps show the market. This page gives you independent win probabilities you can compare against it.
        </p>
        <p className={styles.body}>
          This page was created and is maintained by{" "}
          <a className={styles.link} href="https://www.linkedin.com/in/david-iruegas/" target="_blank" rel="noreferrer">
            David Iruegas
          </a>
          , ACAS. David is an{" "}
          <a
            className={styles.link}
            href="https://en.wikipedia.org/wiki/Actuary"
            target="_blank"
            rel="noreferrer"
          >
            actuary
          </a>
          , an Associate of the{" "}
          <a className={styles.link} href="https://www.casact.org/" target="_blank" rel="noreferrer">
            Casualty Actuarial Society (CAS)
          </a>
          , with 5 years of actuarial experience.
        </p>
        <p className={styles.body}>
          These win probabilities are built using advanced statistical and machine learning techniques.
        </p>
      </section>
    </div>
  );
}
