import { collectCommittedStagingSnapshotErrors, PRIMARY_STAGING_LEAGUE, SHIPPED_STAGING_LEAGUES } from "./staging-contract";

async function main(): Promise<void> {
  const errors = await collectCommittedStagingSnapshotErrors();

  if (errors.length > 0) {
    console.error("Staging contract verification failed:");
    for (const error of errors) {
      console.error(`- ${error}`);
    }
    process.exit(1);
  }

  console.log(
    `Verified MLB-first staging contract. Primary lane: ${PRIMARY_STAGING_LEAGUE}. Shipped leagues: ${SHIPPED_STAGING_LEAGUES.join(", ")}.`
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
