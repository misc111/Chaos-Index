import ResearchDeskExperience from "@/components/research-desk/ResearchDeskExperience";
import { normalizeLeague } from "@/lib/league";

type ResearchDeskPageProps = {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function ResearchDeskPage({ searchParams }: ResearchDeskPageProps) {
  const resolved = searchParams ? await searchParams : {};
  const rawLeague = Array.isArray(resolved.league) ? resolved.league[0] : resolved.league;
  return <ResearchDeskExperience league={normalizeLeague(rawLeague ?? null)} />;
}
