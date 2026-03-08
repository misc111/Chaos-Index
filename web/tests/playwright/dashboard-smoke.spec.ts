import { expect, test, type Page } from "@playwright/test";

type LeagueCode = "NBA" | "NHL";

type RouteExpectation = {
  label: string;
  path: string;
  readyText: RegExp;
  interaction?: (page: Page) => Promise<void>;
};

const UI_ERROR_PATTERN =
  /Application error|Unhandled Runtime Error|Something went wrong|Failed to load|Unable to load dashboard data|Unable to load the market board right now/i;

const mainRoutes: ReadonlyArray<RouteExpectation> = [
  {
    label: "Overview",
    path: "/",
    readyText: /Betting with a Brain/i,
  },
  {
    label: "Games Today",
    path: "/games-today",
    readyText: /Anticipated winner threshold/i,
    interaction: async (page) => {
      const pageTitle = page.getByRole("heading", { name: /Games Today|Games on/i }).first();
      const previousButton = page.getByRole("button", { name: "Previous" }).first();
      const nextButton = page.getByRole("button", { name: "Next" }).first();

      await expect(pageTitle).toHaveText(/Games Today/i);
      await previousButton.click();
      await settle(page);
      await expect(pageTitle).not.toHaveText(/Games Today/i);
      await nextButton.click();
      await settle(page);
      await expect(pageTitle).toHaveText(/Games Today/i);
    },
  },
  {
    label: "Bet Sizing",
    path: "/bet-sizing",
    readyText: /How the App Picks a Bet Amount/i,
    interaction: async (page) => {
      const firstSavedObjective = page.getByRole("button", { name: /Risk-Adjusted Optimal|Aggressive|Capital Preservation/i }).first();
      await firstSavedObjective.click();
      await settle(page);
      await expect(page.locator("main")).toContainText(/Why This Game Becomes|No games are available to preview yet\./i);
    },
  },
  {
    label: "Market Board",
    path: "/market-board",
    readyText: /Best current moneyline by side/i,
  },
  {
    label: "Bet History",
    path: "/bet-history",
    readyText: /Weekly Bet Calendar/i,
    interaction: async (page) => {
      const previousWeekButton = page.getByRole("button", { name: "Previous Week" });
      const nextWeekButton = page.getByRole("button", { name: "Next Week" });

      if (await previousWeekButton.isDisabled()) {
        return;
      }

      const weekLabel = page.getByText(/\b[A-Z][a-z]{2} \d{1,2} - [A-Z][a-z]{2} \d{1,2}, \d{4}\b/).first();
      const before = (await weekLabel.textContent())?.trim() || "";
      await previousWeekButton.click();
      await settle(page);
      await expect(weekLabel).not.toHaveText(before);
      await nextWeekButton.click();
      await settle(page);
      await expect(weekLabel).toHaveText(before);
    },
  },
  {
    label: "Actual vs Expected",
    path: "/actual-vs-expected",
    readyText: /Toss-up band: 45%-55% modeled home-win probability\./i,
    interaction: async (page) => {
      const monthLabel = page.locator(".calendar-month-label");
      const previousButton = page.getByRole("button", { name: "Previous" }).first();
      const nextButton = page.getByRole("button", { name: "Next" }).first();
      const before = (await monthLabel.textContent())?.trim() || "";

      await previousButton.click();
      await settle(page);
      await expect(monthLabel).not.toHaveText(before);
      await nextButton.click();
      await settle(page);
      await expect(monthLabel).toHaveText(before);
    },
  },
  {
    label: "Model Summary",
    path: "/predictions",
    readyText: /Next home games/i,
    interaction: async (page) => {
      const teamFilter = page.locator("#predictions-team-filter");
      const options = teamFilter.locator("option");
      const optionCount = await options.count();

      if (optionCount <= 1) {
        return;
      }

      const selectedOption = options.nth(1);
      const teamValue = await selectedOption.getAttribute("value");
      const teamLabel = ((await selectedOption.textContent()) || "").trim();

      if (!teamValue || !teamLabel) {
        return;
      }

      await teamFilter.selectOption(teamValue);
      await settle(page);
      await expect(page.locator("main")).toContainText(teamLabel);
      await page.getByRole("button", { name: "Clear filter" }).click();
      await settle(page);
      await expect(teamFilter).toHaveValue("");
    },
  },
];

const directRoutes: ReadonlyArray<RouteExpectation> = [
  {
    label: "Leaderboard",
    path: "/leaderboard",
    readyText: /Leaderboard \(rolling \+ cumulative\)/i,
  },
  {
    label: "Performance",
    path: "/performance",
    readyText: /log loss over time/i,
  },
  {
    label: "Calibration",
    path: "/calibration",
    readyText: /Calibration Metrics \(alpha\/beta\/ECE\/MCE\)/i,
  },
  {
    label: "Diagnostics",
    path: "/diagnostics",
    readyText: /GLM\/ML Diagnostics Snapshot/i,
  },
  {
    label: "Slices",
    path: "/slices",
    readyText: /Slice Analysis \+ Drift/i,
  },
  {
    label: "Validation",
    path: "/validation",
    readyText: /split_summary/i,
  },
];

function buildLeaguePath(path: string, league: LeagueCode): string {
  return `${path}${path.includes("?") ? "&" : "?"}league=${league}`;
}

function summarizeUrl(rawUrl: string): string {
  try {
    const url = new URL(rawUrl);
    return `${url.pathname}${url.search}`;
  } catch {
    return rawUrl;
  }
}

function trackPageIssues(page: Page) {
  const issues: string[] = [];

  const pushIssue = (issue: string) => {
    issues.push(issue.replace(/\s+/g, " ").trim());
  };

  page.on("console", (message) => {
    if (message.type() === "error") {
      pushIssue(`console error: ${message.text()}`);
    }
  });

  page.on("pageerror", (error) => {
    pushIssue(`page error: ${error.message}`);
  });

  page.on("requestfailed", (request) => {
    const failure = request.failure();
    const errorText = failure?.errorText || "";
    const shortUrl = summarizeUrl(request.url());
    const ignoreAbortedRsc = errorText === "net::ERR_ABORTED" && shortUrl.includes("_rsc=");

    if (shortUrl.includes("favicon.ico") || ignoreAbortedRsc) {
      return;
    }

    pushIssue(`request failed: ${request.method()} ${shortUrl} ${errorText}`);
  });

  page.on("response", (response) => {
    if (response.status() < 400) {
      return;
    }

    const shortUrl = summarizeUrl(response.url());
    if (shortUrl.includes("favicon.ico")) {
      return;
    }

    pushIssue(`http ${response.status()}: ${response.request().method()} ${shortUrl}`);
  });

  return {
    flush(): string[] {
      const snapshot = [...new Set(issues)];
      issues.length = 0;
      return snapshot;
    },
  };
}

async function settle(page: Page) {
  await page.waitForLoadState("domcontentloaded");
  try {
    await page.waitForLoadState("networkidle", { timeout: 10_000 });
  } catch {
    // Some routes stream data; the assertions below provide the real readiness checks.
  }
  await page.waitForTimeout(250);
}

async function expectDashboardShell(page: Page, league: LeagueCode) {
  await expect(page.locator("h1.app-title")).toHaveText(`${league} Win Probability Forecasting`);
  await expect(page.locator("aside.dashboard-sidebar")).toContainText(/Bet Profile/i);
  await expect(page.locator("aside.dashboard-sidebar")).toContainText(/Amount Bet/i);
  await expect(page.locator("nav.dashboard-nav")).toContainText(/Overview/i);
  const layoutMetrics = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
    sidebarRight: document.querySelector("aside.dashboard-sidebar")?.getBoundingClientRect().right ?? null,
    mainLeft: document.querySelector("main")?.getBoundingClientRect().left ?? null,
    navBottom: document.querySelector("nav.dashboard-nav")?.getBoundingClientRect().bottom ?? null,
    mainTop: document.querySelector("main")?.getBoundingClientRect().top ?? null,
  }));
  expect(layoutMetrics.scrollWidth).toBeLessThanOrEqual(layoutMetrics.viewportWidth + 32);
  if (layoutMetrics.sidebarRight !== null && layoutMetrics.mainLeft !== null) {
    expect(layoutMetrics.sidebarRight).toBeLessThan(layoutMetrics.mainLeft);
  }
  if (layoutMetrics.navBottom !== null && layoutMetrics.mainTop !== null) {
    expect(layoutMetrics.navBottom).toBeLessThan(layoutMetrics.mainTop + 1);
  }
}

async function expectHealthyRoute(
  page: Page,
  league: LeagueCode,
  route: RouteExpectation,
  pageIssues: ReturnType<typeof trackPageIssues>
) {
  await settle(page);
  await expectDashboardShell(page, league);
  await expect(page.locator("main")).toContainText(route.readyText, { timeout: 15_000 });
  await expect(page.locator("body")).not.toContainText(UI_ERROR_PATTERN);

  const issues = pageIssues.flush();
  expect(issues, `Unexpected browser/runtime failures on ${route.path} (${league})`).toEqual([]);
}

test.describe("dashboard smoke", () => {
  for (const league of ["NBA", "NHL"] as const) {
    test(`main navigation is healthy for ${league}`, async ({ page }) => {
      const pageIssues = trackPageIssues(page);
      await page.goto(buildLeaguePath("/", league));
      await expectHealthyRoute(page, league, mainRoutes[0], pageIssues);

      const themeToggle = page.getByRole("button", { name: /Switch to/i });
      const themeBefore = await page.locator("html").getAttribute("data-dashboard-theme");
      await themeToggle.click();
      await expect(page.locator("html")).not.toHaveAttribute("data-dashboard-theme", themeBefore || "");
      await themeToggle.click();
      await expect(page.locator("html")).toHaveAttribute("data-dashboard-theme", themeBefore || "market-board-dark");
      expect(pageIssues.flush(), "Theme toggle produced browser/runtime failures").toEqual([]);

      const alternateLeague: LeagueCode = league === "NBA" ? "NHL" : "NBA";
      await page.getByRole("link", { name: alternateLeague, exact: true }).click();
      await expectDashboardShell(page, alternateLeague);
      await page.getByRole("link", { name: league, exact: true }).click();
      await expectDashboardShell(page, league);
      expect(pageIssues.flush(), "League toggle produced browser/runtime failures").toEqual([]);

      for (const route of mainRoutes.slice(1)) {
        await test.step(`${league} ${route.label}`, async () => {
          await page.getByRole("link", { name: route.label, exact: true }).click();
          await expect(page).toHaveURL(new RegExp(`${route.path === "/" ? "/" : route.path}\\?league=${league}`));
          await expectHealthyRoute(page, league, route, pageIssues);
          if (route.interaction) {
            await route.interaction(page);
            await expectHealthyRoute(page, league, route, pageIssues);
          }
        });
      }
    });

    test(`secondary routes are healthy for ${league}`, async ({ page }) => {
      const pageIssues = trackPageIssues(page);

      for (const route of directRoutes) {
        await test.step(`${league} ${route.label}`, async () => {
          await page.goto(buildLeaguePath(route.path, league));
          await expectHealthyRoute(page, league, route, pageIssues);
        });
      }
    });
  }
});
