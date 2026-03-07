import "./globals.css";
import type { Metadata, Viewport } from "next";
import DashboardHeader from "@/components/DashboardHeader";

const DEFAULT_DASHBOARD_THEME = "market-board-dark";
const THEME_STORAGE_KEY = "dashboard-theme";

export const metadata: Metadata = {
  title: "Chaos Index",
  description: "Predictions, calibration, diagnostics, and validation artifacts",
  applicationName: "Chaos Index",
  openGraph: {
    title: "Chaos Index",
    siteName: "Chaos Index",
    description: "Predictions, calibration, diagnostics, and validation artifacts",
  },
  twitter: {
    card: "summary",
    title: "Chaos Index",
    description: "Predictions, calibration, diagnostics, and validation artifacts",
  },
};

export const viewport: Viewport = {
  themeColor: "#06131a",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const themeInitializer = `(function(){try{var key=${JSON.stringify(THEME_STORAGE_KEY)};var fallback=${JSON.stringify(DEFAULT_DASHBOARD_THEME)};var stored=window.localStorage.getItem(key);var next=stored==="light"||stored==="market-board-dark"?stored:fallback;document.documentElement.setAttribute("data-dashboard-theme",next);}catch(e){document.documentElement.setAttribute("data-dashboard-theme",${JSON.stringify(DEFAULT_DASHBOARD_THEME)});}})();`;

  return (
    <html lang="en" data-dashboard-theme={DEFAULT_DASHBOARD_THEME} suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: themeInitializer }} />
        <div className="container">
          <DashboardHeader />
          <main style={{ paddingBottom: 48 }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
