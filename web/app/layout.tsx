import "./globals.css";
import type { Metadata } from "next";
import DashboardHeader from "@/components/DashboardHeader";

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

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <DashboardHeader />
          <main style={{ paddingBottom: 48 }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
