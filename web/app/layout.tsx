import "./globals.css";
import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "NHL Forecast Dashboard",
  description: "Predictions, calibration, diagnostics, and validation artifacts",
};

const links = [
  ["/", "Overview"],
  ["/predictions", "Predictions"],
  ["/leaderboard", "Leaderboard"],
  ["/performance", "Performance"],
  ["/calibration", "Calibration"],
  ["/diagnostics", "Diagnostics"],
  ["/slices", "Slices"],
  ["/validation", "Validation"],
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="container">
          <h1 className="title" style={{ marginTop: 22 }}>
            NHL Win Probability Forecasting
          </h1>
          <div className="nav">
            {links.map(([href, label]) => (
              <Link href={href} key={href}>
                {label}
              </Link>
            ))}
          </div>
          <main style={{ paddingBottom: 48 }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
