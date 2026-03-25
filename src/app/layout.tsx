import { Inter } from "next/font/google";
import "./globals.css";
import BackgroundAnimation from "@/components/BackgroundAnimation";

const inter = Inter({ subsets: ["latin"] });

import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "StockPro - Futuristic Trading",
  description: "Next-gen stock market analysis with real-time AI predictions.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning>
        <BackgroundAnimation />
        <main className="relative min-h-screen px-4 md:px-8 pt-24 lg:pt-32 pb-10">
          {children}
        </main>
      </body>
    </html>
  );
}
