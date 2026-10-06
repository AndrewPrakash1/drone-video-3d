import type { Metadata } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OnePass — SIH26158",
  description: "Single-pass drone video to a georeferenced, metrically checkable 3D model.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full bg-[var(--apple-bg)] text-[var(--apple-text)]">{children}</body>
    </html>
  );
}
