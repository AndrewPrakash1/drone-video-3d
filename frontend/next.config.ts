import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["@mkkellogg/gaussian-splats-3d"],
  // Browser opens http://127.0.0.1:43123. Without this, Next blocks the
  // client bundle and the page is a dead screenshot (no clicks, no globe).
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8765/:path*",
      },
    ];
  },
};

export default nextConfig;
