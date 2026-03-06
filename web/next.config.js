const isStaticExport = process.env.STATIC_EXPORT === "1";
const basePath = isStaticExport ? process.env.PAGES_BASE_PATH || "" : "";

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      allowedOrigins: ["localhost:3000"]
    }
  },
  ...(isStaticExport
    ? {
        output: "export",
        trailingSlash: true,
        ...(basePath ? { basePath } : {}),
      }
    : {}),
};

module.exports = nextConfig;
