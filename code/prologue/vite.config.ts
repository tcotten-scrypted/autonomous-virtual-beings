import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path, { dirname } from "path";
import { fileURLToPath } from "url";

// Resolve __dirname for ESM environments
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@db": path.resolve(__dirname, "db"), // Alias for database directory
      "@": path.resolve(__dirname, "client", "src"), // Alias for client source directory
    },
  },
  root: path.resolve(__dirname, "client"), // Set the root directory to the client folder
  build: {
    outDir: path.resolve(__dirname, "dist/public"), // Output directory for the build
    emptyOutDir: true, // Clean output directory before building
  },
});
