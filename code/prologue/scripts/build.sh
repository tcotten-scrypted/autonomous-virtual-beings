#!/bin/bash
set -e

echo "Cleaning previous builds..."
rm -rf client/.next server/dist

echo "Running TypeScript checks..."
pnpm exec tsc --noEmit

echo "Building Next.js front-end..."
pnpm run build:client

echo "Building API server..."
pnpm exec tsc --project ./server/tsconfig.json

echo "Build completed!"
