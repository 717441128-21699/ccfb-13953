import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    environment: 'node',
    testTimeout: 30000,
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});
