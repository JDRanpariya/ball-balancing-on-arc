// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://ball-on-arc.pages.dev',
  base: '/',
  vite: {
    plugins: [tailwindcss()],
    // onnxruntime-web is loaded entirely from CDN at runtime
    // (see onnx_inference.ts) to stay under Cloudflare Pages' 25 MiB
    // per-file limit; nothing to bundle or externalize here.
  },
});
