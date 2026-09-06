# Astro Starter Kit: Minimal

```sh
bun create astro@latest -- --template minimal
```

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
/
├── public/
├── src/
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command               | Action                                           |
| :-------------------- | :----------------------------------------------- |
| `bun install`         | Installs dependencies                            |
| `bun dev`             | Starts local dev server at `localhost:4321`      |
| `bun build`           | Build your production site to `./dist/`          |
| `bun preview`         | Preview your build locally, before deploying     |
| `bun astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `bun astro -- --help` | Get help using the Astro CLI                     |

Branch: `web` — it is the dedicated branch where `website/` is tracked.

git checkout web
cd website

# 1. edit source in website/src/ (pages, components, sim scripts)

# 2. build

bun install # only needed if node_modules is missing
bun run build # outputs to website/dist/

# 3. deploy to production

npx wrangler pages deploy dist --project-name ball-on-arc --branch main --commit-dirty=true

# 4. commit and push

git add website
git commit -m "web: <what changed>"
git push origin web

Three gotchas worth remembering:

1. --branch main is mandatory — the Pages project's production branch is main, so without it your deploy lands
   as a preview at <hash>.ball-on-arc.pages.dev and the real site stays stale (this bit us once).
1. Wrangler auth: deploys go through your private-relay Cloudflare account, already logged in on this machine
   (npx wrangler whoami to confirm). Don't log in with a personal account.
1. Pre-deploy sanity check: search `dist/` for stale links or publication-status
   text. If you add images, strip unnecessary metadata first.

Verify after deploy with a hard-refresh (or curl -s "https://ball-on-arc.pages.dev/?v=123" | grep
<something-you-changed>) since the CDN edge can serve the old page for a minute.

## 👀 Want to learn more?

Feel free to check [our documentation](https://docs.astro.build) or jump into our [Discord server](https://astro.build/chat).
