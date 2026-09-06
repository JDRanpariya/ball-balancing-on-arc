# pi.website Design System — Reference Notes

Analyzed from: `pi.website/`, `/research/rlt`, `/research/knowledge_insulation`, `/blog/pi0`, `/blog/pi05`, `/blog/pistar06`

---

## Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| `background` | `#F5F4EF` | Page background (warm cream-gray) |
| `surface-alt` | `#EEECE4` | Card/infographic backgrounds |
| `surface-hover` | `#E6E0CB` | Borders on cards, subtle hover |
| `infographic-card` | `#E6E0CB` | Chart container borders |
| `infographic-card-highlight` | `#FBD45B` | Primary accent — golden yellow |
| `foreground` | `#000000` / `#111111` | Text, borders |
| `muted-foreground` | `#686868` / `#595959` | Secondary text |
| `axis-label` | `#A8A179` | Chart axis labels (olive-muted) |
| `grid-lines` | `#DFDCCC` | Horizontal grid lines in charts |
| `dot-pattern` | `#D1C9AC` | Dot grid overlays on chart areas |
| `border` | `#C0BDAD` | Subtle card borders |
| `green` | `#9CBE70` | "Base policy" bars, success |
| `yellow/gold` | `#FBD45B` / `#FBD35B` | "Ours" bars, highlight |
| `orange` | `#FFB847` | Secondary warm accent |
| `blue` | `#85BDFF` | Tertiary data color |
| `red/burnt` | `#D8683C` | Error/negative data |
| `teal` | `#B7C8C9` / `#C8DEDF` | Architecture diagram accents |

---

## Typography

| Element | Font | Size | Weight | Notes |
|---------|------|------|--------|-------|
| Page body | `ui-monospace` (system mono) | `14px` | 400 | Matches terminal aesthetic |
| H1 (title) | `Signifier` (serif) | `60px` | 400 | Elegant, editorial |
| H3 (section) | `Source Sans 3` (sans) | `24px` | 400 | Clean readable |
| Chart axis labels | `font-mono` | `9-10px` | 500 | Olive color `#A8A179` |
| Chart axis values | `font-mono tabular-nums` | `11px` | 500 | Black, right-aligned |
| Bar labels | `font-mono` | `8-10px` | 700 | Below bars, centered |
| Card titles | `font-mono` | `11-12px` | 600-700 | Uppercase sometimes |
| Metadata | `font-mono` | `text-sm` | 400 | Grid layout |

---

## Chart Design Patterns

### Bar Charts (Throughput/Performance)

```
Structure:
├── Container: bg-[#F5F4EF] p-1, no explicit border on outer
├── Title: font-mono text-[11px] font-medium text-[#111]
├── Y-axis label: text-[#A8A179] font-mono text-[9px] positioned absolute
├── Y-axis values: text-[#111] font-mono text-[11px] tabular-nums
├── Grid lines: h-px w-full bg-[#DFDCCC] (5 evenly spaced)
├── Bars:
│   ├── border border-black
│   ├── shadow-[3px_3px_0px_0px_#000]  ← KEY: hard offset shadow
│   ├── min-h-[3px] w-12 sm:w-14
│   ├── height set via inline style (percentage)
│   ├── Color: #9CBE70 (base), #FBD35B (ours/highlight)
│   └── Error bars: absolute positioned, w-px bg-black with w-2 caps
└── Bar labels: absolute below, font-mono text-[8-10px] font-bold
```

**Key characteristics:**
- Bars are solid color with `border border-black shadow-[3px_3px_0px_0px_#000]`
- NO rounded corners on bars
- Error bars (confidence intervals) as thin black lines with horizontal caps
- Comparison: green bar (base) vs yellow bar (ours) side by side
- Grid lines are subtle tan `#DFDCCC`
- Y-axis labels in olive/muted gold `#A8A179`

### Success Rate Bars (Blog pi05)

```
Structure:
- Horizontal bars with same border/shadow treatment
- Labels inside or beside bars
- Color coding: #9CBE70 (good), #FBD45B (best/ours)
- shadow-[2px_2px_0px_0px_#000] on smaller bars
- shadow-[3px_3px_0px_0px_#000] on larger bars
```

### Line Charts (SVG paths)

```
- Rendered as SVG <path> elements
- Stroke colors: #7C9D52 (green), #FFB847 (orange), #FBD45B (yellow), #FFE961 (light yellow)
- Grid: #F0EAD1 stroke for background lines
- No fill under lines (stroke only)
- Smooth curves (not stepped)
```

---

## Card / Container Patterns

### Highlighted Card ("ours" / featured)
```css
border border-black
shadow-[3px_3px_0px_#000]
bg-white /* or bg-[#FBD45B] for strong emphasis */
hover:shadow-[5px_5px_0px_#000]
transition-all
```

### Secondary Card (research paper)
```css
border border-[#D4D3CB]
bg-white/60
hover:border-[#C0BDAD]
hover:shadow-[3px_3px_0px_#C0BDAD]
transition-all
hover:bg-white
```

### Infographic Container
```css
bg-[#EEECE4]
border border-[#E6E0CB]
rounded-lg
p-4
```

### Data Panel / Chart Area
```css
border border-[#E6E0CB]
bg-[#F5F4EF]
p-4 pb-14  /* extra bottom for labels */
```

---

## Dot Grid Pattern

Used ONLY on chart/infographic areas, NOT full page:
```css
background-image: radial-gradient(circle, #D1C9AC 1px, transparent 1px);
background-size: 14px 14px;
opacity: 0.5;
```
Applied via an `absolute inset-0` overlay div.

---

## Layout Patterns

| Pattern | Implementation |
|---------|---------------|
| Page width | `max-w-xl` (homepage), `max-w-5xl` (research posts) |
| Padding | `p-4 md:p-12` (page), `p-4 md:p-6` (cards) |
| Timeline | `border-l border-gray-300` with dot markers |
| Nav links | `underline underline-offset-8 hover:decoration-2` |
| Metadata | `grid md:grid-cols-[120px_1fr] gap-0.5 font-mono text-sm` |
| Sections | No explicit `<section>` styling, just margin/content |

---

## Interactive Elements

### Video Player Controls
```css
/* Play button */
bg-infographic-card-highlight  /* #FBD45B */
rounded-full OR square
size-8

/* Progress bar */
relative h-1.5 rounded-full bg-infographic-card  /* #E6E0CB */
/* Fill: */ absolute h-full bg-infographic-card-highlight  /* #FBD45B */
```

### Buttons
```css
/* Primary */
border border-black shadow-[3px_3px_0px_#000]
bg-white hover:shadow-[5px_5px_0px_#000]

/* On hover: translate slightly up */
hover:-translate-y-0.5
transition-all
```

### Tooltips
```css
bg-white shadow z-50 px-2 py-1
data-state="closed" / "open"
```

---

## Key Design Principles

1. **Neo-brutalist:** Hard shadows (`3px_3px_0px_0px_#000`), solid borders, no blur/glow
2. **Warm neutral palette:** Cream/tan/olive tones, not cold grays
3. **Monospace-first:** Body text is monospace, headings are serif or sans
4. **Data-forward:** Charts use minimal decoration, strong black outlines on bars
5. **No rounded corners** on data elements (bars, chart containers)
6. **Comparison via color:** Green = baseline, Yellow = ours/highlight
7. **Constrained width:** Content never goes full-bleed (even charts)
8. **Dot pattern sparingly:** Only as subtle texture on specific infographic areas
9. **Error bars always shown:** Confidence intervals as black I-beams
10. **Label positioning:** Below bars (vertically stacked text), axis labels olive-colored

---

## Applying to Ball-on-Arc Website

Our current implementation already uses:
- ✅ `#F5F4EF` background
- ✅ `border-2 border-black shadow-[3px/4px]` cards
- ✅ `#FBD45B` yellow for RL highlight
- ✅ `#9CBE70` green for classical
- ✅ Monospace body + sans headings
- ✅ `font-mono tabular-nums` for data

Could enhance:
- Add error bars / confidence intervals to bar chart
- Use olive `#A8A179` for chart axis labels
- Add dot-grid pattern to chart container backgrounds
- Use `border border-black shadow-[3px_3px_0px_0px_#000]` on individual bars (not just containers)
