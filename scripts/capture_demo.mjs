// Captures documentation screenshots (deviceScaleFactor 2) and a short demo video (webm) of the
// console flow against a running server. Records which git commit produced them.
//   node scripts/capture_demo.mjs http://127.0.0.1:8000 docs/images
import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const base = process.argv[2] || 'http://127.0.0.1:8000';
const out = process.argv[3] || 'docs/images';
fs.mkdirSync(out, { recursive: true });
const git = execSync('git rev-parse --short HEAD').toString().trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2, locale: 'ko-KR',
  recordVideo: { dir: out, size: { width: 1440, height: 900 } } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });

await page.goto(base + '/', { waitUntil: 'networkidle' });
await page.waitForSelector('[data-event-id], .event-row, .empty-state', { timeout: 30000 }).catch(() => {});
await sleep(1500);
await page.screenshot({ path: path.join(out, '01_overview.png') });

const rows = page.locator('[data-event-id]');
const n = await rows.count();
if (n > 0) {
  await rows.nth(Math.min(1, n - 1)).click();
  await sleep(1500);
  await page.screenshot({ path: path.join(out, '02_event_cv_evidence.png') });
  const vlmTab = page.getByRole('button', { name: /VLM 설명/ }).first();
  if (await vlmTab.count()) { await vlmTab.click(); await sleep(800); await page.screenshot({ path: path.join(out, '03_event_vlm.png') }); }
  const reviewTab = page.getByRole('button', { name: /처리 상태/ }).first();
  if (await reviewTab.count()) { await reviewTab.click(); await sleep(800); await page.screenshot({ path: path.join(out, '04_event_review.png') }); }
  const original = page.getByRole('button', { name: /원본 영상에서 보기/ }).first();
  if (await original.count() && await original.isEnabled()) { await original.click(); await sleep(2500); await page.screenshot({ path: path.join(out, '05_original_video.png') }); }
  const details = page.locator('details').first();
  if (await details.count()) { await details.evaluate((d) => { d.open = true; }); await details.scrollIntoViewIfNeeded(); await sleep(500); await page.screenshot({ path: path.join(out, '06_engineer_details.png') }); }
}
const audit = page.getByRole('button', { name: /처리 기록/ }).first();
if (await audit.count()) { await audit.click(); await sleep(800); await page.screenshot({ path: path.join(out, '07_audit.png') }); await page.keyboard.press('Escape'); }
await sleep(1000);
await ctx.close();
await browser.close();
// rename the recorded video
const vids = fs.readdirSync(out).filter((f) => f.endsWith('.webm') && f !== 'demo.webm');
if (vids.length) fs.renameSync(path.join(out, vids[vids.length - 1]), path.join(out, 'demo.webm'));
fs.writeFileSync(path.join(out, 'README.md'), `# 화면 캡처·시연 영상\n\n생성 커밋: \`${git}\` · 생성 시각: ${new Date().toISOString()} · 대상: ${base}\n\n` +
  `| 파일 | 내용 |\n|---|---|\n| 01_overview.png | 첫 화면: 카메라 상태 · 선택 사건 영상(구역·추적 상자 오버레이) · 타임라인 · 사건 목록 |\n` +
  `| 02_event_cv_evidence.png | 사건 상세 — CV 근거(구역·트랙·시각·키프레임) |\n| 03_event_vlm.png | 사건 상세 — VLM 설명(관찰·판단의 한계) |\n` +
  `| 04_event_review.png | 처리 상태 전이와 처리 이력 |\n| 05_original_video.png | 원본 영상에서 사건 구간 보기 |\n| 06_engineer_details.png | 상세 정보(모델·버전·지연·원시 응답) |\n` +
  `| 07_audit.png | 처리 기록(감사 로그) |\n| demo.webm | 위 흐름을 녹화한 짧은 시연 영상 (Playwright 녹화) |\n\n` +
  `\`node scripts/capture_demo.mjs <server-url> docs/images\` 로 재생성합니다. 콘솔 오류: ${errors.length}건${errors.length ? ' — ' + errors.slice(0, 3).join(' | ') : ''}\n`);
console.log(JSON.stringify({ git, screenshots: fs.readdirSync(out).filter((f) => f.endsWith('.png')).length, errors }));
