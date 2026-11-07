#!/usr/bin/env node
/**
 * SceneProof console smoke test (Playwright + Chromium, no extra deps).
 *
 *   node scripts/console_check.mjs [outDir]
 *
 * Opens http://127.0.0.1:8000/ (override with SP_CONSOLE_URL), waits for the event list, selects the
 * first event, checks that the detail panel shows the zone name and a VLM section, walks one
 * needs_review event through 검토 시작 → 처리 완료 (resolution + note), asserts there were no
 * console errors / page errors, and saves 01_overview.png, 02_event_detail.png, 03_review_done.png
 * (plus 04_original_video.png and 05_audit.png as extras) into outDir.
 */
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { createRequire } from 'node:module';

// Playwright is installed globally here (reachable via require/NODE_PATH, not via ESM import).
const { chromium } = createRequire(import.meta.url)('playwright');

// usage: node scripts/console_check.mjs [outDir] [--static <url> | <staticUrl>]
//   default      full read/write check against the live server (SP_CONSOLE_URL or http://127.0.0.1:8000/)
//   --static URL reduced read-only check against a static build (window.SP_STATIC_BASE), e.g. http://127.0.0.1:8090/
const argv = process.argv.slice(2);
let OUT = '/tmp/claude-0/-home-user-chartwire/d5801663-d4d9-5a80-b5d4-4f5fea308eaf/scratchpad/ui';
let STATIC_URL = null;
{
  const positional = [];
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--static') STATIC_URL = argv[++i];
    else positional.push(argv[i]);
  }
  if (positional[0]) OUT = positional[0];
  if (positional[1]) STATIC_URL = positional[1];
}
const BASE = STATIC_URL || process.env.SP_CONSOLE_URL || 'http://127.0.0.1:8000/';
mkdirSync(OUT, { recursive: true });

const failures = [];
const consoleErrors = [];
const check = (cond, msg) => { if (cond) console.log(`  ok   ${msg}`); else { console.log(`  FAIL ${msg}`); failures.push(msg); } };
const shot = async (page, name) => { const p = join(OUT, name); await page.screenshot({ path: p }); console.log(`  shot ${p}`); };

async function runLiveFlow(page) {

  const first = page.locator('.ev-row').first();
  const zone = (await first.locator('.ev-zone').innerText()).trim();
  await first.click();
  await page.waitForSelector('#detail .tab-body');
  await page.waitForFunction(() => document.querySelector('.ev-row.selected') !== null);
  const detailText = await page.locator('#detail').innerText();
  check(detailText.includes(zone), `detail panel shows zone name "${zone}"`);
  check((await page.locator('#sel-header').innerText()).includes(zone), 'selected-event header shows zone name');
  check((await page.locator('#thumbs .thumb').count()) === 5, 'five key-frame thumbnails rendered');
  check((await page.locator('#overlay polygon').count()) === 1, 'zone polygon drawn over the video');
  await page.locator('.tab[data-tab="vlm"]').click();
  await page.waitForSelector('.vlm-footer');
  const vlmText = await page.locator('#detail').innerText();
  check(vlmText.includes('모델 응답은 참고용이며 사건의 정답이 아닙니다'), 'VLM section shows the disclaimer footer');
  check(/판단의 한계|해석 대기 중|해석 중|다시 해석|해석 요청/.test(vlmText), 'VLM section renders a status-specific body');
  check((await page.locator('#timeline .tl-bar').count()) > 0, 'timeline bars rendered for the selected camera');
  await page.waitForTimeout(500);
  await shot(page, '02_event_detail.png');

  console.log('\n[3] review flow: 검토 시작 → 처리 완료');
  const candidate = page.locator('.ev-row:has(.chip.st-needs_review)').first();
  check((await candidate.count()) > 0, 'found an event in 확인 필요 state');
  await candidate.click();
  await page.locator('.tab[data-tab="review"]').click();
  await page.waitForSelector('#btn-start-review');
  await page.locator('#btn-start-review').click();
  await page.waitForSelector('#btn-resolve', { timeout: 10000 });
  check((await page.locator('#detail .review-now .chip').first().innerText()).includes('검토 중'), 'state changed to 검토 중');
  await page.locator('#btn-resolve').click();
  await page.waitForSelector('#resolve-form');
  await page.selectOption('#res-select', 'dismissed');
  await page.fill('#res-note', '스모크 테스트: 작업자 정상 통행으로 판단');
  await page.locator('#btn-resolve-save').click();
  await page.waitForSelector('#btn-reopen', { timeout: 10000 });
  const reviewText = await page.locator('#detail').innerText();
  check(reviewText.includes('처리 완료'), 'state changed to 처리 완료');
  check(reviewText.includes('오탐 처리'), 'resolution 오탐 처리 shown');
  check(reviewText.includes('스모크 테스트'), 'review log shows the note');
  check((await page.locator('.ev-row.selected .chip.st-resolved').count()) === 1, 'list row chip updated to 처리 완료');
  await page.waitForTimeout(400);
  await shot(page, '03_review_done.png');

  console.log('\n[4] extras: original-video mode, keyboard, audit drawer');
  const origBtn = page.locator('#stage-bar button', { hasText: '원본 영상에서 보기' });
  if (await origBtn.count()) {
    await origBtn.click();
    await page.waitForFunction(() => { const v = document.querySelector('#video'); return v && v.readyState >= 1; }, null, { timeout: 15000 }).catch(() => {});
    await page.waitForTimeout(600);
    check((await page.locator('#stage-span').count()) === 1 && !(await page.locator('#stage-span').evaluate((e) => e.classList.contains('hidden'))), 'original-video mode shows event-span label');
    await shot(page, '04_original_video.png');
    await page.locator('#stage-bar button', { hasText: '근거 클립' }).click();
  } else {
    console.log('  skip original-video toggle (no matching file in /api/videos)');
  }
  await page.keyboard.press('ArrowDown');
  await page.waitForTimeout(300);
  await page.keyboard.press('r');
  await page.waitForTimeout(200);
  check((await page.locator('#stage-bar button', { hasText: '원본 프레임' }).count()) === 1, 'r key toggled to raw frames');
  await page.locator('#audit-toggle').click();
  await page.waitForSelector('#audit-body .audit-row');
  check((await page.locator('#audit-body .audit-row').count()) >= 2, 'audit drawer lists review log rows');
  await shot(page, '05_audit.png');
  await page.locator('#audit-close').click();

  console.log('\n[5] console errors');
  check(consoleErrors.length === 0, `no console/page/request errors (${consoleErrors.length})`);
  for (const e of consoleErrors) console.log(`     - ${e}`);
}

/** Reduced read-only check for a static build (no server; JSON/media served as files). */
async function runStatic(page) {
  console.log('\n[S1] static mode badge / export links');
  const badge = await page.locator('#mode-badge').innerText();
  check(badge.includes('저장된 실행 결과 재생 (정적)'), `mode badge reads static replay ("${badge}")`);
  check(await page.locator('#mode-badge').evaluate((e) => e.classList.contains('replay')), 'static badge uses the amber replay style');
  check(await page.locator('#export-csv').evaluate((e) => e.classList.contains('hidden')), 'CSV export hidden');
  check(/data\/export\.json$/.test(await page.locator('#export-json').getAttribute('href')), 'JSON export points at data/export.json');
  check((await page.locator('#cameras .cam-actions button:not([disabled])').count()) === 0, 'camera 분석 시작/정지 buttons disabled');
  await shot(page, 'static_01_overview.png');

  console.log('\n[S2] select first event: frames + clip from data/');
  const first = page.locator('.ev-row').first();
  const zone = (await first.locator('.ev-zone').innerText()).trim();
  await first.click();
  await page.waitForSelector('#thumbs .thumb img[src]');
  check((await page.locator('#detail').innerText()).includes(zone), `detail panel shows zone name "${zone}"`);
  const thumbSrc = await page.locator('#thumbs .thumb img[src]').first().getAttribute('src');
  check(/^data\/evidence\//.test(thumbSrc), `key frames load from data/evidence/ (${thumbSrc})`);
  const frameOk = await page.locator('#frame-view img').evaluate((img) => img.complete && img.naturalWidth > 0);
  check(frameOk, 'large key frame decoded');
  const vsrc = await page.locator('#video').getAttribute('src');
  check(/^data\/evidence\/.*clip\.mp4$/.test(vsrc || ''), `clip src is data/evidence/.../clip.mp4 (${vsrc})`);
  const playing = await page.waitForFunction(() => { const v = document.querySelector('#video'); return v.readyState >= 2 && (v.currentTime > 0 || !v.paused); }, null, { timeout: 15000 }).then(() => true).catch(() => false);
  check(playing, 'clip loaded and is playing');
  check((await page.locator('#overlay polygon').count()) === 1, 'zone polygon drawn');
  check((await page.locator('.ev-row .mini-btn').count()) === 0, 'no VLM retry buttons in the list');

  console.log('\n[S3] read-only review / VLM tabs');
  await page.locator('.tab[data-tab="vlm"]').click();
  await page.waitForSelector('.vlm-footer');
  check((await page.locator('#detail button:has-text("해석"):not([disabled])').count()) === 0, 'VLM (재)해석 buttons disabled');
  await page.locator('.tab[data-tab="review"]').click();
  await page.waitForSelector('#review-static-note');
  check((await page.locator('#detail .review-actions button:not([disabled])').count()) === 0, 'review transition buttons disabled');
  check((await page.locator('#review-static-note').innerText()).includes('정적 재생 모드에서는 변경할 수 없습니다'), 'explanation text shown next to the buttons');
  const toastBefore = await page.locator('.toast').count();
  await page.locator('#detail .review-actions button').first().evaluate((b) => { b.disabled = false; b.click(); });
  await page.waitForTimeout(200);
  check((await page.locator('.toast').count()) > toastBefore && (await page.locator('.toast').last().innerText()).includes('정적 재생 모드'), 'forced click only shows the static-mode toast');
  check((await page.locator('#detail .review-now .chip').first().innerText()).includes('확인 필요'), 'state unchanged after forced click');
  const origBtn = page.locator('#stage-bar button', { hasText: '원본 영상에서 보기' });
  if (await origBtn.count()) {
    await origBtn.click();
    await page.waitForTimeout(400);
    const osrc = await page.locator('#video').getAttribute('src');
    check(/^data\/videos\//.test(osrc || ''), `original video src is data/videos/… (${osrc})`);
  }
  await page.waitForTimeout(300);
  await shot(page, 'static_02_event_detail.png');

  console.log('\n[S4] console errors');
  check(consoleErrors.length === 0, `no console/page/request errors (${consoleErrors.length})`);
  for (const e of consoleErrors) console.log(`     - ${e}`);
}

const browser = await chromium.launch();
try {
  const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, locale: 'ko-KR' });
  const page = await ctx.newPage();
  page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`));
  page.on('requestfailed', (r) => {
    const why = (r.failure() && r.failure().errorText) || '';
    // The browser aborts an in-flight media download whenever the <video>/<img> source changes; that is not an error.
    if (why === 'net::ERR_ABORTED' && r.resourceType() === 'media') return;
    consoleErrors.push(`requestfailed: ${r.url()} ${why}`);
  });
  page.on('response', (r) => { if (r.status() >= 400 && (r.url().includes('/api/') || (STATIC_URL && r.url().includes('/data/')))) consoleErrors.push(`http ${r.status()} ${r.url()}`); });

  console.log(`\n[1] open ${BASE}`);
  await page.goto(BASE, { waitUntil: 'domcontentloaded' });
  await page.waitForSelector('.ev-row', { timeout: 20000 });
  await page.waitForFunction(() => document.querySelectorAll('.ev-row').length > 0 && !document.querySelector('#mode-badge.unknown'));
  const rows = await page.locator('.ev-row').count();
  check(rows > 0, `event list rendered (${rows} rows)`);
  const badge = await page.locator('#mode-badge').innerText();
  check(/실시간 분석|저장된 실행 결과 재생/.test(badge), `mode badge shows Korean mode label ("${badge}")`);
  check((await page.locator('#cameras .cam-row').count()) > 0, 'camera rows rendered in the left column');
  await page.waitForTimeout(700);
  await shot(page, STATIC_URL ? 'static_00_loaded.png' : '01_overview.png');

  if (STATIC_URL) await runStatic(page); else await runLiveFlow(page);
} catch (e) {
  failures.push(`exception: ${e.message}`);
  console.log(`  FAIL exception: ${e.stack || e.message}`);
} finally {
  await browser.close();
}
console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
