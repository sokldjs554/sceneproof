#!/usr/bin/env node
/**
 * ZoneWatch console smoke test (Playwright + Chromium, no extra deps).
 *
 *   node scripts/console_check.mjs [outDir]
 *
 * Opens http://127.0.0.1:8000/ (override with ZW_CONSOLE_URL), waits for the event list, selects the
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

const BASE = process.env.ZW_CONSOLE_URL || 'http://127.0.0.1:8000/';
const OUT = process.argv[2] || '/tmp/claude-0/-home-user-chartwire/d5801663-d4d9-5a80-b5d4-4f5fea308eaf/scratchpad/ui';
mkdirSync(OUT, { recursive: true });

const failures = [];
const consoleErrors = [];
const check = (cond, msg) => { if (cond) console.log(`  ok   ${msg}`); else { console.log(`  FAIL ${msg}`); failures.push(msg); } };
const shot = async (page, name) => { const p = join(OUT, name); await page.screenshot({ path: p }); console.log(`  shot ${p}`); };

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
  page.on('response', (r) => { if (r.status() >= 400 && r.url().includes('/api/')) consoleErrors.push(`http ${r.status()} ${r.url()}`); });

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
  await shot(page, '01_overview.png');

  console.log('\n[2] select first event, check detail');
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
} catch (e) {
  failures.push(`exception: ${e.message}`);
  console.log(`  FAIL exception: ${e.stack || e.message}`);
} finally {
  await browser.close();
}
console.log(failures.length ? `\n${failures.length} check(s) failed` : '\nall checks passed');
process.exit(failures.length ? 1 : 0);
