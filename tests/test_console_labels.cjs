// Run: node --test tests/test_console_labels.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Load the real console helpers without starting browser polling.
const source = fs.readFileSync(path.join(__dirname, '../console/app.js'), 'utf8').replace(/\ninit\(\);\s*$/, '\n');
const scope = { window: {}, document: { querySelector: () => null } };
vm.createContext(scope);
vm.runInContext(source, scope);
const records = fs.readFileSync(path.join(__dirname, '../demo/bundle/explanations.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const people = records.map((r) => JSON.parse(r.raw_text).person);

test('an empty PPE list never claims visible equipment', () => {
  const person = people.find((p) => p.ppe_visible.length === 0);
  assert.ok(person, 'recorded empty-list example must exist');
  assert.equal(scope.formatPpeVisible(person.ppe_visible), '보이는 보호구 없음');
});

test('visible PPE items are named rather than reduced to a boolean', () => {
  const person = people.find((p) => p.ppe_visible.length > 0);
  assert.ok(person, 'recorded visible-equipment example must exist');
  assert.equal(scope.formatPpeVisible(person.ppe_visible), person.ppe_visible.join(' · '));
});

test('missing or invalid PPE data is unknown', () => {
  for (const value of [null, undefined, true, false, 'unknown']) {
    assert.equal(scope.formatPpeVisible(value), '불명');
  }
});
