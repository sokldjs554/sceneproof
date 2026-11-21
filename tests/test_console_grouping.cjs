// Task I: the console's "grouped only" filter and the grouping-gap setting.
// Run: node --test tests/test_console_grouping.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Load the real console code without starting browser polling (same approach as test_console_labels.cjs).
const source = fs.readFileSync(path.join(__dirname, '../console/app.js'), 'utf8').replace(/\ninit\(\);\s*$/, '\n');
const scope = { window: {}, document: { querySelector: () => null } };
vm.createContext(scope);
vm.runInContext(source, scope);

const group = (ids) => ({ group_id: ids[0], size: ids.length, event_ids: ids });
const ev = (id, extra = {}) => ({
  event_id: id, camera_id: 'cam-a', camera_name: 'A동 입구', zone_name: '통제구역', kind: 'EVENT_KIND_ZONE_ENTRY',
  review: { state: 'needs_review' }, concurrent: null, ...extra,
});
const EVENTS = [
  ev('e1', { concurrent: group(['e1', 'e2']) }),
  ev('e2', { concurrent: group(['e1', 'e2']), review: { state: 'in_review' } }),
  ev('e3'),
  ev('e4', { camera_id: 'cam-b', camera_name: 'B동 작업대', zone_name: '작업대 반경', kind: 'EVENT_KIND_ZONE_DWELL' }),
  ev('e5', { concurrent: { group_id: 'e5', size: 1, event_ids: ['e5'] } }),
];
const NONE = { camera: '', state: '', kind: '', q: '', grouped: false };
const ids = (list) => Array.from(list, (e) => e.event_id);

test('filterEvents with no filter returns every event in the same order', () => {
  assert.deepEqual(ids(scope.filterEvents(EVENTS, NONE)), ['e1', 'e2', 'e3', 'e4', 'e5']);
});

test('filterEvents keeps the existing filters: camera, review state, kind, search text', () => {
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, camera: 'cam-b' })), ['e4']);
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, state: 'in_review' })), ['e2']);
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, kind: 'EVENT_KIND_ZONE_DWELL' })), ['e4']);
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, q: '  작업대 ' })), ['e4']);      // zone or camera name, trimmed
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, q: 'E3' })), ['e3']);            // event id, any letter case
});

test('filterEvents grouped=true keeps only events that were in the zone together with another event', () => {
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, grouped: true })), ['e1', 'e2']);   // not e3 (no group), not e5 (group of one)
});

test('filterEvents combines grouped with the other filters', () => {
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, grouped: true, state: 'needs_review' })), ['e1']);
  assert.deepEqual(ids(scope.filterEvents(EVENTS, { ...NONE, grouped: true, camera: 'cam-b' })), []);
});

test('filterEvents does not change the list it is given', () => {
  const before = JSON.stringify(EVENTS);
  scope.filterEvents(EVENTS, { ...NONE, grouped: true, q: 'e' });
  assert.equal(JSON.stringify(EVENTS), before);
});

test('eventsApiPath adds group_gap_s only when a gap is set', () => {
  assert.equal(scope.eventsApiPath(0), '/api/events?limit=500');
  assert.equal(scope.eventsApiPath(undefined), '/api/events?limit=500');
  assert.equal(scope.eventsApiPath(null), '/api/events?limit=500');
  assert.equal(scope.eventsApiPath(2), '/api/events?limit=500&group_gap_s=2');
  assert.equal(scope.eventsApiPath(1.5), '/api/events?limit=500&group_gap_s=1.5');
});

test('parseGapSeconds accepts 0 to 60 seconds and treats an empty box as 0', () => {
  for (const [text, want] of [['', 0], ['   ', 0], [null, 0], [undefined, 0], ['0', 0], ['1.5', 1.5], [' 2 ', 2], ['60', 60], [3, 3]]) {
    assert.equal(scope.parseGapSeconds(text), want, JSON.stringify(text));
  }
});

test('parseGapSeconds returns null for anything the server would refuse', () => {
  for (const text of ['-1', '60.1', '61', 'abc', '1,5', '2초', 'NaN', 'Infinity']) {
    assert.equal(scope.parseGapSeconds(text), null, JSON.stringify(text));
  }
});
