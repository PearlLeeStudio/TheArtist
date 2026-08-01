import { test } from 'node:test';
import assert from 'node:assert/strict';
import { PDF_HEIGHT, PDF_WIDTH, pdfMeasureLayout, pdfPageRanges, withoutPdfCreationDate } from './pdfLayout.ts';

test('PDF pagination keeps every bar once and never adds a trailing empty page', () => {
  assert.deepEqual(pdfPageRanges(28), [{ start: 0, end: 28 }]);
  assert.deepEqual(pdfPageRanges(29), [{ start: 0, end: 28 }, { start: 28, end: 29 }]);
  assert.deepEqual(pdfPageRanges(56), [{ start: 0, end: 28 }, { start: 28, end: 56 }]);
  for (const count of [1, 4, 8, 27, 28, 29, 55, 56, 57, 64, 512]) {
    const pages = pdfPageRanges(count);
    const indexes = pages.flatMap(({ start, end }) => Array.from({ length: end - start }, (_, i) => start + i));
    assert.deepEqual(indexes, Array.from({ length: count }, (_, i) => i));
    assert.ok(pages.every(({ start, end }) => end > start && end - start <= 28));
  }
  assert.throws(() => pdfPageRanges(0));
  assert.throws(() => pdfPageRanges(513));
});

test('seven rows fit on A4 with a title area and page footer', () => {
  for (let i = 0; i < 28; i++) {
    const layout = pdfMeasureLayout(i);
    assert.ok(layout.y >= 100);
    assert.ok(layout.x >= 0 && layout.x + layout.width <= PDF_WIDTH - 10);
    assert.ok(layout.y + 120 < PDF_HEIGHT - 40);
  }
  assert.equal(pdfMeasureLayout(3).y, pdfMeasureLayout(0).y);
  assert.ok(pdfMeasureLayout(4).y > pdfMeasureLayout(3).y);
});

test('PDF work timestamp is removed without changing byte offsets or other document bytes', () => {
  const text = '%PDF\n3 0 obj\n<< /Title (Sketch)\n/CreationDate (D:20000101000000+00\'00\')\n>>\nendobj\nxref';
  const input = new TextEncoder().encode(text);
  const output = withoutPdfCreationDate(input.buffer);
  const decoded = new TextDecoder().decode(output);
  assert.equal(output.length, input.length);
  assert.equal(decoded.includes('CreationDate'), false);
  assert.equal(decoded.includes('20000101'), false);
  assert.equal(decoded.indexOf('xref'), text.indexOf('xref'));
  assert.ok(decoded.includes('/Title (Sketch)'));
  assert.equal(new TextDecoder().decode(input), text);
});
