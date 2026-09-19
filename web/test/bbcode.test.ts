import assert from 'node:assert/strict'
import test from 'node:test'
import { bbcodeToHtml, prepareReplyMarkdown, resolveUploadUrls } from '../src/bbcode.ts'

test('upload short urls resolve to the forum short-url path', () => {
  assert.equal(
    resolveUploadUrls('![图](upload://gXKtlHKPj337QAWosunDgv2yP7R.jpeg)'),
    '![图](https://shuiyuan.sjtu.edu.cn/uploads/short-url/gXKtlHKPj337QAWosunDgv2yP7R.jpeg)',
  )
})

test('details, grid and quotes become html blocks with markdown kept inside', () => {
  const html = bbcodeToHtml('[details=长文]**粗体**[/details][grid]![a](upload://a.png)[/grid][quote="alice, post:3, topic:42"]hi[/quote]')
  assert.match(html, /<details class="bbcode-details"><summary>长文<\/summary>/)
  assert.match(html, /\*\*粗体\*\*/)
  assert.match(html, /<div class="bbcode-grid">/)
  assert.match(html, /<blockquote class="bbcode-quote"><cite>@alice · <a href="https:\/\/shuiyuan.sjtu.edu.cn\/t\/42\/3"/)
})

test('img and url tags only accept http(s) or upload targets', () => {
  assert.equal(bbcodeToHtml('[img]upload://x.png[/img]'), '![](https://shuiyuan.sjtu.edu.cn/uploads/short-url/x.png)')
  assert.equal(bbcodeToHtml('[img]javascript:alert(1)[/img]'), '')
  assert.equal(bbcodeToHtml('[url=https://example.com]站点[/url]'), '[站点](https://example.com)')
  assert.equal(bbcodeToHtml('[url=data:text/html;x]坏链接[/url]'), '坏链接')
  assert.equal(bbcodeToHtml('[url]https://example.com/a[/url]'), '[https://example.com/a](https://example.com/a)')
})

test('summary titles are escaped and the pipeline composes', () => {
  const html = prepareReplyMarkdown('[details=<b>x</b>]![](upload://q.jpeg)[/details]')
  assert.match(html, /<summary>&lt;b&gt;x&lt;\/b&gt;<\/summary>/)
  assert.match(html, /uploads\/short-url\/q\.jpeg/)
})
