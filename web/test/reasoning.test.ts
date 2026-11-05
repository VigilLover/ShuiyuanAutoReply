import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import test from 'node:test'
import { compileScript, parse } from '@vue/compiler-sfc'
import { createRenderer, h, nextTick } from 'vue'
import ts from 'typescript'

const require = createRequire(import.meta.url)
const { descriptor } = parse(readFileSync(new URL('../src/components/ReasoningEvent.vue', import.meta.url), 'utf8'))
const compiled = compileScript(descriptor, { id: 'reasoning-test', inlineTemplate: true })
const js = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const exports: Record<string, any> = {}
new Function('require', 'exports', js)((name: string) => name === '@phosphor-icons/vue' ? { PhCaretRight: () => h('svg') } : require(name), exports)

type Element = { tag: string; children: Element[]; props: Record<string, any>; text: string; parent?: Element }
const element = (tag: string, text = ''): Element => ({ tag, text, children: [], props: {} })
const renderer = createRenderer<Element, Element>({
  createElement: tag => element(tag),
  createText: text => element('#text', text),
  createComment: text => element('#comment', text),
  setText: (node, text) => { node.text = text },
  setElementText: (node, text) => { node.text = text; node.children = [] },
  patchProp: (node, key, _previous, value) => { node.props[key] = value },
  insert(node, parent, anchor = null) {
    if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1)
    const index = anchor ? parent.children.indexOf(anchor) : -1
    parent.children.splice(index < 0 ? parent.children.length : index, 0, node)
    node.parent = parent
  },
  remove(node) { if (node.parent) node.parent.children.splice(node.parent.children.indexOf(node), 1) },
  parentNode: node => node.parent || null,
  nextSibling: node => node.parent?.children[node.parent.children.indexOf(node) + 1] || null,
})
function find(node: Element, tag: string): Element | undefined {
  return node.tag === tag ? node : node.children.map(child => find(child, tag)).find(Boolean)
}

test('reasoning starts collapsed, mounts the full escaped text on expand, and unmounts on collapse', async () => {
  const text = '<img src=x onerror=alert(1)>\n'.repeat(5000)
  const root = element('root')
  const app = renderer.createApp({ render: () => h(exports.default, { text }) })
  app.mount(root)
  const details = find(root, 'details')!
  assert.ok(details)
  assert.equal(Boolean(details.props.open), false)
  assert.equal(find(root, 'pre'), undefined)
  details.props.onToggle({ currentTarget: { open: true } })
  await nextTick()
  assert.equal(find(root, 'pre')?.text, text)
  assert.equal(find(root, 'img'), undefined)
  details.props.onToggle({ currentTarget: { open: false } })
  await nextTick()
  assert.equal(find(root, 'pre'), undefined)
  app.unmount()
})

test('missing or whitespace-only reasoning does not render a disclosure', () => {
  const root = element('root')
  const app = renderer.createApp({ render: () => h(exports.default, { text: ' \n ' }) })
  app.mount(root)
  assert.equal(find(root, 'details'), undefined)
  app.unmount()
})
