import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import test from 'node:test'
import { compileScript, parse } from '@vue/compiler-sfc'
import { createRenderer, h, nextTick, ref } from 'vue'
import ts from 'typescript'

const require = createRequire(import.meta.url)
function loadComponent(filename: string): Record<string, any> {
  const { descriptor } = parse(readFileSync(new URL(`../src/components/${filename}`, import.meta.url), 'utf8'))
  const compiled = compileScript(descriptor, { id: filename, inlineTemplate: true })
  const js = ts.transpileModule(compiled.content, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
  const exports: Record<string, any> = {}
  new Function('require', 'exports', js)((name: string) => {
    if (name === '@phosphor-icons/vue') return { PhCaretRight: () => h('svg') }
    if (name.startsWith('./') && name.endsWith('.vue')) return loadComponent(name.slice(2))
    return require(name)
  }, exports)
  return exports
}

const ReasoningEvent = loadComponent('ReasoningEvent.vue').default
const PromptEvent = loadComponent('PromptEvent.vue').default

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
  const app = renderer.createApp({ render: () => h(ReasoningEvent, { text }) })
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
  const app = renderer.createApp({ render: () => h(ReasoningEvent, { text: ' \n ' }) })
  app.mount(root)
  assert.equal(find(root, 'details'), undefined)
  app.unmount()
})

test('prompt disclosures preserve controlled open state and render message text safely', async () => {
  const open = ref(true)
  const text = '<img src=x onerror=alert(1)>\n原始提示词'
  const root = element('root')
  const app = renderer.createApp({ render: () => h(PromptEvent, {
    open: open.value,
    payload: { messages: [{ role: 'human', content: text }], scope: 'forum' },
  }) })
  app.mount(root)
  assert.equal(find(root, 'details')?.props.open, true)
  assert.equal(find(root, 'pre')?.text, text)
  assert.equal(find(root, 'img'), undefined)
  open.value = false
  await nextTick()
  assert.equal(find(root, 'details')?.props.open, false)
  assert.equal(find(root, 'pre'), undefined)
  open.value = true
  await nextTick()
  assert.equal(find(root, 'pre')?.text, text)
  app.unmount()
})
