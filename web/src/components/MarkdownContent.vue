<script setup lang="ts">
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import type { Attachment } from '../api'

const props = withDefaults(defineProps<{
  content: string
  attachments?: Attachment[]
}>(), { attachments: () => [] })

const emit = defineEmits<{ preview: [url: string] }>()

const imagePathPattern = /\.(?:jpe?g|png|gif|webp)(?:\/[^?#\s]*)?(?:[?#].*)?$/i

function sourceLabel(source: string) {
  return ({
    user_upload: '用户上传', forum_post: '论坛帖子', forum_search: '论坛搜索',
    web_search: '网页搜索', generated: '生成图片',
  } as Record<string, string>)[source] || '搜索图片'
}

function safeImageUrl(value: string) {
  try {
    const url = new URL(value, window.location.origin)
    return url.protocol === 'http:' || url.protocol === 'https:'
  } catch {
    return false
  }
}

function isExplicitImageUrl(value: string) {
  if (!safeImageUrl(value)) return false
  try {
    return imagePathPattern.test(new URL(value, window.location.origin).pathname)
  } catch {
    return false
  }
}

function matchingAttachment(sourceUrl: string) {
  return props.attachments.find(item =>
    item.source_url === sourceUrl || item.url === sourceUrl || `artifact://${item.artifact_id}` === sourceUrl,
  )
}

function buildSearchGallery(document: Document) {
  const links = Array.from(document.querySelectorAll<HTMLAnchorElement>('a[href]'))
  const candidates: Array<{
    link: HTMLAnchorElement
    sourceUrl: string
    displayUrl: string
    label: string
    attachment?: Attachment
  }> = []
  const seen = new Set<string>()

  for (const link of links) {
    const sourceUrl = link.getAttribute('href') || ''
    const attachment = matchingAttachment(sourceUrl)
    if (!attachment && !isExplicitImageUrl(sourceUrl)) continue
    const identity = attachment?.artifact_id || sourceUrl
    if (seen.has(identity)) {
      link.remove()
      continue
    }
    seen.add(identity)
    candidates.push({
      link,
      sourceUrl: attachment?.source_url || sourceUrl,
      displayUrl: attachment?.url || sourceUrl,
      label: link.textContent?.trim() || attachment?.filename || `图片 ${candidates.length + 1}`,
      attachment,
    })
  }
  if (!candidates.length) return

  const gallery = document.createElement('div')
  gallery.className = 'search-image-gallery'
  gallery.setAttribute('aria-label', '搜索结果图片')
  for (const [index, item] of candidates.entries()) {
    const figure = document.createElement('figure')
    figure.className = 'search-image-card'

    const preview = document.createElement('button')
    preview.type = 'button'
    preview.className = 'search-image-preview'
    preview.dataset.previewUrl = item.displayUrl
    preview.setAttribute('aria-label', `预览${item.label}`)
    const image = document.createElement('img')
    image.src = item.displayUrl
    image.alt = item.label
    image.loading = 'lazy'
    image.dataset.searchImage = String(index + 1)
    preview.appendChild(image)
    figure.appendChild(preview)

    const caption = document.createElement('figcaption')
    const label = document.createElement('span')
    label.textContent = item.attachment ? sourceLabel(item.attachment.source_kind) : '网页搜索'
    caption.appendChild(label)
    if (item.sourceUrl.startsWith('http')) {
      const source = document.createElement('a')
      source.href = item.sourceUrl
      source.target = '_blank'
      source.rel = 'noopener noreferrer'
      source.textContent = item.label
      caption.appendChild(source)
    }
    figure.appendChild(caption)
    gallery.appendChild(figure)
  }

  const firstLink = candidates[0].link
  const commonList = firstLink.closest('ul, ol')
  if (commonList && candidates.every(item => item.link.closest('ul, ol') === commonList)) {
    commonList.insertAdjacentElement('afterend', gallery)
  } else {
    firstLink.insertAdjacentElement('beforebegin', gallery)
  }
  for (const { link } of candidates) {
    const listItem = link.closest('li')
    link.remove()
    if (listItem && !listItem.textContent?.trim() && !listItem.querySelector('img')) listItem.remove()
  }
}

const rendered = computed(() => {
  const source = props.content || ''
  const parsed = marked.parse(source, {
    async: false,
    breaks: true,
    gfm: true,
  }) as string
  const sanitized = DOMPurify.sanitize(parsed, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['style'],
  })
  const document = new DOMParser().parseFromString(`<div id="markdown-root">${sanitized}</div>`, 'text/html')
  buildSearchGallery(document)
  return DOMPurify.sanitize(document.querySelector('#markdown-root')?.innerHTML || sanitized, {
    USE_PROFILES: { html: true },
    ADD_ATTR: ['target', 'rel', 'data-preview-url', 'data-search-image'],
    FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['style'],
  })
})

function previewImage(event: MouseEvent) {
  const target = event.target as HTMLElement
  const button = target.closest<HTMLElement>('[data-preview-url]')
  const url = button?.dataset.previewUrl
  if (!url || !safeImageUrl(url)) return
  event.preventDefault()
  emit('preview', url)
}
</script>

<template>
  <div class="markdown-body" @click="previewImage" v-html="rendered"></div>
</template>
