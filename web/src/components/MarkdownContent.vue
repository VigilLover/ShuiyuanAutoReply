<script setup lang="ts">
import { computed } from 'vue'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import type { Attachment } from '../api'

const props = withDefaults(defineProps<{
  content: string
  attachments?: Attachment[]
  showUnreferencedAttachments?: boolean
}>(), {
  attachments: () => [],
  showUnreferencedAttachments: false,
})

const emit = defineEmits<{ preview: [url: string] }>()

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

function matchingAttachment(sourceUrl: string) {
  return props.attachments.find(item =>
    item.source_url === sourceUrl || item.url === sourceUrl || `artifact://${item.artifact_id}` === sourceUrl,
  )
}

function localizeInlineImages(document: Document, consumed: Set<string>) {
  for (const image of Array.from(document.querySelectorAll<HTMLImageElement>('img[src]'))) {
    const sourceUrl = image.getAttribute('src') || ''
    const attachment = matchingAttachment(sourceUrl)
    if (attachment) {
      image.src = attachment.url
      image.dataset.previewUrl = attachment.url
      image.dataset.artifactId = attachment.artifact_id
      consumed.add(attachment.artifact_id)
      continue
    }
    if (!sourceUrl.startsWith('http://') && !sourceUrl.startsWith('https://')) continue
    const link = document.createElement('a')
    link.href = sourceUrl
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.textContent = image.alt || '查看原图'
    image.replaceWith(link)
  }
}

function buildAttachmentGallery(document: Document, attachments: Attachment[], deferImages = false) {
  const gallery = document.createElement('div')
  gallery.className = 'attachment-grid'
  for (const attachment of attachments) {
    const figure = document.createElement('figure')
    figure.className = 'message-image-card'
    const preview = document.createElement('button')
    preview.type = 'button'
    preview.className = 'attachment-preview'
    preview.dataset.previewUrl = attachment.url
    const image = document.createElement('img')
    if (deferImages) image.dataset.deferredSrc = attachment.url
    else image.src = attachment.url
    image.className = 'message-image'
    image.alt = attachment.filename || sourceLabel(attachment.source_kind)
    image.loading = 'lazy'
    image.dataset.artifactId = attachment.artifact_id
    preview.appendChild(image)
    figure.appendChild(preview)
    const caption = document.createElement('figcaption')
    const label = document.createElement('span')
    label.textContent = sourceLabel(attachment.source_kind)
    caption.appendChild(label)
    figure.appendChild(caption)
    gallery.appendChild(figure)
  }
  return gallery
}

function appendRemainingAttachments(document: Document, consumed: Set<string>) {
  const remaining = props.attachments.filter(item => !consumed.has(item.artifact_id))
  const searchImages = remaining.filter(item => item.source_kind === 'web_search')
  const otherImages = remaining.filter(item => item.source_kind !== 'web_search')
  const root = document.querySelector('#markdown-root')
  if (!root) return

  if (searchImages.length) {
    const disclosure = document.createElement('details')
    disclosure.className = 'search-attachment-disclosure'
    const summary = document.createElement('summary')
    summary.textContent = `网页搜索 · ${searchImages.length} 张图片`
    const body = document.createElement('div')
    body.className = 'search-attachment-disclosure-body'
    body.appendChild(buildAttachmentGallery(document, searchImages, true))
    disclosure.append(summary, body)
    root.appendChild(disclosure)
  }
  if (props.showUnreferencedAttachments && otherImages.length) {
    root.appendChild(buildAttachmentGallery(document, otherImages))
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
  const consumed = new Set<string>()
  localizeInlineImages(document, consumed)
  appendRemainingAttachments(document, consumed)
  return DOMPurify.sanitize(document.querySelector('#markdown-root')?.innerHTML || sanitized, {
    USE_PROFILES: { html: true },
    ADD_TAGS: ['details', 'summary'],
    ADD_ATTR: ['target', 'rel', 'data-preview-url', 'data-deferred-src', 'data-artifact-id'],
    FORBID_TAGS: ['style', 'iframe', 'object', 'embed', 'form'],
    FORBID_ATTR: ['style'],
  })
})

function previewImage(event: MouseEvent) {
  const target = event.target as HTMLElement
  const disclosure = target.closest<HTMLDetailsElement>('.search-attachment-disclosure')
  if (target.closest('summary') && disclosure) {
    for (const image of Array.from(disclosure.querySelectorAll<HTMLImageElement>('img[data-deferred-src]'))) {
      const deferredSrc = image.dataset.deferredSrc
      if (deferredSrc && safeImageUrl(deferredSrc)) image.src = deferredSrc
      delete image.dataset.deferredSrc
    }
    return
  }
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
