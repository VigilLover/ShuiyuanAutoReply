/** Discourse-flavoured BBCode and upload:// handling for rendered replies. */

export const FORUM_ORIGIN = 'https://shuiyuan.sjtu.edu.cn'
export const FORUM_HOSTS = new Set(['shuiyuan.sjtu.edu.cn'])

const UPLOAD_RE = /upload:\/\/([A-Za-z0-9][A-Za-z0-9._-]*)/g

/** upload://token.ext short URLs become the forum's authenticated short-url path. */
export function resolveUploadUrls(text: string): string {
  return text.replace(UPLOAD_RE, (_match, token: string) => `${FORUM_ORIGIN}/uploads/short-url/${token}`)
}

function escapeAttribute(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function safeHref(value: string): string | null {
  const trimmed = value.trim()
  if (/^https?:\/\//i.test(trimmed)) return trimmed
  if (trimmed.startsWith('upload://')) return resolveUploadUrls(trimmed)
  return null
}

/**
 * Convert the BBCode subset the bot and forum actually use into HTML that
 * `marked` leaves alone and DOMPurify accepts:
 * [details=title]…[/details], [grid]…[/grid], [quote="user, post:N, topic:T"]…[/quote],
 * [img]…[/img], [url=…]…[/url], [spoiler]…[/spoiler].
 * Markdown inside the blocks is still parsed afterwards.
 */
export function bbcodeToHtml(text: string): string {
  let out = text
  out = out.replace(/\[details(?:=("?)([^\]"]*)\1)?\]([\s\S]*?)\[\/details\]/gi, (_m, _q, title: string, body: string) => {
    const summary = escapeAttribute((title || '详情').trim())
    return `\n\n<details class="bbcode-details"><summary>${summary}</summary>\n\n${body.trim()}\n\n</details>\n\n`
  })
  out = out.replace(/\[grid\]([\s\S]*?)\[\/grid\]/gi, (_m, body: string) => `\n\n<div class="bbcode-grid">\n\n${body.trim()}\n\n</div>\n\n`)
  out = out.replace(/\[quote(?:=("?)([^\]"]*)\1)?\]([\s\S]*?)\[\/quote\]/gi, (_m, _q, meta: string, body: string) => {
    const parts = (meta || '').split(',').map(part => part.trim()).filter(Boolean)
    const user = parts[0] || ''
    const post = parts.find(part => part.startsWith('post:'))?.slice(5)
    const topic = parts.find(part => part.startsWith('topic:'))?.slice(6)
    let source = user ? `@${user}` : ''
    if (post && topic) source += ` · <a href="${FORUM_ORIGIN}/t/${escapeAttribute(topic)}/${escapeAttribute(post)}" target="_blank" rel="noopener noreferrer">#${escapeAttribute(post)}</a>`
    const cite = source ? `<cite>${source}</cite>\n\n` : ''
    return `\n\n<blockquote class="bbcode-quote">${cite}${body.trim()}\n\n</blockquote>\n\n`
  })
  out = out.replace(/\[img\]([\s\S]*?)\[\/img\]/gi, (_m, src: string) => {
    const href = safeHref(src)
    return href ? `![](${href})` : ''
  })
  out = out.replace(/\[url=("?)([^\]"]*)\1\]([\s\S]*?)\[\/url\]/gi, (_m, _q, href: string, label: string) => {
    const safe = safeHref(href)
    return safe ? `[${label.trim() || safe}](${safe})` : label
  })
  out = out.replace(/\[url\]([\s\S]*?)\[\/url\]/gi, (_m, href: string) => {
    const safe = safeHref(href)
    return safe ? `[${safe}](${safe})` : href
  })
  out = out.replace(/\[spoiler\]([\s\S]*?)\[\/spoiler\]/gi, (_m, body: string) => `<span class="bbcode-spoiler">${body.trim()}</span>`)
  return out
}

/** Full pre-processing pipeline applied before Markdown parsing. */
export function prepareReplyMarkdown(text: string): string {
  return resolveUploadUrls(bbcodeToHtml(text || ''))
}
