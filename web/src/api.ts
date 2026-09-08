export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    credentials: 'same-origin',
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers || {}),
    },
    ...init,
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail))
  }
  return response.json()
}

export interface Conversation {
  id: string; channel: 'web' | 'forum' | 'api'; title: string; updated_at: string
  external_id: string; bot_id: string; persona_id: string; title_custom: boolean; context_epoch?: number
}
export type AttachmentSource = 'user_upload' | 'forum_post' | 'forum_search' | 'web_search' | 'generated'
export interface Attachment {
  artifact_id: string; url: string; mime_type: string; filename?: string
  width?: number; height?: number; source_kind: AttachmentSource; source_url?: string
}
export interface Message {
  id: string; role: string; content: string; status: string; run_id?: string
  attachments: Attachment[]; created_at: string; epoch: number
}
export interface RunEvent { id: number; run_id: string; type: string; payload: Record<string, unknown>; created_at: string }
export interface ConversationDetail { conversation: Conversation; messages: Message[]; events: RunEvent[] }
