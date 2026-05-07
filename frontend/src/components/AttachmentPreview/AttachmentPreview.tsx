import type { AttachmentMeta } from '../../store/types';

interface AttachmentPreviewProps {
  attachments: AttachmentMeta[];
  onRemove: (id: string) => void;
  isVisionModel: boolean;
}

export default function AttachmentPreview({ attachments, onRemove, isVisionModel }: AttachmentPreviewProps) {
  if (attachments.length === 0) return null;

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, padding: '8px 0' }}>
      {attachments.map((att) => (
        <div key={att.id} className="chip" style={{ position: 'relative' }}>
          {att.type === 'image' ? (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
              <circle cx="8.5" cy="8.5" r="1.5" />
              <polyline points="21 15 16 10 5 21" />
            </svg>
          ) : (
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ flexShrink: 0 }}>
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          )}
          <span style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {att.name}
          </span>
          <span
            className="chip-remove"
            onClick={() => onRemove(att.id)}
          >
            ×
          </span>
        </div>
      ))}
      {!isVisionModel && attachments.some((a) => a.type === 'image') && (
        <span style={{ fontSize: 11, color: 'var(--warning)', display: 'flex', alignItems: 'center' }}>
          Model may not support vision
        </span>
      )}
    </div>
  );
}
