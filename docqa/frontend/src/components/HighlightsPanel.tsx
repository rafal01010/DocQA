interface HighlightsPanelProps {
  mode: 'highlights' | 'sections'
  onModeChange: (mode: 'highlights' | 'sections') => void
}

const highlightItems = [
  {
    title: 'Primary takeaway',
    detail: 'Summaries and key claims appear here once extraction runs.',
  },
  {
    title: 'Evidence trail',
    detail: 'Each highlight will cite the matching page span.',
  },
  {
    title: 'Actionable notes',
    detail: 'Turn insights into tasks or questions instantly.',
  },
]

const sectionItems = [
  {
    title: '1. Executive overview',
    detail: 'Detected from the document table of contents.',
  },
  {
    title: '2. Methodology',
    detail: 'Jump to page ranges with OCR-confirmed anchors.',
  },
  {
    title: '3. Findings',
    detail: 'Syncs with the transcript panel on the right.',
  },
]

export function HighlightsPanel({ mode, onModeChange }: HighlightsPanelProps) {
  const items = mode === 'highlights' ? highlightItems : sectionItems

  return (
    <section className="panel highlights-panel">
      <header className="panel-header">
        <div>
          <p className="panel-title">Highlights</p>
          <p className="panel-subtitle">Auto-generated summary stack</p>
        </div>
        <div className="segmented">
          <button
            className={mode === 'highlights' ? 'segment active' : 'segment'}
            onClick={() => onModeChange('highlights')}
            type="button"
          >
            Highlights
          </button>
          <button
            className={mode === 'sections' ? 'segment active' : 'segment'}
            onClick={() => onModeChange('sections')}
            type="button"
          >
            Sections
          </button>
        </div>
      </header>
      <div className="list">
        {items.map((item) => (
          <div className="list-item" key={item.title}>
            <p className="list-title">{item.title}</p>
            <p className="list-detail">{item.detail}</p>
          </div>
        ))}
      </div>
    </section>
  )
}
