interface TranscriptPanelProps {
  fileName: string | null
}

export function TranscriptPanel({ fileName }: TranscriptPanelProps) {
  return (
    <div className="panel-body transcript-body">
      <p className="panel-subtitle">Markdown transcript</p>
      <div className="transcript-block">
        <p className="transcript-title">{fileName ?? 'No document loaded'}</p>
        <pre>
{`# Transcript preview

- OCR output will appear here.
- Sections align with the current PDF page.
- Highlights will reference the same spans.`}
        </pre>
      </div>
    </div>
  )
}
