interface DocumentViewerProps {
  fileUrl: string | null
  fileName: string | null
}

export function DocumentViewer({ fileUrl, fileName }: DocumentViewerProps) {
  const isPdf = fileName?.toLowerCase().endsWith('.pdf')

  return (
    <section className="panel viewer-panel">
      <header className="panel-header">
        <div>
          <p className="panel-title">Document</p>
          <p className="panel-subtitle">Synced with your highlights</p>
        </div>
        <span className="pill">Page 1</span>
      </header>
      <div className="viewer-viewport">
        {!fileUrl ? (
          <div className="empty-state">
            <p className="empty-title">Upload a document to begin</p>
            <p className="empty-copy">
              PDF previews render here. DOC/DOCX files will show a text preview later.
            </p>
          </div>
        ) : !isPdf ? (
          <div className="empty-state">
            <p className="empty-title">Preview not available yet</p>
            <p className="empty-copy">
              We will render non-PDF formats after OCR and Markdown conversion.
            </p>
          </div>
        ) : (
          <object className="viewer-object" data={fileUrl} type="application/pdf">
            <div className="empty-state">
              <p className="empty-title">PDF preview unavailable</p>
              <p className="empty-copy">
                Your browser does not support inline PDFs. Use the download link instead.
              </p>
            </div>
          </object>
        )}
      </div>
    </section>
  )
}
