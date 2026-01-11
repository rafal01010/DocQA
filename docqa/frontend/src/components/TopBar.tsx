import type { ChangeEvent } from 'react'

interface TopBarProps {
  onFileSelected: (file: File) => void
  isUploading: boolean
  fileName?: string | null
  uploadError?: string | null
}

export function TopBar({
  onFileSelected,
  isUploading,
  fileName,
  uploadError,
}: TopBarProps) {
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      onFileSelected(file)
    }
    event.target.value = ''
  }

  return (
    <header className="top-bar">
      <div className="branding">
        <span className="app-name">DocQA</span>
        <span className="app-tag">Document RAG workspace</span>
      </div>
      <div className="upload-group">
        <label className="button ghost">
          <input
            className="file-input"
            type="file"
            accept=".pdf,.doc,.docx,.txt,.md"
            onChange={handleChange}
            disabled={isUploading}
          />
          {isUploading ? 'Uploading...' : 'Upload PDF / Doc'}
        </label>
        {fileName ? <span className="file-name">{fileName}</span> : null}
        {uploadError ? <span className="error-text">{uploadError}</span> : null}
      </div>
    </header>
  )
}
