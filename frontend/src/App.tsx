import { useState } from 'react'
import { DocumentViewer } from './components/DocumentViewer'
import { HighlightsPanel } from './components/HighlightsPanel'
import { RightPanel } from './components/RightPanel'
import { TopBar } from './components/TopBar'
import './App.css'

function App() {
  const [uploadedFile, setUploadedFile] = useState<{
    url: string
    name: string
  } | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [highlightsMode, setHighlightsMode] = useState<'highlights' | 'sections'>(
    'highlights'
  )
  const [activeTab, setActiveTab] = useState<'chat' | 'transcript'>('chat')

  const handleFileSelected = async (file: File) => {
    setIsUploading(true)
    setUploadError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Upload failed')
      }

      const data = (await response.json()) as { url: string; filename: string }
      setUploadedFile({ url: data.url, name: data.filename })
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : 'Upload failed. Try again.'
      )
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="app-shell">
      <TopBar
        onFileSelected={handleFileSelected}
        isUploading={isUploading}
        fileName={uploadedFile?.name ?? null}
        uploadError={uploadError}
      />
      <main className="layout">
        <div className="left-stack">
          <DocumentViewer
            fileUrl={uploadedFile?.url ?? null}
            fileName={uploadedFile?.name ?? null}
          />
          <HighlightsPanel mode={highlightsMode} onModeChange={setHighlightsMode} />
        </div>
        <RightPanel
          activeTab={activeTab}
          onTabChange={setActiveTab}
          fileName={uploadedFile?.name ?? null}
          isReady={Boolean(uploadedFile)}
        />
      </main>
    </div>
  )
}

export default App
