import { ChatPanel } from './ChatPanel'
import { TranscriptPanel } from './TranscriptPanel'

interface RightPanelProps {
  activeTab: 'chat' | 'transcript'
  onTabChange: (tab: 'chat' | 'transcript') => void
  fileName: string | null
  isReady: boolean
}

export function RightPanel({
  activeTab,
  onTabChange,
  fileName,
  isReady,
}: RightPanelProps) {
  return (
    <section className="panel right-panel">
      <header className="panel-header tabs">
        <button
          className={activeTab === 'chat' ? 'tab active' : 'tab'}
          onClick={() => onTabChange('chat')}
          type="button"
        >
          Chat
        </button>
        <button
          className={activeTab === 'transcript' ? 'tab active' : 'tab'}
          onClick={() => onTabChange('transcript')}
          type="button"
        >
          Transcript
        </button>
      </header>
      {activeTab === 'chat' ? (
        <ChatPanel isReady={isReady} />
      ) : (
        <TranscriptPanel fileName={fileName} />
      )}
    </section>
  )
}
