interface ChatPanelProps {
  isReady: boolean
}

const starterMessages = [
  {
    role: 'assistant',
    text: 'Ask anything about the document. Responses will cite the matching pages.',
  },
]

export function ChatPanel({ isReady }: ChatPanelProps) {
  return (
    <div className="panel-body chat-body">
      <div className="chat-thread">
        {starterMessages.map((message, index) => (
          <div className="chat-message" key={`${message.role}-${index}`}>
            <span className="chat-role">{message.role}</span>
            <p className="chat-text">{message.text}</p>
          </div>
        ))}
        {!isReady ? (
          <div className="chat-message muted">
            <span className="chat-role">system</span>
            <p className="chat-text">Upload a document to enable retrieval.</p>
          </div>
        ) : null}
      </div>
      <div className="chat-input">
        <input
          placeholder="Ask a question about this document..."
          disabled={!isReady}
        />
        <button className="button" type="button" disabled={!isReady}>
          Send
        </button>
      </div>
    </div>
  )
}
