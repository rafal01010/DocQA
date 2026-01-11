# Document RAG App - High-Level Plan

## Goals (from your requirements)
- Upload a PDF/document, run OCR (dots.ocr), convert to Markdown, embed, and store in LanceDB.
- Monochrome UI with three panes:
  - Left: document viewer (~50% width).
  - Below left: highlights/sections panel (same width, shorter height).
  - Right: chat/transcript panel (~25% width, full height of left column).
- Transcript tab shows the Markdown for the currently visible document section.
- Highlights tab can switch to "sections" (TOC-based) if a TOC exists.
- Prioritize readability and clean code; avoid over-abstracted frameworks.

## Recommended Stack (minimal, readable)

### Frontend
- **Framework**: React + Vite + TypeScript
  - Rationale: light setup, readable file structure, large ecosystem, no heavy "meta" abstractions.
- **PDF viewer**:
  - `react-pdf` (wraps `pdf.js`) for quick integration; use a custom viewer if you need advanced scrolling/section sync later.
- **Markdown rendering**:
  - `react-markdown` + `remark-gfm` for transcript rendering.
- **Styling**:
  - Plain CSS Modules or vanilla CSS + CSS variables for monochrome theme (cleaner than utility sprawl).
- **State management**:
  - Local React state + a small store like `zustand` if cross-pane syncing grows complex.

### Backend
- **Framework**: FastAPI
  - Rationale: clear, typed, minimal ceremony, easy to read.
- **Processing**:
  - Synchronous pipeline to keep the flow simple while the core features land.
- **OCR**:
  - `dots.ocr` (as you decided).
- **Markdown conversion**:
  - Keep conversion in your pipeline; if needed, `pymupdf` or `pdfplumber` can help with layout metadata.
- **Embeddings**:
  - `sentence-transformers` (consistent with your HF experience) or a small HF model.
- **Vector DB**:
  - LanceDB (already chosen).

## Avoiding Over-Abstraction
- **Skip LangChain**: You already prefer this, and it keeps code paths explicit.
- **LangGraph**: likely unnecessary now. It's best when you have a complex, branching workflow with many tools/agents.
- **DSPy**: useful if you want systematic prompt optimization for:
  - Highlight generation quality
  - Question answering consistency
  - Section summarization
  Use DSPy for these discrete steps but keep it optional and isolated, so your core pipeline stays readable.

## Pipeline Overview (minimal + clear)
1. **Upload** -> store raw PDF (disk or S3).
2. **OCR** -> dots.ocr -> raw text.
3. **Markdown build** -> structured Markdown + section headings (extract TOC when possible).
4. **Chunking** -> split by headings + size limits.
5. **Embedding** -> HF model (batch).
6. **Index** -> LanceDB.
7. **Highlights generation** -> LLM prompt (DSPy-optimized optional).
8. **Chat** -> retrieve top-k chunks -> prompt -> response (stream if desired).

## UI Flow (what to build)
- **Left column**: PDF viewer with page scroll + active page tracking.
- **Bottom left**: highlights list (toggle to "sections" if TOC exists).
- **Right column**: tabbed panel
  - Transcript tab: render Markdown for active page/section.
  - Chat tab: question input + streaming response + citations.

## Suggested Project Structure
- `frontend/`
  - `src/components/` (DocumentViewer, HighlightsPanel, RightPanel, TopBar)
  - `src/` (App layout, styling, upload flow)
- `backend/`
  - `app/` (FastAPI app entry)
  - `data/uploads/` (stored documents)
  - `pipeline/` (OCR -> Markdown -> chunking -> embeddings -> LanceDB)
  - `rag/` (retrieval + prompt assembly)

## Next Decisions (after the skeleton)
1. Define page-to-section mapping and how OCR chunks track PDF page numbers.
2. Choose the embedding model and chunking strategy for LanceDB indexing.
3. Implement the OCR -> Markdown pipeline and persist section metadata.
4. Add transcript rendering + highlight linking to the PDF viewer.

If you want, I can proceed with the OCR/Markdown pipeline and page-to-section syncing.
