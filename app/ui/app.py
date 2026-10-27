"""Gradio interface: investigate, manage documents, inspect raw search."""
from collections.abc import Callable
from pathlib import Path

import gradio as gr

from app.engine import Engine
from app.ingest.loaders import UnsupportedFormatError
from app.rag.answer import InvestigationResult
from app.rag.llm import LLMError
from app.search.vector_store import VectorStoreError

EXAMPLES = [
    "Which magazine was started first, Arthur's Magazine or First for Women?",
    "Were Scott Derrickson and Ed Wood of the same nationality?",
    "In what year was Lumora Robotics founded?",
    "What was Lumora Robotics' revenue in 2030?",
    "What was the revenue of Lumora Robotics in 2031?",
]
INTRO = """**What this can do.** Ask a question about the indexed documents (about 60,000 Wikipedia paragraphs
from HotpotQA plus your uploads). The engine splits it into searches, collects evidence, answers with
numbered sources, flags sources that contradict each other, and says so when the documents are not enough."""


def render_answer(result: InvestigationResult) -> tuple[str, str, str]:
    status = "⚠️ **Not enough evidence in the documents.**\n\n" if result.insufficient_evidence else ""
    answer = status + result.answer
    if result.contradictions:
        answer += "\n\n**Conflicting sources**\n" + "\n".join(
            f"- {c['description']} ({', '.join(c['citations'])})" for c in result.contradictions)
    if result.notes:
        answer += "\n\n<small>" + " ".join(result.notes) + "</small>"
    cited = {c.label for c in result.citations}
    sources = "\n\n".join(
        f"**[{c.label}] {c.title}**{' · cited' if c.label in cited else ''}  \n`{c.chunk_id}`  \n{c.snippet}…"
        for c in result.retrieved) or "No sources."
    plan = "\n".join(f"{i}. {q}" for i, q in enumerate(result.queries, 1))
    return answer, sources, plan


def build_ui(get_engine: Callable[[], Engine]) -> gr.Blocks:
    def investigate(question: str, mode: str):
        if not question.strip():
            return "Type a question or pick an example.", "", ""
        try:
            return render_answer(get_engine().investigate(question.strip(), mode))
        except (LLMError, VectorStoreError) as err:
            return f"❌ {err}", "", ""

    def upload(files):
        lines = []
        for path in files or []:
            name = Path(path).name
            try:
                r = get_engine().add_file(Path(path).read_bytes(), name)
                extra = f", same content as {r.duplicate_of}" if r.duplicate_of else ""
                lines.append(f"- **{name}**: {r.status}, {r.chunks} chunks{extra}")
            except (UnsupportedFormatError, ValueError) as err:
                lines.append(f"- **{name}**: rejected, {err}")
        return "\n".join(lines) or "No files selected.", documents_table()

    def documents_table():
        return [[d["doc_id"], d["title"], d["fmt"], d["chunks"]] for d in get_engine().list_documents()]

    def delete(doc_id: str):
        ok = get_engine().delete(doc_id.strip())
        return (f"Deleted {doc_id}." if ok else f"{doc_id} not found."), documents_table()

    def raw_search(query: str, mode: str, k: float):
        hits = get_engine().search(query, int(k), mode)
        return [[h.chunk.title, round(h.score, 4), h.chunk.text[:200]] for h in hits]

    with gr.Blocks(title="AI Investigation Engine") as ui:
        gr.Markdown("# AI Investigation Engine")
        with gr.Tab("Investigate"):
            gr.Markdown(INTRO)
            question = gr.Textbox(label="Question", lines=2, placeholder="Ask about people, films, places, companies…")
            gr.Examples(EXAMPLES, inputs=question, label="Try one of these")
            mode = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="Retrieval mode")
            run = gr.Button("Investigate", variant="primary")
            answer = gr.Markdown()
            with gr.Accordion("Search plan", open=False):
                plan = gr.Markdown()
            with gr.Accordion("Sources", open=True):
                sources = gr.Markdown()
            run.click(investigate, [question, mode], [answer, sources, plan])
            question.submit(investigate, [question, mode], [answer, sources, plan])
        with gr.Tab("Documents"):
            gr.Markdown("Upload TXT, MD, PDF or DOCX. Uploading a file with the same name updates it.")
            files = gr.File(file_count="multiple", file_types=[".txt", ".md", ".pdf", ".docx"], type="filepath")
            upload_btn = gr.Button("Upload")
            upload_log = gr.Markdown()
            table = gr.Dataframe(headers=["doc_id", "title", "format", "chunks"], interactive=False)
            with gr.Row():
                doc_id = gr.Textbox(label="doc_id to delete")
                delete_btn = gr.Button("Delete", variant="stop")
            delete_log = gr.Markdown()
            upload_btn.click(upload, files, [upload_log, table])
            delete_btn.click(delete, doc_id, [delete_log, table])
            ui.load(documents_table, None, table)
        with gr.Tab("Search debug"):
            q = gr.Textbox(label="Query")
            m = gr.Radio(["hybrid", "vector", "bm25"], value="hybrid", label="Mode")
            k = gr.Slider(1, 30, value=8, step=1, label="k")
            out = gr.Dataframe(headers=["title", "score", "text"], interactive=False)
            gr.Button("Search").click(raw_search, [q, m, k], out)
    return ui
