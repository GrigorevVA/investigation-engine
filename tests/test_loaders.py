from pathlib import Path

import docx
import pytest
from fpdf import FPDF

from app.ingest.loaders import UnsupportedFormatError, load_file, make_doc_id


def test_txt(tmp_path: Path):
    path = tmp_path / "Report 2030.txt"
    path.write_text("First paragraph.\r\n\r\n\r\n\r\nSecond paragraph.   \n", encoding="utf-8")
    doc = load_file(path)
    assert doc.doc_id == "file::report-2030"
    assert doc.fmt == "txt"
    assert doc.text == "First paragraph.\n\nSecond paragraph."
    assert doc.title == "Report 2030"


def test_markdown_title_from_heading(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Board notes\n\nText.", encoding="utf-8")
    assert load_file(path).title == "Board notes"


def test_pdf(tmp_path: Path):
    path = tmp_path / "spec.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_title("Atlas spec")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 6, "Payload is 12 kg.")
    pdf.output(str(path))
    doc = load_file(path)
    assert "Payload is 12 kg." in doc.text
    assert doc.title == "Atlas spec"
    assert doc.metadata["pages"] == 1


def test_docx(tmp_path: Path):
    path = tmp_path / "interview.docx"
    d = docx.Document()
    d.core_properties.title = "CEO interview"
    d.add_paragraph("Revenue was 52 million.")
    d.save(str(path))
    doc = load_file(path)
    assert doc.text == "Revenue was 52 million."
    assert doc.title == "CEO interview"


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("a,b")
    with pytest.raises(UnsupportedFormatError):
        load_file(path)


def test_empty_file_is_rejected(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n")
    with pytest.raises(ValueError):
        load_file(path)


def test_original_name_wins_over_temp_name(tmp_path: Path):
    path = tmp_path / "tmp123.txt"
    path.write_text("x y z")
    assert load_file(path, original_name="Real Name.txt").doc_id == make_doc_id("Real Name.txt")
