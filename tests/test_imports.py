"""Smoke tests — verify core modules import and basic logic works without ML deps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_base_types():
    from pipeline.extractors.base import PersonRecord, ExtractionResult
    p = PersonRecord(barcode_number="123456789012", name="Test", user_id="USR-2024-0001", category="A")
    assert p.barcode_number == "123456789012"
    assert p.category == "A"


def test_confidence_routing():
    from pipeline.confidence import route, EASY_THRESHOLD
    assert route(0.9) == "easy"
    assert route(0.5) == "hard"
    assert route(EASY_THRESHOLD) == "easy"


def test_case_store_creates_db(tmp_path):
    import os
    os.environ["CASES_DB_PATH"] = str(tmp_path / "test.db")
    from pipeline.case_store import get_db, get_stats, insert_case
    db = get_db()
    assert db is not None
    stats = get_stats()
    assert isinstance(stats, dict)
    case_id = insert_case("img.jpg", None, "test_model", {"persons": []}, 0.6)
    assert len(case_id) > 0
    stats2 = get_stats()
    assert stats2.get("open", 0) >= 1


def test_barcode_reader_no_crash():
    from pipeline.barcode_reader import read_barcodes
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    result = read_barcodes(img)
    assert isinstance(result, list)


def test_layout_detector_fallback():
    from pipeline.layout_detector import detect_person_regions
    from PIL import Image
    img = Image.new("RGB", (400, 800), "white")
    regions = detect_person_regions(img)
    assert len(regions) >= 1
