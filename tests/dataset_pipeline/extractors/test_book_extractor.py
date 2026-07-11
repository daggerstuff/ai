from unittest.mock import MagicMock, patch

from dataset_pipeline.extractors.book_extractor import BookExtractor


@patch("dataset_pipeline.extractors.book_extractor.fitz")
def test_extract_pdf_handles_exceptions(mock_fitz):
    # Mock fitz.open to raise an Exception
    mock_fitz.open.side_effect = Exception("Mocked PDF extraction error")

    # Create BookExtractor with a dummy streamer
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    # Call the method
    result = extractor.extract_pdf("dummy/path.pdf")

    # Verify the result is an empty string as per the error handling block logic
    assert result == ""

@patch("dataset_pipeline.extractors.book_extractor.fitz")
def test_extract_pdf_success(mock_fitz):
    # Mock fitz.open to return a mock document with mocked pages
    mock_doc = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 text"
    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 text"

    # Configure the document to act as a context manager and an iterable
    mock_doc.__enter__.return_value = [mock_page1, mock_page2]
    mock_fitz.open.return_value = mock_doc

    # Create BookExtractor with a dummy streamer
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    # Call the method
    result = extractor.extract_pdf("dummy/path.pdf")

    # Verify the result contains the extracted text from all pages
    assert result == "Page 1 text\nPage 2 text\n"

@patch("dataset_pipeline.extractors.book_extractor.epub")
@patch("dataset_pipeline.extractors.book_extractor.BeautifulSoup")
def test_extract_epub_success(mock_bs, mock_epub):
    # Mock epub.read_epub and BeautifulSoup
    mock_book = MagicMock()
    mock_item = MagicMock()
    mock_item.get_type.return_value = 9 # ebooklib.ITEM_DOCUMENT
    mock_item.get_body_content.return_value = b"<html><body>Page 1 text</body></html>"
    mock_book.get_items.return_value = [mock_item]
    mock_epub.read_epub.return_value = mock_book

    mock_soup = MagicMock()
    mock_soup.get_text.return_value = "Page 1 text"
    mock_bs.return_value = mock_soup

    # Create BookExtractor with a dummy streamer
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    # Call the method
    result = extractor.extract_epub("dummy/path.epub")

    # Verify the result contains the extracted text
    assert result == "Page 1 text\n"

@patch("dataset_pipeline.extractors.book_extractor.epub")
def test_extract_epub_handles_exceptions(mock_epub):
    # Mock epub.read_epub to raise an Exception
    mock_epub.read_epub.side_effect = Exception("Mocked EPUB extraction error")

    # Create BookExtractor with a dummy streamer
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    # Call the method
    result = extractor.extract_epub("dummy/path.epub")

    # Verify the result is an empty string
    assert result == ""

def test_chunk_text():
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    text = "word1 word2 word3 word4 word5 word6"
    chunks = list(extractor.chunk_text(text, chunk_size=3))

    assert chunks == ["word1 word2 word3", "word4 word5 word6"]

@patch("dataset_pipeline.extractors.book_extractor.os.unlink")
@patch("dataset_pipeline.extractors.book_extractor.os.path.exists")
def test_extract_all_success(mock_exists, mock_unlink):
    mock_streamer = MagicMock()
    mock_streamer.list_files.return_value = ["archive/local_books_import/book1.pdf", "archive/local_books_import/book2.epub", "archive/local_books_import/not_a_book.txt"]

    extractor = BookExtractor(mock_streamer)

    # Mock the extract_pdf and extract_epub methods
    with patch.object(extractor, "extract_pdf", return_value="Short PDF text."), \
         patch.object(extractor, "extract_epub", return_value="This is a very long epub text that should definitely be chunked because it is long enough."):

         # Mock chunk_text to control exactly what is yielded
         with patch.object(extractor, "chunk_text") as mock_chunk_text:
            # Short text will be skipped if length < 50, long text will be yielded
            # But here we just mock the chunk_text output directly
            def side_effect_chunk_text(text, **kwargs):
                if "PDF" in text:
                    yield "Short PDF text."
                else:
                    yield "This is a very long epub text that should definitely be chunked because it is long enough."

            mock_chunk_text.side_effect = side_effect_chunk_text

            mock_exists.return_value = True

            results = list(extractor.extract_all())

            # Check results
            assert len(results) == 1 # PDF should be skipped because len < 50

            assert results[0]["raw_data"]["text"] == "This is a very long epub text that should definitely be chunked because it is long enough."
            assert results[0]["metadata"]["book_name"] == "book2.epub"
            assert results[0]["metadata"]["chunk_index"] == 0

            # Check download calls
            assert mock_streamer.download_to_file.call_count == 2

            # Check unlink calls
            assert mock_unlink.call_count == 2

def test_extract_all_missing_dependencies():
    mock_streamer = MagicMock()
    extractor = BookExtractor(mock_streamer)

    # Simulate missing dependencies
    extractor.has_pdf = False

    # The generator should yield nothing (empty)
    results = list(extractor.extract_all())
    assert results == []
