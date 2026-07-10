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
