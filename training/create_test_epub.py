from ebooklib import epub


def create_test_epub(path):
    book = epub.EpubBook()
    book.set_identifier("test123456")
    book.set_title("Test Clinical Book")
    book.set_language("en")
    book.add_author("Test Author")

    # Add a chapter
    c1 = epub.EpubHtml(title="Introduction", file_name="intro.xhtml", lang="en")
    c1.content = "<html><body><h1>Introduction</h1><p>Cognitive Behavioral Therapy (CBT) is a form of psychological treatment that has been demonstrated to be effective for a range of problems including depression, anxiety disorders, alcohol and drug use problems, marital problems, eating disorders, and severe mental illness.</p></body></html>"
    book.add_item(c1)

    # Add default NCX and Nav
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Define TOC
    book.toc = (epub.Link("intro.xhtml", "Introduction", "intro"),)

    # Add spine
    book.spine = ["nav", c1]

    # Write EPUB
    epub.write_epub(path, book)

if __name__ == "__main__":
    create_test_epub("ai/training/test_data/books/test_clinical_book.epub")
