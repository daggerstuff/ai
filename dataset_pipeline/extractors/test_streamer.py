from s3_streamer import S3Streamer


def main():
    streamer = S3Streamer()

    # Test listing books
    books = list(streamer.list_files("archive/local_books_import/"))
    for _b in books[:5]:
        pass

    # Test listing voice transcripts
    voices = list(streamer.list_files("archive/local_voice_import/"))
    for _v in voices[:5]:
        pass

if __name__ == "__main__":
    main()
