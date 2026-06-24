from s3_streamer import S3Streamer

def main():
    streamer = S3Streamer()
    print("Testing connection to Hetzner S3...")
    
    # Test listing books
    print("\n--- Books in archive/local_books_import/ ---")
    books = list(streamer.list_files("archive/local_books_import/"))
    print(f"Found {len(books)} books. First 5:")
    for b in books[:5]:
        print(" -", b)
        
    # Test listing voice transcripts
    print("\n--- Voice files in archive/local_voice_import/ ---")
    voices = list(streamer.list_files("archive/local_voice_import/"))
    print(f"Found {len(voices)} voice files. First 5:")
    for v in voices[:5]:
        print(" -", v)

if __name__ == "__main__":
    main()
