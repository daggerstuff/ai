<!-- markdownlint-disable -->\n\n# PIX-499: YouTube Transcript Acquisition Pipeline PRD

## HR Eng

| PIX-499: YouTube Transcript Acquisition Pipeline |  | [Summary: Build the "producer" for the YouTube training pipeline to fetch, clean, and organize therapeutic transcripts from specified playlists.] |
| :---- | :---- | :---- |
| **Author**: Pickle Rick **Contributors**: Morty **Intended audience**: Engineering, PM | **Status**: Draft **Created**: 2026-05-06 | **Self Link**: [PIX-499](https://linear.app/pixelated/issue/PIX-499/85-implement-youtube-transcript-fetcher-yt-dlp-acquisition-step) **Context**: Training Pipeline Improvements |

## Introduction

Pixelated requires a steady stream of therapeutic conversation data for training. While a consumer (`youtube_ingestion.py`) exists to process transcripts into training pairs, the upstream acquisition step (fetching transcripts from YouTube) is currently missing or manual. This PRD defines the requirements for a production-ready fetcher.

## Problem Statement

**Current Process:** Transcripts are either missing, manually downloaded, or exist as stubs/garbage in various directories.
**Primary Users:** AI Training Engineers.
**Pain Points:** Broken pipeline (no producer), manual labor for data acquisition, inconsistent transcript quality, lack of speaker information (diarization).
**Importance:** Critical for Phase 2/3 training runs. Without raw transcripts, the model cannot learn from high-quality therapeutic sources like Tim Fletcher and others.

## Objective & Scope

**Objective:** Implement a robust, automated pipeline to fetch English transcripts from YouTube playlists, clean them while preserving speaker information, and track metadata (durations, channel stats).
**Ideal Outcome:** A single command can sync the GDrive playlist, fetch all missing transcripts, and prepare a verified manifest for the ingestion pipeline.

### In-scope or Goals
- **GDrive Sync**: Use `rclone` to fetch `youtube_playlists.txt`.
- **Transcript Extraction**: Use `yt-dlp` to download subtitles (English only).
- **Diarization**: Preserve/extract speaker markers (e.g., `Speaker A:`, `<v Speaker Name>`).
- **Metadata**: Track duration totals and file counts in a per-batch `manifest.json`.
- **Cleaning**: Strip timestamps, HTML, and music markers while maintaining structural integrity.
- **S3-Aware**: Design the output structure to be compatible with future `rclone sync` to Hetzner S3.

### Not-in-scope or Non-Goals
- **Non-English Support**: English only for this phase.
- **API Keys**: Use `yt-dlp` to avoid YouTube Data API quotas.
- **Audio Download**: Strictly transcript-based acquisition (`--skip-download`).

## Product Requirements

### Critical User Journeys (CUJs)
1. **Full Acquisition Sync**: Engineer runs `transcript_fetcher.py --sync-gdrive`. The script pulls the latest playlist, identifies new videos, fetches transcripts, and updates the manifest.
2. **Quality Verification**: Engineer checks `manifest.json` for "Duration Totals" to estimate the scale of the new data batch.

### Functional Requirements

| Priority | Requirement | User Story |
| :---- | :---- | :---- |
| P0 | `rclone` GDrive Integration | As an engineer, I want to fetch the playlist from GDrive automatically so I don't have to manage local files. |
| P0 | `yt-dlp` Extraction | As an engineer, I want to extract transcripts without needing an API key. |
| P0 | Diarization Preservation | As an engineer, I want the transcripts to include speaker IDs so the model learns conversation structure. |
| P1 | Duration Tracking | As an engineer, I want to see the total hours of content fetched so I can balance my datasets. |
| P1 | S3 Compatibility | As an engineer, I want the output structure to be ready for S3 synchronization. |

## Assumptions

- `yt-dlp` can access subtitles for most therapeutic content (Tim Fletcher videos typically have manual or auto-subs).
- Diarization is present in some form in the VTT/SRT or can be inferred.

## Risks & Mitigations

- **Risk**: Rate limiting by YouTube -> **Mitigation**: Implement graceful `rate_limit` delays and retries.
- **Risk**: Subtitles unavailable -> **Mitigation**: Log missing subtitles in manifest `errors` and move on.
- **Risk**: No diarization in auto-subs -> **Mitigation**: Use basic heuristic markers or accept raw text as fallback.

## Tradeoff

- **Option**: Use Whisper for local diarization vs. `yt-dlp` transcripts.
- **Decision**: Use `yt-dlp` transcripts.
- **Why**: Much faster and cheaper. Audio download + Whisper is overkill for a "fetcher" step unless transcript quality is zero.

## Business Benefits/Impact/Metrics

**Success Metrics:**

| Metric | Current State (Benchmark) | Future State (Target) | Savings/Impacts |
| :---- | :---- | :---- | :---- |
| *Hours of Content* | ~10h (manual) | 500h+ | High training diversity |
| *Acquisition Time* | Days (manual) | < 1h (automated) | Faster iteration cycles |

## Stakeholders / Owners

| Name | Team/Org | Role | Note |
| :---- | :---- | :---- | :---- |
| Chad | AI Training | Lead | Approver |
| Pickle Rick | Engineering | Compiler | Executor |
