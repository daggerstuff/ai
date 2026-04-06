    def _orchestration_loop(self):
        """Main orchestration loop running in background thread"""

        while self.orchestrator_active:
            try:
                # Process resume queue
                if (
                    self.resume_queue
                    and len(self.active_resumes) < self.max_concurrent_resumes
                ):
                    self._dispatch_next_resume()

                # Adaptive sleep based on queue state
                sleep_time = 0.5 if self.resume_queue else 2.0
                time.sleep(sleep_time)

            except asyncio.CancelledError:
                # Handle cancellation explicitly to allow responsive shutdown (Review suggestion)
                logger.info("Orchestration loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                time.sleep(5)  # Backoff on error
