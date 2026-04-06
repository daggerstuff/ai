    def start(self):
        """Start the resume orchestrator"""
        self.resume_engine.start()
        self.orchestrator_active = True
        
        # Store a reference to the loop for thread-safe operations (Review suggestion)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        # Start orchestration thread
        self.orchestrator_thread = threading.Thread(
            target=self._orchestration_loop, daemon=True
        )
        self.orchestrator_thread.start()
        logger.info("Resume orchestrator started")

    def stop(self):
        """Stop the resume orchestrator"""
        self.orchestrator_active = False
        
        if self.orchestrator_thread:
            # Join thread with timeout to allow graceful exit
            self.orchestrator_thread.join(timeout=10)

        self.resume_engine.stop()
        logger.info("Resume orchestrator stopped")
