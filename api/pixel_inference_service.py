# Module-level constants for security (Review suggestion)
DEFAULT_ERROR_DETAIL = "Internal server error"


@app.post("/infer", response_model=PixelInferenceResponse)
async def infer(request: PixelInferenceRequest, background_tasks: BackgroundTasks):
    """Generate response using Pixel model"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return await inference_engine.generate_response(request)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        # Mask internal exception details to prevent information exposure (Review suggestion)
        raise HTTPException(status_code=500, detail=DEFAULT_ERROR_DETAIL)


@app.post("/batch-infer")
async def batch_infer(requests: list[PixelInferenceRequest]):
    """Batch inference for multiple queries"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    responses = []
    for req in requests:
        try:
            response = await inference_engine.generate_response(req)
            responses.append(response)
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            # Include machine-readable error code alongside generic message (Review suggestion)
            responses.append({
                "error": DEFAULT_ERROR_DETAIL,
                "error_code": "INFERENCE_FAILED"
            })

    return responses


@app.post("/reload-model")
async def reload_model():
    """Reload model from disk"""
    try:
        inference_engine.model_loaded = False
        if inference_engine.load_model():
            return {"status": "success", "message": "Model reloaded"}
        raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Reload error: {e}")
        # Mask internal exception details
        raise HTTPException(status_code=500, detail=DEFAULT_ERROR_DETAIL)
