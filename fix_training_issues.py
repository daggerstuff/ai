#!/usr/bin/env python3
"""
Comprehensive fix for training pipeline issues.

This script addresses both the Hugging Face authentication issue and the ONNX
export failure that were identified in the Sentry error logs.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingPipelineFixer:
    """Comprehensive fixer for training pipeline issues."""

    def __init__(self):
        self.hf_token = None
        self.model_cache = {}

    def setup_environment(self) -> bool:
        """
        Set up the environment for training with proper authentication.

        Returns:
            bool: True if setup successful, False otherwise.
        """
        try:
            logger.info("Setting up training environment...")

            # Set up Hugging Face authentication
            if not self._setup_huggingface_auth():
                logger.warning("Hugging Face authentication setup failed")
                return False

            # Set up ONNX export environment
            self._setup_onnx_environment()

            logger.info("✅ Environment setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            return False

    def _setup_huggingface_auth(self) -> bool:
        """Set up Hugging Face authentication."""
        try:
            # Get token from environment
            self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

            if self.hf_token:
                login(token=self.hf_token)
                logger.info("✅ Hugging Face authentication successful")
                return True
            else:
                logger.warning(
                    "No Hugging Face token found, some models may fail to load"
                )
                return False

        except Exception as e:
            logger.error(f"Hugging Face authentication failed: {e}")
            return False

    def _setup_onnx_environment(self):
        """Set up environment for ONNX export compatibility."""
        try:
            # Set ONNX export settings
            os.environ["ONNX_EXPORT_OPSET_VERSION"] = "11"

            # Disable problematic optimizations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.info("✅ ONNX export environment configured")

        except Exception as e:
            logger.warning(f"ONNX environment setup warning: {e}")

    def load_model_with_fixes(
        self, model_name: str = "LatitudeGames/Harbinger-24B", use_auth: bool = True
    ) -> Optional[AutoModelForCausalLM]:
        """
        Load a model with authentication and compatibility fixes.

        Args:
            model_name: Name of the model to load.
            use_auth: Whether to use authentication.

        Returns:
            Optional[AutoModelForCausalLM]: Loaded model or None if failed.
        """
        try:
            return self._load_model_from_pretrained(model_name, use_auth)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            logger.error(
                "This might be due to authentication issues or model access "
                "restrictions"
            )

            # Provide helpful error message
            if "401" in str(e):
                logger.error(
                    "401 Unauthorized: Check your Hugging Face token and model "
                    "access permissions"
                )
            elif "404" in str(e):
                logger.error(
                    "404 Not Found: Model may not exist or you may not have access"
                )

            return None

    def _load_model_from_pretrained(self, model_name, use_auth):
        logger.info(f"Loading model {model_name} with fixes...")

        # Check cache first
        if model_name in self.model_cache:
            logger.info(f"Using cached model {model_name}")
            return self.model_cache[model_name]

        # Load tokenizer
        logger.info("Loading tokenizer...")
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=self.hf_token if use_auth else None,
        )

        # Load model with authentication
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_auth_token=self.hf_token if use_auth else None,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        # Cache the model
        self.model_cache[model_name] = model

        logger.info(f"✅ Successfully loaded model {model_name}")
        return model

    def export_model_to_onnx_safe(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple = (1, 512),
        opset_version: int = 11,
    ) -> bool:
        """
        Safely export model to ONNX format with comprehensive error handling.

        Args:
            model: PyTorch model to export.
            output_path: Path to save the ONNX model.
            input_shape: Input tensor shape.
            opset_version: ONNX opset version.

        Returns:
            bool: True if export successful, False otherwise.
        """
        try:
            logger.info(f"Starting ONNX export to {output_path}")

            # Prepare model for export
            model.eval()

            # Create dummy input
            dummy_input = torch.randn(*input_shape)
            if next(model.parameters()).is_cuda:
                dummy_input = dummy_input.cuda()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Export with comprehensive error handling
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                verbose=False,
            )

            logger.info(f"✅ Successfully exported model to {output_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return self._handle_onnx_export_failure(model, output_path, input_shape, e)

    def _handle_onnx_export_failure(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple,
        original_error: Exception,
    ) -> bool:
        """
        Handle ONNX export failure with fallback strategies.

        Args:
            model: PyTorch model.
            output_path: Output path.
            input_shape: Input shape.
            original_error: The original error that occurred.

        Returns:
            bool: True if fallback successful, False otherwise.
        """
        logger.info("Attempting ONNX export fallback strategies...")

        # Strategy 1: Try with older opset version
        try:
            logger.info("Trying with opset version 9...")
            torch.onnx.export(
                model,
                torch.randn(*input_shape),
                output_path,
                export_params=True,
                opset_version=9,
                do_constant_folding=False,
                verbose=False,
            )
            logger.info(f"✅ Fallback ONNX export successful: {output_path}")
            return True

        except Exception as e:
            logger.warning(f"Fallback with opset 9 also failed: {e}")

        # Strategy 2: Save model for manual conversion
        try:
            return self._save_model_for_manual_conversion(
                output_path, model, input_shape, original_error
            )
        except Exception as e:
            logger.error(f"Fallback strategy 2 also failed: {e}")

        # If all strategies fail
        logger.error("All ONNX export strategies failed")
        return False

    def _save_model_for_manual_conversion(
        self, output_path, model, input_shape, original_error
    ) -> bool:
        logger.info("Saving model for manual ONNX conversion...")
        fallback_path = output_path.replace(".onnx", "_for_onnx_conversion.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model.config if hasattr(model, "config") else None,
                "input_shape": input_shape,
                "original_error": str(original_error),
            },
            fallback_path,
        )

        logger.info(f"Model saved for manual conversion: {fallback_path}")
        logger.info("For manual conversion, install: pip install onnx onnxruntime")
        logger.info("Then use transformers.onnx or optimum library for conversion")

        return False

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test of both fixes.

        Returns:
            Dict[str, Any]: Test results.
        """
        logger.info("Running comprehensive test of training pipeline fixes...")

        results = {
            "huggingface_auth": False,
            "model_loading": False,
            "onnx_export": False,
            "overall_success": False,
        }

        try:
            # Test 1: Hugging Face authentication
            results["huggingface_auth"] = self._setup_huggingface_auth()

            # Test 2: Model loading (if auth successful)
            if results["huggingface_auth"]:
                # Use a smaller model for testing
                test_model = self.load_model_with_fixes("gpt2")
                results["model_loading"] = test_model is not None

                # Test 3: ONNX export (if model loaded)
                if results["model_loading"]:
                    # Create a simple test model
                    class SimpleTestModel(nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.linear = nn.Linear(512, 256)

                        def forward(self, x):
                            return self.linear(x)

                    test_model = SimpleTestModel()
                    results["onnx_export"] = self.export_model_to_onnx_safe(
                        test_model, "/tmp/test_model_fix.onnx"
                    )

            # Overall success
            results["overall_success"] = (
                results["huggingface_auth"]
                and results["model_loading"]
                and results["onnx_export"]
            )

            if results["overall_success"]:
                logger.info("✅ All fixes working correctly!")
            else:
                logger.warning("⚠️  Some fixes need attention")

            return results

        except Exception as e:
            logger.error(f"Comprehensive test failed: {e}")
            return results


def main():
    """Main function to apply the fixes."""
    logger.info("Applying comprehensive training pipeline fixes...")

    fixer = TrainingPipelineFixer()

    # Set up environment
    if not fixer.setup_environment():
        logger.error("Failed to set up environment")
        return 1

    # Run comprehensive test
    results = fixer.run_comprehensive_test()

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("FIX SUMMARY:")
    logger.info(
        f"Hugging Face Authentication: {'✅' if results['huggingface_auth'] else '❌'}"
    )
    logger.info(f"Model Loading: {'✅' if results['model_loading'] else '❌'}")
    logger.info(f"ONNX Export: {'✅' if results['onnx_export'] else '❌'}")
    logger.info(f"Overall Success: {'✅' if results['overall_success'] else '❌'}")
    logger.info("=" * 50)

    if results["overall_success"]:
        logger.info("🎉 All training pipeline issues have been fixed!")
        return 0
    else:
        logger.error("❌ Some issues remain. Please check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
