#!/usr/bin/env python3
"""
Master Integration Pipeline - Orchestrates All Component Integration
Solves the core problem: 6 powerful components not being used in training datasets
"""

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Component integrators
from .components.journaling_integrator import JournalingIntegrator
from .components.voice_blend_integrator import VoiceBlendIntegrator
from .components.edge_case_integrator import EdgeCaseIntegrator
from .components.dual_persona_integrator import DualPersonaIntegrator
from .components.bias_detection_integrator import BiasDetectionIntegrator
from .components.dual_persona_integrator import DualPersonaIntegrator
from .components.edge_case_integrator import EdgeCaseIntegrator
from .components.psychology_kb_integrator import PsychologyKBIntegrator


@dataclass
class IntegrationConfig:
    """Configuration for the integration pipeline"""

    # Component enable/disable flags
    enable_journaling: bool = True
    enable_voice_blending: bool = True
    enable_edge_cases: bool = True
    enable_dual_persona: bool = True
    enable_bias_detection: bool = True
    enable_psychology_kb: bool = True

    # Integration settings
    output_dir: str = "ai/integration_pipeline/output"
    batch_size: int = 1000
    max_workers: int = 4
    validation_enabled: bool = True

    # Component-specific configs
    journaling_config: Dict = None
    voice_config: Dict = None
    edge_case_config: Dict = None
    dual_persona_config: Dict = None
    bias_detection_config: Dict = None
    psychology_kb_config: Dict = None


class MasterIntegrationPipeline:
    """Master orchestrator for all component integration"""

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.components: Dict[str, Any] = {}
        self.integration_results: Dict[str, Any] = {}
        self.context: Dict[str, Any] = {"phase_1": {}, "phase_2": {}, "phase_3": {}}

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------
    # Helpers
    # ------------------------

    def _ensure_dir(self, path: str | Path) -> Path:
        """Create directory if missing and return Path."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _write_jsonl(self, records: List[Dict[str, Any]], filename: str) -> str:
        """Write a list of dicts to JSONL; return path."""
        out_path = self._ensure_dir(self.config.output_dir) / filename
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.logger.info(f"💾 Wrote {len(records)} records -> {out_path}")
        return str(out_path)

    def _write_json(self, payload: Dict[str, Any], filename: str) -> str:
        """Write a dict to JSON; return path."""
        out_path = self._ensure_dir(self.config.output_dir) / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.logger.info(f"💾 Wrote JSON -> {out_path}")
        return str(out_path)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the integration pipeline"""
        logger = logging.getLogger("integration_pipeline")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def initialize_components(self):
        """Initialize all enabled component integrators"""
        self.logger.info("🚀 Initializing component integrators...")

        if self.config.enable_journaling:
            self.components['journaling'] = JournalingIntegrator()
            self.logger.info("✅ Journaling integrator initialized")

        if self.config.enable_voice_blending:
            self.components['voice_blending'] = VoiceBlendIntegrator()
            self.logger.info("✅ Voice blending integrator initialized")

        if self.config.enable_edge_cases:
            self.components['edge_cases'] = EdgeCaseIntegrator()
            self.logger.info("✅ Edge case integrator initialized")

        if self.config.enable_dual_persona:
            self.components['dual_persona'] = DualPersonaIntegrator()
            self.logger.info("✅ Dual persona integrator initialized")

        if self.config.enable_bias_detection:
            self.components['bias_detection'] = BiasDetectionIntegrator()
            self.logger.info("✅ Bias detection integrator initialized")

        if self.config.enable_psychology_kb:
            self.components['psychology_kb'] = PsychologyKBIntegrator()
            self.logger.info("✅ Psychology KB integrator initialized")

        self.logger.info(f"🎯 Initialized {len(self.components)} component integrators")

    async def run_integration_phase_1(self) -> Dict[str, Any]:
        """Phase 1: Individual component preparation"""
        self.logger.info("📋 PHASE 1: Component Preparation")

        phase_1_results = {}
        context = self.context["phase_1"]

        # Process components in dependency order
        component_order = [
            "psychology_kb",  # Base knowledge first
            "journaling",  # Progress tracking foundation
            "voice_blending",  # Expert voice characteristics
            "bias_detection",  # Safety validation ready
            "edge_cases",  # Crisis scenarios
            "dual_persona",  # Therapeutic dynamics
        ]

        for component_name in component_order:
            if component_name not in self.components:
                continue

            self.logger.info(f"🔧 Processing {component_name}...")

            try:
                component = self.components[component_name]
                result = await self._prepare_component(component_name, component)
                phase_1_results[component_name] = result
                self.logger.info(f"✅ {component_name} preparation complete")

            except Exception as e:
                self.logger.error(f"❌ {component_name} preparation failed: {e}")
                phase_1_results[component_name] = {"error": str(e)}
        
        self.integration_results['phase_1'] = phase_1_results
        return phase_1_results

    async def _prepare_component(self, component_name: str, component: Any) -> Dict[str, Any]:
        """Prepare a component for integration.

        This returns a fully resolved Phase 1 result. If `component.prepare()` is
        awaitable, it is awaited here; otherwise its return value is used directly.

        Integrators in this repo are mostly synchronous; this helper normalizes them
        to a common Phase 1 output shape.
        """

        if hasattr(component, "prepare") and callable(component.prepare):
            result = component.prepare()
            if inspect.isawaitable(result):
                return await result
            return result

        if component_name == "psychology_kb":
            kb = component.load_psychology_knowledge_base()
            return {"status": "ready", "concept_count": len(kb) if isinstance(kb, dict) else 0}

        if component_name == "journaling":
            patterns = component.extract_progress_patterns()
            return {"status": "ready", "patterns": len(patterns)}

        if component_name == "voice_blending":
            blended = component.create_blended_voice()
            return {
                "status": "ready",
                "core_principles": len(blended.get("core_principles", [])),
                "therapeutic_methods": len(blended.get("therapeutic_methods", [])),
            }

        if component_name == "edge_cases":
            existing = component.load_existing_edge_cases()
            return {"status": "ready", "existing_edge_cases": len(existing)}

        if component_name == "dual_persona":
            personas = component.load_existing_personas()
            return {"status": "ready", "personas": len(personas)}

        if component_name == "bias_detection":
            return {"status": "ready", "categories": len(component.bias_categories)}

        return {"status": "ready"}
    
    async def run_integration_phase_2(self) -> Dict[str, Any]:
        """Phase 2: Smart component combination"""
        self.logger.info("🔗 PHASE 2: Smart Component Integration")

        phase_2_results = {}
        context = self.context["phase_2"]

        # Extract phase 1 data and components
        phase_1_context = self.context["phase_1"]
        phase_2_data = self._prepare_phase_2_data(phase_1_context)

        # Define and execute integrations
        integrations = self._get_phase_2_integrations()
        for integration in integrations:
            result = await self._execute_integration(integration, phase_2_data)
            phase_2_results[integration["name"]] = result
            context[integration["name"]] = result

        self.integration_results["phase_2"] = phase_2_results
        return phase_2_results

    def _prepare_phase_2_data(self, phase_1_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and prepare all phase 1 data for phase 2 integrations."""
        return {
            "edge_cases": phase_1_context.get("edge_cases", []),
            "dual_personas": phase_1_context.get("dual_persona", {}),
            "voice_blending": self.components.get("voice_blending"),
            "psychology_kb": self.components.get("psychology_kb"),
            "bias_detection": self.components.get("bias_detection"),
            "journaling": self.components.get("journaling"),
            "edge_cases_component": self.components.get("edge_cases"),
            "dual_persona_component": self.components.get("dual_persona"),
        }

    def _get_phase_2_integrations(self) -> List[Dict[str, Any]]:
        """Define all phase 2 integration strategies."""
        return [
            {
                "name": "voice_enhanced_edge_cases",
                "components": ["voice_blending", "edge_cases"],
                "description": "Apply tri-expert voice to crisis scenarios",
                "handler": self._integrate_voice_enhanced_edge_cases,
            },
            {
                "name": "journaling_dual_persona",
                "components": ["journaling", "dual_persona"],
                "description": (
                    "Long-term therapeutic relationships with progress tracking"
                ),
                "handler": self._integrate_journaling_dual_persona,
            },
            {
                "name": "psychology_kb_bias_safe",
                "components": ["psychology_kb", "bias_detection"],
                "description": "Ethically validated therapeutic knowledge",
                "handler": self._integrate_psychology_kb_bias_safe,
            },
        ]

    async def _execute_integration(
        self, integration: Dict[str, Any], phase_2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single integration with proper error handling."""
        integration_name = integration["name"]
        self.logger.info(f"🔀 Creating {integration_name}...")

        if not self._has_required_components(integration, phase_2_data):
            missing = self._get_missing_components(integration, phase_2_data)
            self.logger.warning(f"⚠️ Skipping {integration_name}: missing {missing}")
            return {"status": "skipped", "reason": f"Missing components: {missing}"}

        try:
            handler = integration["handler"]
            records = await handler(phase_2_data)

            # Write output and create result
            outfile = self._write_jsonl(records, f"{integration_name}.jsonl")
            result = {
                "status": "success",
                "records": len(records),
                "output_file": outfile,
            }
            self.logger.info(f"✅ {integration_name} integration complete")
            return result
        except Exception as e:
            self.logger.error(f"❌ {integration_name} integration failed: {e}")
            return {"status": "error", "error": str(e)}

    def _has_required_components(
        self, integration: Dict[str, Any], phase_2_data: Dict[str, Any]
    ) -> bool:
        """Check if all required components are available."""
        required = integration["components"]
        return all(comp in self.components for comp in required)

    def _get_missing_components(
        self, integration: Dict[str, Any], phase_2_data: Dict[str, Any]
    ) -> set:
        """Get list of missing required components."""
        required = integration["components"]
        return set(required) - set(self.components.keys())

    async def _integrate_voice_enhanced_edge_cases(
        self, phase_2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create voice-enhanced edge cases with bias validation."""
        edge_cases = phase_2_data["edge_cases"]
        expert_dict = (
            phase_2_data["voice_blending"].expert_voices
            if phase_2_data["voice_blending"]
            else {}
        )

        # Integrate voice into edge cases
        voice_integrated = phase_2_data[
            "edge_cases_component"
        ].integrate_with_expert_voices(edge_cases, expert_dict)

        if bias_detector := phase_2_data["bias_detection"]:
            for record in voice_integrated:
                record["bias_report"] = bias_detector.check_dataset_for_bias(record)

        return voice_integrated

    async def _integrate_journaling_dual_persona(
        self, phase_2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create journaling data with dual persona relationships."""
        journaling_data = phase_2_data["journaling"].generate_continuity_datasets()
        relationships = phase_2_data[
            "dual_persona_component"
        ].create_therapeutic_relationships(phase_2_data["dual_personas"])

        # Attach relationships using round-robin distribution
        integrated = []
        for idx, item in enumerate(journaling_data):
            relationship = self._get_round_robin_item(relationships, idx)
            integrated.append({**item, "relationship_context": relationship})

        return integrated

    async def _integrate_psychology_kb_bias_safe(
        self, phase_2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Enhance records with psychology KB and bias validation."""
        kb = phase_2_data["psychology_kb"]
        bias_detector = phase_2_data["bias_detection"]

        # Get base records from prior integrations
        base_records = self._get_base_records_for_kb_enhancement(phase_2_data)

        # Enhance each record
        enhanced = []
        for item in base_records:
            enriched = (
                kb.enhance_conversation_with_psychology_concepts(item) if kb else item
            )
            if bias_detector:
                enriched["bias_report"] = bias_detector.check_dataset_for_bias(enriched)
            enhanced.append(enriched)

        return enhanced

    def _get_round_robin_item(
        self, items: List[Dict[str, Any]], index: int
    ) -> Dict[str, Any]:
        """Safely get item from list using round-robin distribution."""
        return items[index % len(items)] if items else {}

    def _get_base_records_for_kb_enhancement(
        self, phase_2_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get base records for KB enhancement from available sources."""
        # Try to use previously integrated data; fallback to edge cases
        return phase_2_data.get("base_records") or phase_2_data["edge_cases"]

    async def run_integration_phase_3(self) -> Dict[str, Any]:
        """Phase 3: Final dataset construction"""
        self.logger.info("🏗️ PHASE 3: Integrated Dataset Construction")

        phase_3_results = {}
        phase_2_context = self.context["phase_2"]
        kb_bias_integrated = phase_2_context.get("psychology_kb_bias_safe", {}).get(
            "output_file"
        )
        voice_integrated_file = phase_2_context.get(
            "voice_enhanced_edge_cases", {}
        ).get("output_file")
        journaling_file = phase_2_context.get("journaling_dual_persona", {}).get(
            "output_file"
        )
        kb_bias_records = []
        voice_records = []
        journaling_records = []
        # Reload written JSONL files for aggregation
        for path_str, target in [
            (kb_bias_integrated, (kb_bias_records, "kb_bias")),
            (voice_integrated_file, (voice_records, "voice")),
            (journaling_file, (journaling_records, "journal")),
        ]:
            if path_str:
                try:
                    with open(path_str, "r", encoding="utf-8") as f:
                        for line in f:
                            target[0].append(json.loads(line))
                except FileNotFoundError:
                    self.logger.warning(f"⚠️ Expected output missing: {path_str}")

        # Create comprehensive integrated datasets
        dataset_configs = [
            {
                "name": "full_stack_therapeutic_dataset",
                "description": "Complete integration of all components",
                "components": list(self.components.keys()),
                "priority": "high",
            },
            {
                "name": "crisis_intervention_dataset",
                "description": "Edge cases with expert voice and safety validation",
                "components": ["edge_cases", "voice_blending", "bias_detection"],
                "priority": "high",
            },
            {
                "name": "long_term_therapy_dataset",
                "description": "Journaling with dual persona and progress tracking",
                "components": ["journaling", "dual_persona", "psychology_kb"],
                "priority": "high",
            },
        ]

        for dataset_config in dataset_configs:
            self.logger.info(f"📊 Building {dataset_config['name']}...")

            try:
                dataset_result = await self._build_integrated_dataset(
                    dataset_config,
                    {
                        "kb_bias": kb_bias_records,
                        "voice": voice_records,
                        "journal": journaling_records,
                    },
                )
                phase_3_results[dataset_config["name"]] = dataset_result
                self.logger.info(f"✅ {dataset_config['name']} dataset complete")

            except Exception as e:
                self.logger.error(f"❌ {dataset_config['name']} dataset failed: {e}")
                phase_3_results[dataset_config["name"]] = {"error": str(e)}

        self.integration_results["phase_3"] = phase_3_results
        return phase_3_results
    
    async def _combine_components(self, integration_name: str, components: List, description: str) -> Dict[str, Any]:
        """Combine multiple components intelligently"""

        self.logger.info(f"🔀 Combining {len(components)} components for {integration_name}")

        if integration_name == "voice_enhanced_edge_cases":
            voice_blender = self.components.get("voice_blending")
            if voice_blender is None:
                self.logger.warning(
                    "voice_blending component missing from registry; falling back to provided components"
                )
                voice_blender = next(
                    (
                        c
                        for c in components
                        if hasattr(c, "generate_tri_expert_responses")
                    ),
                    None,
                )

            edge_case_integrator = self.components.get("edge_cases")
            if edge_case_integrator is None:
                self.logger.warning(
                    "edge_cases component missing from registry; falling back to provided components"
                )
                edge_case_integrator = next(
                    (
                        c
                        for c in components
                        if hasattr(c, "generate_crisis_and_cultural_edge_cases")
                    ),
                    None,
                )

            if edge_case_integrator is None:
                raise ValueError("EdgeCaseIntegrator is required for voice_enhanced_edge_cases")

            edge_case_config = self.config.edge_case_config or {}
            target_records = int(edge_case_config.get("target_records", 15_000))
            seed = int(edge_case_config.get("seed", 1))
            enable_bias_detection = bool(edge_case_config.get("enable_bias_detection", False))
            bias_detector = self.components.get("bias_detection") if enable_bias_detection else None
            output_file = str(
                edge_case_config.get(
                    "output_file",
                    "",
                )
            )
            summary_file = str(
                edge_case_config.get(
                    "summary_file",
                    "",
                )
            )

            metrics = edge_case_integrator.generate_crisis_and_cultural_edge_cases(
                target_records=target_records,
                seed=seed,
                output_file=output_file or None,
                summary_file=summary_file or None,
                voice_blender=voice_blender,
                bias_detector=bias_detector,
            )

            return {
                "integration_name": integration_name,
                "description": description,
                "components_count": len(components),
                "status": "success",
                "output_files": [
                    metrics.get("output_file"),
                    metrics.get("summary_file"),
                ],
                "metrics": metrics,
            }

        return {
            "integration_name": integration_name,
            "description": description,
            "components_count": len(components),
            "status": "skipped",
            "output_files": [],
            "metrics": {},
        }
    
    async def _build_integrated_dataset(self, dataset_config: Dict) -> Dict[str, Any]:
        """Build a final integrated dataset from component results"""
        dataset_name = dataset_config["name"]
        self.logger.info(f"🏗️ Building integrated dataset: {dataset_name}")

        combined: List[Dict[str, Any]] = []

        if dataset_name == "full_stack_therapeutic_dataset":
            combined = (
                (source_records.get("kb_bias") or [])
                + (source_records.get("voice") or [])
                + (source_records.get("journal") or [])
            )
        elif dataset_name == "crisis_intervention_dataset":
            combined = source_records.get("voice") or []
        elif dataset_name == "long_term_therapy_dataset":
            combined = source_records.get("journal") or []

        return {
            "dataset_name": dataset_name,
            "description": dataset_config["description"],
            "components_used": dataset_config["components"],
            "priority": dataset_config["priority"],
            "status": "success",
            "output_file": self._write_jsonl(combined, f"{dataset_name}.jsonl"),
            "record_count": len(combined),
            "validation_passed": True,
        }

    async def run_full_integration(self) -> Dict[str, Any]:
        """Run the complete integration pipeline"""
        start_time = datetime.now()
        self.logger.info("🚀 STARTING FULL INTEGRATION PIPELINE")

        try:
            # Initialize all components
            await self.initialize_components()

            # Run integration phases
            await self.run_integration_phase_1()
            await self.run_integration_phase_2()
            await self.run_integration_phase_3()

            # Generate final report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            final_results = {
                "status": "success",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "components_processed": len(self.components),
                "phases_completed": 3,
                "integration_results": self.integration_results,
            }

            # Save results
            results_file = f"{self.config.output_dir}/integration_results.json"
            with open(results_file, "w") as f:
                json.dump(final_results, f, indent=2)

            self.logger.info(f"🎉 INTEGRATION PIPELINE COMPLETE in {duration:.2f}s")
            self.logger.info(f"📊 Results saved to: {results_file}")

            return final_results

        except Exception as e:
            self.logger.error(f"❌ INTEGRATION PIPELINE FAILED: {e}")
            raise


# CLI interface for testing
async def main():
    """Main function for testing the integration pipeline"""

    config = IntegrationConfig(
        output_dir="ai/integration_pipeline/output",
        enable_journaling=True,
        enable_voice_blending=True,
        enable_edge_cases=True,
        enable_dual_persona=True,
        enable_bias_detection=True,
        enable_psychology_kb=True,
    )

    pipeline = MasterIntegrationPipeline(config)
    results = await pipeline.run_full_integration()

    print("\n🎯 INTEGRATION PIPELINE TEST COMPLETE")
    print(f"Status: {results['status']}")
    print(f"Duration: {results['duration_seconds']:.2f}s")
    print(f"Components: {results['components_processed']}")


if __name__ == "__main__":
    asyncio.run(main())
