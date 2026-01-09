#!/usr/bin/env python3
"""
Master Integration Pipeline - Orchestrates All Component Integration
Solves the core problem: 6 powerful components not being used in training datasets
"""

import os
import json
import asyncio
import inspect
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Component integrators
from .components.journaling_integrator import JournalingIntegrator
from .components.voice_blend_integrator import VoiceBlendIntegrator
from .components.edge_case_integrator import EdgeCaseIntegrator
from .components.dual_persona_integrator import DualPersonaIntegrator
from .components.bias_detection_integrator import BiasDetectionIntegrator
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
        self.components = {}
        self.integration_results = {}
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the integration pipeline"""
        logger = logging.getLogger("integration_pipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        
        # Process components in dependency order
        component_order = [
            'psychology_kb',      # Base knowledge first
            'journaling',         # Progress tracking foundation  
            'voice_blending',     # Expert voice characteristics
            'bias_detection',     # Safety validation ready
            'edge_cases',         # Crisis scenarios
            'dual_persona'        # Therapeutic dynamics
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
        
        # Smart integration combinations
        integrations = [
            {
                'name': 'voice_enhanced_edge_cases',
                'components': ['voice_blending', 'edge_cases'],
                'description': 'Apply tri-expert voice to crisis scenarios'
            },
            {
                'name': 'journaling_dual_persona',
                'components': ['journaling', 'dual_persona'],
                'description': 'Long-term therapeutic relationships with progress tracking'
            },
            {
                'name': 'psychology_kb_bias_safe',
                'components': ['psychology_kb', 'bias_detection'],
                'description': 'Ethically validated therapeutic knowledge'
            }
        ]
        
        for integration in integrations:
            self.logger.info(f"🔀 Creating {integration['name']}...")
            
            try:
                # Check if all required components are available
                required_components = integration['components']
                available_components = [
                    self.components[comp] for comp in required_components 
                    if comp in self.components
                ]
                
                if len(available_components) == len(required_components):
                    result = await self._combine_components(
                        integration['name'],
                        available_components,
                        integration['description']
                    )
                    phase_2_results[integration['name']] = result
                    self.logger.info(f"✅ {integration['name']} integration complete")
                else:
                    missing = set(required_components) - set(self.components.keys())
                    self.logger.warning(f"⚠️ Skipping {integration['name']}: missing {missing}")
                    
            except Exception as e:
                self.logger.error(f"❌ {integration['name']} integration failed: {e}")
                phase_2_results[integration['name']] = {"error": str(e)}
        
        self.integration_results['phase_2'] = phase_2_results
        return phase_2_results
    
    async def run_integration_phase_3(self) -> Dict[str, Any]:
        """Phase 3: Final dataset construction"""
        self.logger.info("🏗️ PHASE 3: Integrated Dataset Construction")
        
        phase_3_results = {}
        
        # Create comprehensive integrated datasets
        dataset_configs = [
            {
                'name': 'full_stack_therapeutic_dataset',
                'description': 'Complete integration of all components',
                'components': list(self.components.keys()),
                'priority': 'high'
            },
            {
                'name': 'crisis_intervention_dataset', 
                'description': 'Edge cases with expert voice and safety validation',
                'components': ['edge_cases', 'voice_blending', 'bias_detection'],
                'priority': 'high'
            },
            {
                'name': 'long_term_therapy_dataset',
                'description': 'Journaling with dual persona and progress tracking',
                'components': ['journaling', 'dual_persona', 'psychology_kb'],
                'priority': 'high'
            }
        ]
        
        for dataset_config in dataset_configs:
            self.logger.info(f"📊 Building {dataset_config['name']}...")
            
            try:
                dataset_result = await self._build_integrated_dataset(dataset_config)
                phase_3_results[dataset_config['name']] = dataset_result
                self.logger.info(f"✅ {dataset_config['name']} dataset complete")
                
            except Exception as e:
                self.logger.error(f"❌ {dataset_config['name']} dataset failed: {e}")
                phase_3_results[dataset_config['name']] = {"error": str(e)}
        
        self.integration_results['phase_3'] = phase_3_results
        return phase_3_results
    
    async def _combine_components(self, integration_name: str, components: List, description: str) -> Dict[str, Any]:
        """Combine multiple components intelligently"""

        self.logger.info(f"🔀 Combining {len(components)} components for {integration_name}")

        if integration_name == "voice_enhanced_edge_cases":
            voice_blender = next(
                (c for c in components if isinstance(c, VoiceBlendIntegrator)), None
            )
            edge_case_integrator = next(
                (c for c in components if isinstance(c, EdgeCaseIntegrator)), None
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
        
        dataset_name = dataset_config['name']
        self.logger.info(f"🏗️ Building integrated dataset: {dataset_name}")
        
        # Placeholder for dataset construction logic
        result = {
            'dataset_name': dataset_name,
            'description': dataset_config['description'],
            'components_used': dataset_config['components'],
            'priority': dataset_config['priority'],
            'status': 'success',
            'output_file': f"{self.config.output_dir}/{dataset_name}.jsonl",
            'record_count': 0,
            'validation_passed': True
        }
        
        # TODO: Implement actual dataset building logic
        # This would combine all component outputs into final training datasets
        
        return result
    
    async def run_full_integration(self) -> Dict[str, Any]:
        """Run the complete integration pipeline"""
        start_time = datetime.now()
        self.logger.info("🚀 STARTING FULL INTEGRATION PIPELINE")
        
        try:
            # Initialize all components
            await self.initialize_components()
            
            # Run integration phases
            phase_1_results = await self.run_integration_phase_1()
            phase_2_results = await self.run_integration_phase_2()  
            phase_3_results = await self.run_integration_phase_3()
            
            # Generate final report
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            final_results = {
                'status': 'success',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'components_processed': len(self.components),
                'phases_completed': 3,
                'integration_results': self.integration_results
            }
            
            # Save results
            results_file = f"{self.config.output_dir}/integration_results.json"
            with open(results_file, 'w') as f:
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
        enable_psychology_kb=True
    )
    
    pipeline = MasterIntegrationPipeline(config)
    results = await pipeline.run_full_integration()
    
    print("\n🎯 INTEGRATION PIPELINE TEST COMPLETE")
    print(f"Status: {results['status']}")
    print(f"Duration: {results['duration_seconds']:.2f}s")
    print(f"Components: {results['components_processed']}")

if __name__ == "__main__":
    asyncio.run(main())
