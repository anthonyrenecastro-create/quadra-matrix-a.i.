"""
Quick integration test for IFT field substrate.

Verifies that:
1. Interpreter initializes with IFT enabled
2. Field engine is created and accessible
3. Inference runs without errors
4. Field metrics are present in output
5. Adaptive parameters are computed
"""

import asyncio
import sys


async def test_integration():
    print("Testing IFT integration...")
    print("-" * 50)
    
    try:
        # Import
        from quadra.core.symbolic.interpreter import StatefulSymbolicPredictiveInterpreter
        print("✓ Import successful")
        
        # Initialize with IFT
        interpreter = StatefulSymbolicPredictiveInterpreter(
            model_version="integration-test-1.0",
            enable_ift=True,
            field_shape=(16, 16)
        )
        print("✓ Interpreter initialized with IFT")
        
        # Check field engine exists
        assert interpreter.oscillator.field_engine is not None, "Field engine not created"
        print("✓ Field engine created")
        
        # Check field shape
        assert interpreter.oscillator.field_engine.shape == (16, 16), "Wrong field shape"
        print("✓ Field shape correct: (16, 16)")
        
        # Run inference
        result = await interpreter.process({
            'text': 'test input',
            'concepts': ['test']
        }, request_id="integration-test-1")
        print("✓ Inference completed")
        
        # Check output structure
        assert 'ift_field_metrics' in result, "Missing IFT metrics in output"
        print("✓ IFT metrics present in output")
        
        ift_metrics = result['ift_field_metrics']
        assert ift_metrics['enabled'] == True, "IFT not marked as enabled"
        assert 'field_energy' in ift_metrics, "Missing field_energy"
        assert 'symmetry_order' in ift_metrics, "Missing symmetry_order"
        assert 'global_potential' in ift_metrics, "Missing global_potential"
        print(f"✓ Field energy: {ift_metrics['field_energy']:.4f}")
        print(f"✓ Symmetry order: {ift_metrics['symmetry_order']:.4f}")
        print(f"✓ Global potential: {ift_metrics['global_potential']:.4f}")
        
        # Check adaptive parameters
        assert 'adaptive_parameters' in result, "Missing adaptive parameters"
        params = result['adaptive_parameters']
        assert 'threshold' in params, "Missing threshold"
        assert 'leak' in params, "Missing leak"
        print(f"✓ Adaptive threshold: {params['threshold']:.4f}")
        print(f"✓ Adaptive leak: {params['leak']:.4f}")
        
        # Test baseline mode (IFT disabled)
        print("\nTesting baseline mode (IFT disabled)...")
        interpreter_baseline = StatefulSymbolicPredictiveInterpreter(
            model_version="baseline-test-1.0",
            enable_ift=False
        )
        print("✓ Baseline interpreter initialized")
        
        result_baseline = await interpreter_baseline.process({
            'text': 'test input',
            'concepts': ['test']
        }, request_id="baseline-test-1")
        print("✓ Baseline inference completed")
        
        assert result_baseline['ift_field_metrics']['enabled'] == False, "IFT should be disabled"
        print("✓ IFT correctly disabled in baseline mode")
        
        # Success!
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED")
        print("=" * 50)
        print("\nIFT field system integration verified.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
