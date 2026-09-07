"""Test script to verify data loading fixes"""
import sys
from pathlib import Path
import pandas as pd

# Add papercode to path
sys.path.append(str(Path(__file__).parent))

from papercode.datasets import CamelsTXT

# Test parameters
camels_root = Path("/data0/funato/hsp2/2_MERV/varssim_US")
basin = "001"  # Test with first basin
dates = [pd.to_datetime('01101999', format='%d%m%Y'), 
         pd.to_datetime('01111999', format='%d%m%Y')]  # Short date range for testing

print(f"Testing data loading for basin {basin}...")
print(f"camels_root: {camels_root}")
print(f"dates: {dates}")

try:
    # Create dataset
    ds = CamelsTXT(
        camels_root=camels_root,
        basin=basin,
        dates=dates,
        is_train=False,
        seq_length=270,
        with_attributes=False
    )
    
    print(f"\n✓ Successfully created dataset!")
    print(f"  Number of samples: {len(ds)}")
    print(f"  Period start: {ds.period_start}")
    print(f"  Period end: {ds.period_end}")
    
    # Test loading one sample
    if len(ds) > 0:
        x, y = ds[0]
        print(f"  Sample shape - X: {x.shape}, Y: {y.shape}")
        print(f"  X features (should be 3: Precip, PET, Temp): {x.shape[-1]}")
        print("\n✓ All tests passed!")
    else:
        print("\n⚠ Warning: No samples in dataset")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
