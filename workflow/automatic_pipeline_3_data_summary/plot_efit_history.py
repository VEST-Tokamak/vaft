#!/usr/bin/env python3
"""
EFIT History Data Mining Script

This script mines data from all kfile and gfile pairs in the public directory,
extracting shot information, convergence status, and current analysis.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# OMFIT imports
from omfit_classes.omfit_eqdsk import OMFITkeqdsk, OMFITgeqdsk

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EFITHistoryMiner:
    """Class to mine EFIT history data from kfile and gfile pairs."""
    
    def __init__(self, public_dir: str = "/srv/vest.filedb/public"):
        self.public_dir = Path(public_dir)
        self.results = []
        
    def find_all_kfiles(self) -> List[Path]:
        """Find all kfile paths in the public directory."""
        kfiles = []
        for kfile_path in self.public_dir.rglob("kfile/k0*.00*"):
            if kfile_path.is_file():
                kfiles.append(kfile_path)
        return sorted(kfiles)
    
    def get_corresponding_gfile(self, kfile_path: Path) -> Optional[Path]:
        """Get the corresponding gfile path for a given kfile."""
        # Extract shot number and time from kfile name
        # kfile format: k0{shotnumber}.00{xxx}
        kfile_name = kfile_path.name
        match = re.match(r'k0(\d+)\.00(\d+)', kfile_name)
        if not match:
            return None
            
        shot_number = match.group(1)
        time_str = match.group(2)
        
        # Construct corresponding gfile path
        gfile_name = f"g0{shot_number}.00{time_str}"
        gfile_path = kfile_path.parent.parent / "gfile" / gfile_name
        
        return gfile_path if gfile_path.exists() else None
    
    def extract_shot_info(self, kfile_path: Path) -> Tuple[Optional[int], Optional[float]]:
        """Extract shot number and time from kfile path."""
        kfile_name = kfile_path.name
        match = re.match(r'k0(\d+)\.00(\d+)', kfile_name)
        if not match:
            return None, None
            
        shot_number = int(match.group(1))
        time_ms = float(match.group(2))  # Time in milliseconds
        
        return shot_number, time_ms
    
    def read_kfile_data(self, kfile_path: Path) -> Optional[Dict]:
        """Read and extract data from kfile."""
        plasma_current = None
        kfile_data = None
        try:
            k = OMFITkeqdsk(str(kfile_path))
            # Extract plasma current from IN1 section safely
            plasma_current = k['IN1']['PLASMA']
            kfile_data = k
        except (KeyError, TypeError) as e:
            print(f"Failed to read plasma current from kfile {kfile_path}: {e}")
            return None
        except Exception as e:
            print(f"Failed to read kfile {kfile_path}: {e}")
            return None
            
        return {
            'kfile_path': str(kfile_path),
            'plasma_current': plasma_current,
            'kfile_data': kfile_data
        }
    
    def read_gfile_data(self, gfile_path: Path) -> Optional[Dict]:
        """Read and extract data from gfile."""
        current = None
        gfile_data = None
        try:
            g = OMFITgeqdsk(str(gfile_path))
            current = g['CURRENT']
            gfile_data = g
        except (KeyError, TypeError) as e:
            print(f"Failed to read current from gfile {gfile_path}: {e}")
            return None
        except Exception as e:
            print(f"Failed to read gfile {gfile_path}: {e}")
            return None
            
        return {
            'gfile_path': str(gfile_path),
            'current': current,
            'gfile_data': gfile_data
        }
        
    def determine_phase(self, shot_data: List[Dict]) -> List[str]:
        """Determine ramp-up/ramp-down phase for each time point in a shot."""
        if len(shot_data) < 2:
            return ['unknown'] * len(shot_data)
        
        # Sort by time
        shot_data_sorted = sorted(shot_data, key=lambda x: x['time_ms'])
        
        phases = []
        for i, data in enumerate(shot_data_sorted):
            if i == 0:
                phases.append('unknown')  # First point
            else:
                prev_current = shot_data_sorted[i-1].get('kfile_plasma_current')
                curr_current = data.get('kfile_plasma_current')
                
                if prev_current is not None and curr_current is not None:
                    if curr_current > prev_current:
                        phases.append('ramp-up')
                    elif curr_current < prev_current:
                        phases.append('ramp-down')
                    else:
                        phases.append('flat')
                else:
                    phases.append('unknown')
        
        return phases
    
    def process_single_kfile(self, kfile_path: Path) -> Optional[Dict]:
        """Process a single kfile and its corresponding gfile."""
        logger.info(f"Processing {kfile_path}")
        
        # Extract shot info
        shot_number, time_ms = self.extract_shot_info(kfile_path)
        if shot_number is None or time_ms is None:
            logger.warning(f"Could not extract shot info from {kfile_path}")
            return None
        
        # Read kfile data
        kfile_data = self.read_kfile_data(kfile_path)
        if not kfile_data:
            return None
        
        # Check for corresponding gfile
        gfile_path = self.get_corresponding_gfile(kfile_path)
        gfile_exists = gfile_path is not None
        
        # Read gfile data if it exists
        gfile_data = None
        if gfile_exists:
            gfile_data = self.read_gfile_data(gfile_path)
        
        # Extract plasma current from kfile safely
        kfile_plasma_current = kfile_data.get('plasma_current') if kfile_data else None
        
        # Extract current from gfile safely
        gfile_current = gfile_data.get('current') if gfile_data else None
        
        # Calculate current error if both files exist
        current_error = None
        if kfile_plasma_current is not None and gfile_current is not None and gfile_current != 0:
            current_error = abs(kfile_plasma_current - gfile_current) / abs(gfile_current) * 100
        
        result = {
            'shot_number': shot_number,
            'time_ms': time_ms,
            'kfile_path': str(kfile_path),
            'gfile_path': str(gfile_path) if gfile_exists else None,
            'gfile_exists': gfile_exists,
            'kfile_plasma_current': kfile_plasma_current,
            'gfile_current': gfile_current,
            'current_relative_error_percent': current_error,
            'phase': 'unknown'  # Will be determined later per shot
        }
        
        return result
    
    def mine_all_data(self) -> pd.DataFrame:
        """Mine data from all kfiles and gfiles."""
        logger.info("Starting EFIT history data mining...")
        
        # Find all kfiles
        kfiles = self.find_all_kfiles()
        logger.info(f"Found {len(kfiles)} kfiles")
        
        # Process ALL kfiles for ALL shots
        test_kfiles = kfiles
        
        logger.info(f"Processing {len(test_kfiles)} kfiles from all shots")
        
        # Process each kfile
        for i, kfile_path in enumerate(test_kfiles):
            if i % 1000 == 0:
                logger.info(f"Processing {i}/{len(test_kfiles)} files...")
            
            result = self.process_single_kfile(kfile_path)
            if result:
                self.results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        if df.empty:
            logger.warning("No data was extracted!")
            return df
        
        # Determine phases for each shot
        logger.info("Determining ramp-up/ramp-down phases...")
        df['phase'] = 'unknown'
        
        for shot_num in df['shot_number'].unique():
            shot_mask = df['shot_number'] == shot_num
            shot_data = df[shot_mask].copy()
            
            if len(shot_data) > 1:
                phases = self.determine_phase(shot_data.to_dict('records'))
                df.loc[shot_mask, 'phase'] = phases
        
        logger.info(f"Data mining completed. Extracted {len(df)} records.")
        return df
    
    def save_results(self, df: pd.DataFrame, output_file: str = None):
        """Save results to CSV file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"efit_history_mining_{timestamp}.csv"
        
        output_path = Path(output_file).resolve()
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print summary statistics
        print("\n=== Data Mining Summary ===")
        print(f"Total records: {len(df)}")
        print(f"Unique shots: {df['shot_number'].nunique()}")
        print(f"Files with gfile convergence: {df['gfile_exists'].sum()}")
        print(f"Files without gfile convergence: {(~df['gfile_exists']).sum()}")
        
        if 'current_relative_error_percent' in df.columns:
            valid_errors = df['current_relative_error_percent'].dropna()
            if len(valid_errors) > 0:
                print(f"Average current relative error: {valid_errors.mean():.2f}%")
                print(f"Max current relative error: {valid_errors.max():.2f}%")
        
        phase_counts = df['phase'].value_counts()
        print("\nPhase distribution:")
        for phase, count in phase_counts.items():
            print(f"  {phase}: {count}")
        
        return output_path

def main():
    """Main function to run the EFIT history mining."""
    miner = EFITHistoryMiner()
    
    # Mine all data
    df = miner.mine_all_data()
    
    if not df.empty:
        # Save results
        output_file = miner.save_results(df)
        print(f"\nResults saved to: {output_file}")
    else:
        print("No data was extracted. Please check the file paths and permissions.")

if __name__ == "__main__":
    main()
