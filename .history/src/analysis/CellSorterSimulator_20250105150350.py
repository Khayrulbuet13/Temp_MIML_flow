import os
import csv
import logging
import cv2
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

logger = logging.getLogger(__name__)

class CellSorterSimulator:
    """
    Simulates a cell sorter by organizing detected cells into appropriate folders
    and tracking their properties in a CSV file.
    """

    def __init__(
        self,
        base_output_dir: str,
        cell_types: List[str],
        debug: bool = False
    ):
        """
        Initialize the cell sorter simulator.

        :param base_output_dir: Base directory for all output files.
        :param cell_types: List of possible cell types (e.g., ['T4', 'T8', 'NK'])
        :param debug: Whether to log debug messages.
        """
        self.debug = debug
        self.base_dir = Path(base_output_dir)
        self.cell_types = set(cell_types)
        
        # Create base directory
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup directories for all cell types plus unknown
        self.cell_dirs = {
            cell_type: self.base_dir / f"{cell_type}_cells"
            for cell_type in self.cell_types
        }
        self.cell_dirs['unknown'] = self.base_dir / 'unknown_cells'
        
        # Create all directories
        for dir_path in self.cell_dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # CSV file for tracking all cells
        self.csv_path = self.base_dir / 'cell_sorting_results.csv'
        self.initialize_csv()
        
        # Statistics tracking for all possible types
        self.stats = {cell_type: 0 for cell_type in self.cell_dirs.keys()}
        self.total_processed = 0
        
        if self.debug:
            logger.debug(f"Initialized CellSorterSimulator with cell types: {cell_types}")

    def initialize_csv(self):
        """Initialize the CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'cell_type',
                    'confidence',
                    'max_DI',
                    'max_velocity',
                    'transition_time',
                    'patch_path',
                    'track_id'
                ])

    def _generate_unique_filename(self, cell_type: str, track_id: int) -> str:
        """Generate a unique filename for the cell patch."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"cell_{cell_type}_{track_id}_{timestamp}.png"

    def get_cell_types(self) -> Set[str]:
        """Return the set of recognized cell types."""
        return self.cell_types

    def process_cell(
        self,
        patch: Any,
        cell_type: str,
        confidence: float,
        cell_props: Dict[str, Any],
        track_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Process a detected cell by saving its patch and properties.

        :param patch: The cell patch image array
        :param cell_type: Detected cell type
        :param confidence: Model's confidence in the classification
        :param cell_props: Dictionary of cell properties
        :param track_id: Unique identifier for the cell track
        :return: Dictionary with processing results or None if failed
        """
        try:
            # If cell_type not in recognized types, mark as unknown
            if cell_type not in self.cell_types:
                logger.warning(f"Unrecognized cell type: {cell_type}. Marking as unknown.")
                cell_type = 'unknown'
            
            # Get the appropriate directory
            output_dir = self.cell_dirs[cell_type]
            
            # Generate unique filename
            filename = self._generate_unique_filename(cell_type, track_id)
            patch_path = output_dir / filename
            
            # Save the patch image
            cv2.imwrite(str(patch_path), patch)
            
            # Prepare row for CSV
            row_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                'cell_type': cell_type,
                'confidence': confidence,
                'max_DI': cell_props.get('max_DI'),
                'max_velocity': cell_props.get('max_velocity'),
                'transition_time': cell_props.get('transition_time'),
                'patch_path': str(patch_path),
                'track_id': track_id
            }
            
            # Append to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row_data.keys())
                writer.writerow(row_data)
            
            # Update statistics
            self.stats[cell_type] = self.stats.get(cell_type, 0) + 1
            self.total_processed += 1
            
            if self.debug:
                logger.debug(f"Processed cell track {track_id} as {cell_type} "
                           f"(confidence: {confidence:.2f})")
            
            return row_data
            
        except Exception as e:
            logger.error(f"Error processing cell track {track_id}: {str(e)}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get current sorting statistics."""
        stats = {
            'total_processed': self.total_processed,
            'cell_counts': self.stats.copy()
        }
        
        # Add percentages for each cell type
        if self.total_processed > 0:
            stats['cell_percentages'] = {
                cell_type: (count / self.total_processed) * 100
                for cell_type, count in self.stats.items()
            }
            
            # Calculate known vs unknown ratio
            known_cells = sum(self.stats[ct] for ct in self.cell_types)
            unknown_cells = self.stats.get('unknown', 0)
            stats['known_ratio'] = known_cells / self.total_processed if self.total_processed > 0 else 0
            
        return stats

    def generate_report(self) -> str:
        """Generate a summary report of the sorting session."""
        try:
            df = pd.read_csv(self.csv_path)
            
            report = ["Cell Sorting Session Report",
                     "=========================",
                     f"Total Cells Processed: {self.total_processed}",
                     "\nCell Type Distribution:",
                     "---------------------"]
            
            for cell_type, count in self.stats.items():
                percentage = (count / max(self.total_processed, 1)) * 100
                report.append(f"{cell_type}: {count} ({percentage:.1f}%)")
            
            # Only include confidence stats if we have data
            if len(df) > 0:
                report.extend([
                    "\nConfidence Statistics:",
                    "--------------------",
                    f"Mean Confidence: {df['confidence'].mean():.2f}",
                    f"Min Confidence: {df['confidence'].min():.2f}",
                    f"Max Confidence: {df['confidence'].max():.2f}",
                    "\nCell Properties (Mean Values):",
                    "----------------------------",
                    f"Mean DI: {df['max_DI'].mean():.2f}",
                    f"Mean Velocity: {df['max_velocity'].mean():.2f}",
                    f"Mean Transition Time: {df['transition_time'].mean():.2f}s"
                ])
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return "Error generating report"

    def cleanup(self):
        """Cleanup resources and save final statistics."""
        try:
            # Save final statistics
            stats_path = self.base_dir / 'final_statistics.txt'
            with open(stats_path, 'w') as f:
                f.write(self.generate_report())
            
            if self.debug:
                logger.debug("Cleanup completed successfully")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")