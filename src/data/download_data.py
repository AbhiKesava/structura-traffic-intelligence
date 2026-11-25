"""
CLI script to prepare and verify METR-LA dataset.
"""

from src.data.loader import load_metr_la
from src.utils.config import RAW_DATA_DIR

def main():
    print(f"Loading/preparing METR-LA data in {RAW_DATA_DIR}...")
    speed_df, adj_mx, sensor_df = load_metr_la()
    print("METR-LA Dataset Summary:")
    print(f" - Speed records: {len(speed_df)} time steps x {speed_df.shape[1]} sensors")
    print(f" - Time range: {speed_df.index.min()} to {speed_df.index.max()}")
    print(f" - Speed statistics (mph): Min={speed_df.values.min():.1f}, Mean={speed_df.values.mean():.1f}, Max={speed_df.values.max():.1f}")
    print(f" - Adjacency matrix shape: {adj_mx.shape}, Non-zero edges: {(adj_mx > 0).sum()}")
    print(f" - Sensor highways: {sensor_df['highway'].value_counts().to_dict()}")
    print("\nSample Sensor Metadata:")
    print(sensor_df.head())

if __name__ == "__main__":
    main()
