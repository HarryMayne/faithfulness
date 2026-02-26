"""
Create a small test parquet file with a subset of rows for testing.
"""
import pandas as pd
import argparse


def create_test_parquet(input_file: str, output_file: str, num_rows: int = 5):
    """
    Create a small test parquet with first N rows using pandas.
    
    Args:
        input_file: Path to full parquet file
        output_file: Path to save test parquet
        num_rows: Number of rows to include (default: 5)
    """
    print(f"Loading first {num_rows} rows from {input_file}...")
    
    # Read only first N rows
    df = pd.read_parquet(input_file)
    df_subset = df.head(num_rows)
    
    # Save to new parquet
    df_subset.to_parquet(output_file)
    
    print(f"✓ Created test parquet with {len(df_subset)} rows: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a small test parquet for testing")
    parser.add_argument("input_file", help="Path to full parquet file")
    parser.add_argument("output_file", help="Path to save test parquet")
    parser.add_argument("-n", "--num_rows", type=int, default=5, 
                        help="Number of rows to include (default: 5)")
    
    args = parser.parse_args()
    
    create_test_parquet(args.input_file, args.output_file, args.num_rows)
