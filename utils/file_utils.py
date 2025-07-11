"""
Market Research System v1.0 - File Utilities
Created: 2022
Author: Independent Market Researcher

File handling utility functions for data storage and management.
"""

import os
import json
import csv
import shutil
import glob
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import pickle
from datetime import datetime
import hashlib

def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    
    Returns:
        Path object of the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_to_csv(data: Union[pd.DataFrame, List[Dict]], filepath: Union[str, Path], 
                index: bool = False, **kwargs) -> bool:
    """
    Save data to CSV file.
    
    Args:
        data: Data to save (DataFrame or list of dictionaries)
        filepath: Output file path
        index: Whether to include index in CSV
        **kwargs: Additional arguments for pandas.to_csv
    
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=index, **kwargs)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=index, **kwargs)
        else:
            raise ValueError("Data must be DataFrame or list of dictionaries")
        
        return True
    except Exception as e:
        print(f"Error saving CSV to {filepath}: {e}")
        return False

def load_from_csv(filepath: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Load data from CSV file.
    
    Args:
        filepath: Input file path
        **kwargs: Additional arguments for pandas.read_csv
    
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        print(f"Error loading CSV from {filepath}: {e}")
        return None

def save_to_json(data: Union[Dict, List], filepath: Union[str, Path], 
                 indent: int = 2, ensure_ascii: bool = False) -> bool:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
        ensure_ascii: Whether to ensure ASCII encoding
    
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, default=str)
        
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False

def load_from_json(filepath: Union[str, Path]) -> Optional[Union[Dict, List]]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data if successful, None otherwise
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None

def save_to_pickle(data: Any, filepath: Union[str, Path]) -> bool:
    """
    Save data to pickle file.
    
    Args:
        data: Data to save
        filepath: Output file path
    
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        ensure_directory(filepath.parent)
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    except Exception as e:
        print(f"Error saving pickle to {filepath}: {e}")
        return False

def load_from_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data if successful, None otherwise
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle from {filepath}: {e}")
        return None

def get_file_size(filepath: Union[str, Path]) -> Optional[int]:
    """
    Get file size in bytes.
    
    Args:
        filepath: File path
    
    Returns:
        File size in bytes, None if file doesn't exist
    """
    try:
        filepath = Path(filepath)
        if filepath.exists():
            return filepath.stat().st_size
        return None
    except Exception as e:
        print(f"Error getting file size for {filepath}: {e}")
        return None

def backup_file(filepath: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """
    Create backup of a file.
    
    Args:
        filepath: File to backup
        backup_dir: Backup directory (same directory if None)
    
    Returns:
        Path to backup file if successful
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return None
        
        if backup_dir is None:
            backup_dir = filepath.parent
        else:
            backup_dir = Path(backup_dir)
            ensure_directory(backup_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(filepath, backup_path)
        return backup_path
    except Exception as e:
        print(f"Error backing up file {filepath}: {e}")
        return None

def clean_filename(filename: str) -> str:
    """
    Clean filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
    
    Returns:
        Cleaned filename
    """
    # Characters not allowed in filenames
    invalid_chars = '<>:"/\\|?*'
    
    cleaned = filename
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip('. ')
    
    # Limit length to 255 characters (common filesystem limit)
    if len(cleaned) > 255:
        name, ext = os.path.splitext(cleaned)
        cleaned = name[:255-len(ext)] + ext
    
    return cleaned

def get_file_extension(filepath: Union[str, Path]) -> str:
    """
    Get file extension.
    
    Args:
        filepath: File path
    
    Returns:
        File extension (including dot)
    """
    return Path(filepath).suffix

def list_files_by_pattern(directory: Union[str, Path], pattern: str) -> List[Path]:
    """
    List files matching pattern in directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., '*.csv', '*.json')
    
    Returns:
        List of matching file paths
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return []
        
        return list(directory.glob(pattern))
    except Exception as e:
        print(f"Error listing files with pattern {pattern} in {directory}: {e}")
        return []

def get_files_by_date(directory: Union[str, Path], days_old: int = 0) -> List[Path]:
    """
    Get files modified within specified days.
    
    Args:
        directory: Directory to search
        days_old: Number of days (0 for today, 1 for yesterday, etc.)
    
    Returns:
        List of file paths
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return []
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        matching_files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime >= cutoff_time:
                matching_files.append(file_path)
        
        return matching_files
    except Exception as e:
        print(f"Error getting files by date in {directory}: {e}")
        return []

def compress_files(file_paths: List[Union[str, Path]], output_path: Union[str, Path]) -> bool:
    """
    Compress files into a ZIP archive.
    
    Args:
        file_paths: List of files to compress
        output_path: Output ZIP file path
    
    Returns:
        True if successful
    """
    try:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in file_paths:
                file_path = Path(file_path)
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
        
        return True
    except Exception as e:
        print(f"Error compressing files to {output_path}: {e}")
        return False

def extract_files(zip_path: Union[str, Path], extract_to: Union[str, Path]) -> bool:
    """
    Extract files from ZIP archive.
    
    Args:
        zip_path: ZIP file path
        extract_to: Directory to extract to
    
    Returns:
        True if successful
    """
    try:
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)
        
        if not zip_path.exists():
            print(f"ZIP file not found: {zip_path}")
            return False
        
        ensure_directory(extract_to)
        
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_to)
        
        return True
    except Exception as e:
        print(f"Error extracting {zip_path} to {extract_to}: {e}")
        return False

def get_file_hash(filepath: Union[str, Path], algorithm: str = 'md5') -> Optional[str]:
    """
    Calculate file hash.
    
    Args:
        filepath: File path
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
    
    Returns:
        File hash string
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return None
        
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Move file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        True if successful
    """
    try:
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            print(f"Source file not found: {src}")
            return False
        
        ensure_directory(dst.parent)
        shutil.move(str(src), str(dst))
        return True
    except Exception as e:
        print(f"Error moving file from {src} to {dst}: {e}")
        return False

def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    
    Returns:
        True if successful
    """
    try:
        src = Path(src)
        dst = Path(dst)
        
        if not src.exists():
            print(f"Source file not found: {src}")
            return False
        
        ensure_directory(dst.parent)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        print(f"Error copying file from {src} to {dst}: {e}")
        return False

def delete_file(filepath: Union[str, Path]) -> bool:
    """
    Delete file safely.
    
    Args:
        filepath: File path to delete
    
    Returns:
        True if successful
    """
    try:
        filepath = Path(filepath)
        if filepath.exists():
            filepath.unlink()
            return True
        else:
            print(f"File not found: {filepath}")
            return False
    except Exception as e:
        print(f"Error deleting file {filepath}: {e}")
        return False

def cleanup_old_files(directory: Union[str, Path], days_old: int = 30, 
                     file_pattern: str = "*", dry_run: bool = True) -> List[Path]:
    """
    Clean up old files in directory (for Indian market data maintenance).
    
    Args:
        directory: Directory to clean
        days_old: Files older than this many days
        file_pattern: Pattern to match files
        dry_run: If True, only list files without deleting
    
    Returns:
        List of files that were/would be deleted
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return []
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        old_files = []
        
        for file_path in directory.glob(file_pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                old_files.append(file_path)
                if not dry_run:
                    file_path.unlink()
                    print(f"Deleted old file: {file_path}")
        
        if dry_run:
            print(f"Found {len(old_files)} old files (dry run - not deleted)")
        
        return old_files
    except Exception as e:
        print(f"Error cleaning up old files in {directory}: {e}")
        return []

def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
    
    Returns:
        Total size in bytes
    """
    try:
        directory = Path(directory)
        total_size = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    except Exception as e:
        print(f"Error calculating directory size for {directory}: {e}")
        return 0

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.2f} {size_names[i]}"

def create_data_manifest(directory: Union[str, Path], output_file: str = "data_manifest.json") -> bool:
    """
    Create manifest file with metadata of all data files (for Indian market tracking).
    
    Args:
        directory: Directory to scan
        output_file: Output manifest filename
    
    Returns:
        True if successful
    """
    try:
        directory = Path(directory)
        if not directory.exists():
            return False
        
        manifest = {
            "created_at": datetime.now().isoformat(),
            "directory": str(directory),
            "files": []
        }
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                file_info = {
                    "name": file_path.name,
                    "path": str(file_path.relative_to(directory)),
                    "size_bytes": stat.st_size,
                    "size_formatted": format_file_size(stat.st_size),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "extension": file_path.suffix,
                    "hash_md5": get_file_hash(file_path, 'md5')
                }
                manifest["files"].append(file_info)
        
        manifest["total_files"] = len(manifest["files"])
        manifest["total_size_bytes"] = sum(f["size_bytes"] for f in manifest["files"])
        manifest["total_size_formatted"] = format_file_size(manifest["total_size_bytes"])
        
        output_path = directory / output_file
        return save_to_json(manifest, output_path)
    except Exception as e:
        print(f"Error creating data manifest: {e}")
        return False

def validate_csv_structure(filepath: Union[str, Path], required_columns: List[str]) -> Dict[str, Any]:
    """
    Validate CSV file structure for Indian market data consistency.
    
    Args:
        filepath: CSV file path
        required_columns: List of required column names
    
    Returns:
        Validation results dictionary
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            return {"valid": False, "error": "File not found"}
        
        df = pd.read_csv(filepath)
        
        result = {
            "valid": True,
            "file_path": str(filepath),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_columns": [],
            "extra_columns": [],
            "data_types": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "file_size": get_file_size(filepath),
            "validated_at": datetime.now().isoformat()
        }
        
        # Check for missing required columns
        for col in required_columns:
            if col not in df.columns:
                result["missing_columns"].append(col)
                result["valid"] = False
        
        # Check for extra columns
        for col in df.columns:
            if col not in required_columns:
                result["extra_columns"].append(col)
        
        return result
    except Exception as e:
        return {"valid": False, "error": str(e)}

def archive_old_data(source_dir: Union[str, Path], archive_dir: Union[str, Path], 
                    days_old: int = 90, compress: bool = True) -> bool:
    """
    Archive old Indian market data files to separate directory.
    
    Args:
        source_dir: Source directory with data files
        archive_dir: Archive directory
        days_old: Archive files older than this many days
        compress: Whether to compress archived files
    
    Returns:
        True if successful
    """
    try:
        source_dir = Path(source_dir)
        archive_dir = Path(archive_dir)
        
        if not source_dir.exists():
            return False
        
        ensure_directory(archive_dir)
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        archived_files = []
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                # Create archive subdirectory structure
                rel_path = file_path.relative_to(source_dir)
                archive_path = archive_dir / rel_path
                ensure_directory(archive_path.parent)
                
                # Move file to archive
                shutil.move(str(file_path), str(archive_path))
                archived_files.append(archive_path)
        
        # Compress archived files if requested
        if compress and archived_files:
            archive_date = datetime.now().strftime("%Y%m%d")
            zip_path = archive_dir / f"archived_data_{archive_date}.zip"
            
            if compress_files(archived_files, zip_path):
                # Remove individual files after successful compression
                for file_path in archived_files:
                    if file_path.exists():
                        file_path.unlink()
                
                print(f"Archived and compressed {len(archived_files)} files to {zip_path}")
            else:
                print(f"Archived {len(archived_files)} files (compression failed)")
        else:
            print(f"Archived {len(archived_files)} files")
        
        return True
    except Exception as e:
        print(f"Error archiving old data: {e}")
        return False

def create_daily_backup(data_dir: Union[str, Path], backup_base_dir: Union[str, Path]) -> Optional[Path]:
    """
    Create daily backup of Indian market data directory.
    
    Args:
        data_dir: Data directory to backup
        backup_base_dir: Base backup directory
    
    Returns:
        Path to backup if successful
    """
    try:
        data_dir = Path(data_dir)
        backup_base_dir = Path(backup_base_dir)
        
        if not data_dir.exists():
            return None
        
        # Create dated backup directory
        backup_date = datetime.now().strftime("%Y%m%d")
        backup_dir = backup_base_dir / f"backup_{backup_date}"
        ensure_directory(backup_dir)
        
        # Copy all files
        for file_path in data_dir.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(data_dir)
                backup_path = backup_dir / rel_path
                ensure_directory(backup_path.parent)
                shutil.copy2(file_path, backup_path)
        
        # Create backup manifest
        create_data_manifest(backup_dir, "backup_manifest.json")
        
        # Compress backup
        zip_path = backup_base_dir / f"market_data_backup_{backup_date}.zip"
        if compress_files(list(backup_dir.rglob('*')), zip_path):
            # Remove uncompressed backup directory
            shutil.rmtree(backup_dir)
            print(f"Daily backup created: {zip_path}")
            return zip_path
        
        return backup_dir
    except Exception as e:
        print(f"Error creating daily backup: {e}")
        return None

def get_indian_market_data_stats(data_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Get statistics for Indian market data files.
    
    Args:
        data_dir: Data directory path
    
    Returns:
        Statistics dictionary
    """
    try:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return {}
        
        stats = {
            "directory": str(data_dir),
            "scan_time": datetime.now().isoformat(),
            "file_types": {},
            "total_files": 0,
            "total_size_bytes": 0,
            "oldest_file": None,
            "newest_file": None,
            "data_coverage": {}
        }
        
        oldest_time = float('inf')
        newest_time = 0
        
        for file_path in data_dir.rglob('*'):
            if file_path.is_file():
                stats["total_files"] += 1
                
                # File size
                size = file_path.stat().st_size
                stats["total_size_bytes"] += size
                
                # File type
                ext = file_path.suffix.lower()
                if ext not in stats["file_types"]:
                    stats["file_types"][ext] = {"count": 0, "size_bytes": 0}
                stats["file_types"][ext]["count"] += 1
                stats["file_types"][ext]["size_bytes"] += size
                
                # File dates
                mtime = file_path.stat().st_mtime
                if mtime < oldest_time:
                    oldest_time = mtime
                    stats["oldest_file"] = {
                        "path": str(file_path),
                        "date": datetime.fromtimestamp(mtime).isoformat()
                    }
                if mtime > newest_time:
                    newest_time = mtime
                    stats["newest_file"] = {
                        "path": str(file_path),
                        "date": datetime.fromtimestamp(mtime).isoformat()
                    }
        
        stats["total_size_formatted"] = format_file_size(stats["total_size_bytes"])
        
        # Format file type sizes
        for ext in stats["file_types"]:
            stats["file_types"][ext]["size_formatted"] = format_file_size(
                stats["file_types"][ext]["size_bytes"]
            )
        
        return stats
    except Exception as e:
        print(f"Error getting market data stats: {e}")
        return {}

# Indian Market Specific Utilities (2022 focus)
def validate_nse_data_format(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate NSE data file format.
    
    Args:
        filepath: NSE data file path
    
    Returns:
        Validation results
    """
    nse_required_columns = [
        'SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'DATE'
    ]
    
    return validate_csv_structure(filepath, nse_required_columns)

def validate_bse_data_format(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate BSE data file format.
    
    Args:
        filepath: BSE data file path
    
    Returns:
        Validation results
    """
    bse_required_columns = [
        'SC_CODE', 'SC_NAME', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'NO_TRADES', 'NO_OF_SHRS', 'NET_TURNOV', 'TDCLOINDI'
    ]
    
    return validate_csv_structure(filepath, bse_required_columns)

def setup_indian_market_directories(base_dir: Union[str, Path]) -> Dict[str, Path]:
    """
    Setup directory structure for Indian market research system.
    
    Args:
        base_dir: Base directory path
    
    Returns:
        Dictionary of created directory paths
    """
    base_dir = Path(base_dir)
    
    directories = {
        'raw_data': base_dir / 'data' / 'raw',
        'processed_data': base_dir / 'data' / 'processed',
        'nse_data': base_dir / 'data' / 'raw' / 'nse',
        'bse_data': base_dir / 'data' / 'raw' / 'bse',
        'mutual_funds': base_dir / 'data' / 'raw' / 'mutual_funds',
        'economic_indicators': base_dir / 'data' / 'raw' / 'economic',
        'reports': base_dir / 'reports',
        'daily_reports': base_dir / 'reports' / 'daily',
        'weekly_reports': base_dir / 'reports' / 'weekly',
        'monthly_reports': base_dir / 'reports' / 'monthly',
        'logs': base_dir / 'logs',
        'backups': base_dir / 'backups',
        'temp': base_dir / 'temp'
    }
    
    for name, path in directories.items():
        ensure_directory(path)
        print(f"Created directory: {path}")
    
    return directories

# End of file_utils.py for Version 1 (2022)