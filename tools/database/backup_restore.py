#!/usr/bin/env python3
"""
Database Backup and Restore Tool for Market Research System v1.0
Handles backup and restoration of market data and analysis results
"""

import os
import sys
import json
import sqlite3
import shutil
import tarfile
import datetime
import argparse
import logging
from pathlib import Path
from typing import Optional, List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseBackupRestore:
    """Database backup and restore utilities"""
    
    def __init__(self, db_path: str = "data/market_research.db", 
                 backup_dir: str = "data/backups"):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a complete backup of the database and data files"""
        if not backup_name:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"market_research_backup_{timestamp}"
            
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        try:
            # Create temporary directory for backup files
            temp_dir = self.backup_dir / f"temp_{backup_name}"
            temp_dir.mkdir(exist_ok=True)
            
            # Backup database
            if self.db_path.exists():
                db_backup_path = temp_dir / "market_research.db"
                shutil.copy2(self.db_path, db_backup_path)
                logger.info(f"Database backed up to {db_backup_path}")
            
            # Backup data directories
            data_dirs = [
                "data/raw",
                "data/processed", 
                "data/cache",
                "reports",
                "logs"
            ]
            
            for data_dir in data_dirs:
                src_path = Path(data_dir)
                if src_path.exists():
                    dst_path = temp_dir / data_dir
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    logger.info(f"Data directory {data_dir} backed up")
            
            # Create metadata file
            metadata = {
                "backup_name": backup_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "1.0",
                "system": "Market Research System",
                "files_included": [str(p.relative_to(temp_dir)) for p in temp_dir.rglob("*") if p.is_file()]
            }
            
            metadata_path = temp_dir / "backup_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create compressed archive
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(temp_dir, arcname=backup_name)
            
            # Cleanup temporary directory
            shutil.rmtree(temp_dir)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise
    
    def restore_backup(self, backup_path: str, target_dir: str = ".") -> bool:
        """Restore from backup file"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
            
        try:
            # Extract backup
            with tarfile.open(backup_file, "r:gz") as tar:
                tar.extractall(path=target_dir)
            
            # Find extracted directory
            extracted_dirs = [d for d in Path(target_dir).iterdir() 
                            if d.is_dir() and d.name.startswith("market_research_backup_")]
            
            if not extracted_dirs:
                logger.error("No backup directory found in extracted files")
                return False
                
            backup_content_dir = extracted_dirs[0]
            
            # Read metadata
            metadata_path = backup_content_dir / "backup_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                logger.info(f"Restoring backup from {metadata['timestamp']}")
            
            # Restore database
            db_backup = backup_content_dir / "market_research.db"
            if db_backup.exists():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(db_backup, self.db_path)
                logger.info("Database restored")
            
            # Restore data directories
            for item in backup_content_dir.iterdir():
                if item.is_dir() and item.name in ["data", "reports", "logs"]:
                    target_path = Path(target_dir) / item.name
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(item, target_path)
                    logger.info(f"Directory {item.name} restored")
            
            # Cleanup extracted directory
            shutil.rmtree(backup_content_dir)
            
            logger.info("Restore completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                with tarfile.open(backup_file, "r:gz") as tar:
                    # Try to extract metadata
                    metadata_members = [m for m in tar.getmembers() 
                                      if m.name.endswith("backup_metadata.json")]
                    
                    if metadata_members:
                        metadata_file = tar.extractfile(metadata_members[0])
                        metadata = json.load(metadata_file)
                        
                        backups.append({
                            "file": str(backup_file),
                            "name": metadata.get("backup_name", backup_file.stem),
                            "timestamp": metadata.get("timestamp", "Unknown"),
                            "size": backup_file.stat().st_size,
                            "version": metadata.get("version", "Unknown")
                        })
                    else:
                        # Fallback for backups without metadata
                        backups.append({
                            "file": str(backup_file),
                            "name": backup_file.stem,
                            "timestamp": datetime.datetime.fromtimestamp(
                                backup_file.stat().st_mtime
                            ).isoformat(),
                            "size": backup_file.stat().st_size,
                            "version": "Unknown"
                        })
                        
            except Exception as e:
                logger.warning(f"Could not read backup {backup_file}: {e}")
                
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_minimum: int = 5):
        """Remove old backup files"""
        backups = self.list_backups()
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=keep_days)
        
        # Keep at least minimum number of backups
        if len(backups) <= keep_minimum:
            logger.info(f"Only {len(backups)} backups found, keeping all")
            return
            
        deleted_count = 0
        for i, backup in enumerate(backups):
            # Skip if within minimum keep count
            if i < keep_minimum:
                continue
                
            try:
                backup_time = datetime.datetime.fromisoformat(
                    backup["timestamp"].replace("Z", "+00:00")
                )
                if backup_time < cutoff_date:
                    os.remove(backup["file"])
                    deleted_count += 1
                    logger.info(f"Deleted old backup: {backup['name']}")
                    
            except Exception as e:
                logger.warning(f"Could not process backup {backup['name']}: {e}")
        
        logger.info(f"Cleanup completed. Deleted {deleted_count} old backups")

def main():
    parser = argparse.ArgumentParser(description="Database Backup/Restore Tool")
    parser.add_argument("action", choices=["backup", "restore", "list", "cleanup"],
                       help="Action to perform")
    parser.add_argument("--backup-file", help="Backup file path for restore")
    parser.add_argument("--backup-name", help="Custom backup name")
    parser.add_argument("--db-path", default="data/market_research.db",
                       help="Database file path")
    parser.add_argument("--backup-dir", default="data/backups",
                       help="Backup directory")
    parser.add_argument("--keep-days", type=int, default=30,
                       help="Days to keep backups during cleanup")
    parser.add_argument("--keep-minimum", type=int, default=5,
                       help="Minimum number of backups to keep")
    
    args = parser.parse_args()
    
    backup_tool = DatabaseBackupRestore(args.db_path, args.backup_dir)
    
    if args.action == "backup":
        try:
            backup_path = backup_tool.create_backup(args.backup_name)
            print(f"Backup created: {backup_path}")
        except Exception as e:
            print(f"Backup failed: {e}")
            sys.exit(1)
            
    elif args.action == "restore":
        if not args.backup_file:
            print("--backup-file is required for restore action")
            sys.exit(1)
        
        success = backup_tool.restore_backup(args.backup_file)
        if success:
            print("Restore completed successfully")
        else:
            print("Restore failed")
            sys.exit(1)
            
    elif args.action == "list":
        backups = backup_tool.list_backups()
        if not backups:
            print("No backups found")
        else:
            print(f"{'Name':<30} {'Date':<20} {'Size (MB)':<10} {'Version':<10}")
            print("-" * 80)
            for backup in backups:
                size_mb = backup["size"] / (1024 * 1024)
                timestamp = backup["timestamp"][:19]  # Remove microseconds
                print(f"{backup['name']:<30} {timestamp:<20} {size_mb:<10.2f} {backup['version']:<10}")
                
    elif args.action == "cleanup":
        backup_tool.cleanup_old_backups(args.keep_days, args.keep_minimum)
        print("Cleanup completed")

if __name__ == "__main__":
    main()