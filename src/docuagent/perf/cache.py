"""Persistent caching for OCR and VLM results."""

import json
import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from dataclasses import dataclass
import pickle
import base64


@dataclass
class CacheEntry:
    """A cache entry with metadata."""
    key: str
    value: Any
    created_at: float
    access_count: int
    last_accessed: float
    size_bytes: int


class PersistentCache:
    """Persistent cache using SQLite for OCR and VLM results."""
    
    def __init__(self, cache_dir: str = ".cache", max_size_mb: int = 1000):
        """Initialize persistent cache.
        
        Args:
            cache_dir: Directory to store cache database
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.cache_dir / "cache.db"
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at REAL,
                    access_count INTEGER,
                    last_accessed REAL,
                    size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache(last_accessed)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count ON cache(access_count)
            """)
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)
    
    def _compute_key(self, prefix: str, data: Any) -> str:
        """Compute cache key from prefix and data."""
        # Create hash of data for key
        if isinstance(data, (str, bytes)):
            data_str = data if isinstance(data, str) else data.decode('utf-8', errors='ignore')
        else:
            data_str = json.dumps(data, sort_keys=True)
        
        hash_obj = hashlib.md5(data_str.encode('utf-8'))
        return f"{prefix}:{hash_obj.hexdigest()}"
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
            result = cursor.fetchone()
            return result[0] or 0
    
    def _evict_old_entries(self):
        """Evict old entries if cache is too large."""
        total_size = self._get_total_size()
        
        if total_size <= self.max_size_bytes:
            return
        
        # Remove least recently used entries
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT key, size_bytes FROM cache 
                ORDER BY last_accessed ASC, access_count ASC
            """)
            
            current_size = total_size
            for key, size_bytes in cursor:
                if current_size <= self.max_size_bytes * 0.8:  # Keep 80% of max size
                    break
                
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                current_size -= size_bytes
    
    def get(self, prefix: str, data: Any) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            prefix: Cache prefix (e.g., 'ocr', 'vlm')
            data: Data to use for key computation
            
        Returns:
            Cached value or None if not found
        """
        key = self._compute_key(prefix, data)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT value, access_count FROM cache WHERE key = ?
            """, (key,))
            
            result = cursor.fetchone()
            if result is None:
                return None
            
            value_data, access_count = result
            
            # Update access statistics
            conn.execute("""
                UPDATE cache 
                SET access_count = ?, last_accessed = ?
                WHERE key = ?
            """, (access_count + 1, time.time(), key))
            
            return self._deserialize_value(value_data)
    
    def set(self, prefix: str, data: Any, value: Any) -> None:
        """Set value in cache.
        
        Args:
            prefix: Cache prefix (e.g., 'ocr', 'vlm')
            data: Data to use for key computation
            value: Value to cache
        """
        key = self._compute_key(prefix, data)
        value_data = self._serialize_value(value)
        size_bytes = len(value_data)
        
        with sqlite3.connect(self.db_path) as conn:
            # Check if key already exists
            cursor = conn.execute("SELECT key FROM cache WHERE key = ?", (key,))
            if cursor.fetchone():
                # Update existing entry
                conn.execute("""
                    UPDATE cache 
                    SET value = ?, created_at = ?, access_count = 0, 
                        last_accessed = ?, size_bytes = ?
                    WHERE key = ?
                """, (value_data, time.time(), time.time(), size_bytes, key))
            else:
                # Insert new entry
                conn.execute("""
                    INSERT INTO cache (key, value, created_at, access_count, last_accessed, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (key, value_data, time.time(), 0, time.time(), size_bytes))
        
        # Evict old entries if needed
        self._evict_old_entries()
    
    def delete(self, prefix: str, data: Any) -> bool:
        """Delete value from cache.
        
        Args:
            prefix: Cache prefix (e.g., 'ocr', 'vlm')
            data: Data to use for key computation
            
        Returns:
            True if deleted, False if not found
        """
        key = self._compute_key(prefix, data)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            return cursor.rowcount > 0
    
    def clear(self, prefix: Optional[str] = None) -> int:
        """Clear cache entries.
        
        Args:
            prefix: Optional prefix to clear (if None, clears all)
            
        Returns:
            Number of entries cleared
        """
        with sqlite3.connect(self.db_path) as conn:
            if prefix:
                cursor = conn.execute("DELETE FROM cache WHERE key LIKE ?", (f"{prefix}:%",))
            else:
                cursor = conn.execute("DELETE FROM cache")
            
            return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Total entries
            cursor = conn.execute("SELECT COUNT(*) FROM cache")
            total_entries = cursor.fetchone()[0]
            
            # Total size
            cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
            total_size = cursor.fetchone()[0] or 0
            
            # Size by prefix
            cursor = conn.execute("""
                SELECT 
                    SUBSTR(key, 1, INSTR(key, ':') - 1) as prefix,
                    COUNT(*) as count,
                    SUM(size_bytes) as size
                FROM cache 
                GROUP BY prefix
            """)
            size_by_prefix = {row[0]: {'count': row[1], 'size': row[2]} for row in cursor}
            
            # Access statistics
            cursor = conn.execute("""
                SELECT 
                    AVG(access_count) as avg_access,
                    MAX(access_count) as max_access,
                    AVG(last_accessed) as avg_last_access
                FROM cache
            """)
            access_stats = cursor.fetchone()
            
            return {
                'total_entries': total_entries,
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'size_by_prefix': size_by_prefix,
                'avg_access_count': access_stats[0] or 0,
                'max_access_count': access_stats[1] or 0,
                'avg_last_access': access_stats[2] or 0
            }


class OCRCache:
    """Specialized cache for OCR results."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize OCR cache."""
        self.cache = PersistentCache(cache_dir)
        self.prefix = "ocr"
    
    def get_ocr_result(self, image_path: str, lang: str = "ml") -> Optional[Dict[str, Any]]:
        """Get cached OCR result.
        
        Args:
            image_path: Path to image file
            lang: OCR language
            
        Returns:
            Cached OCR result or None
        """
        key_data = {
            'image_path': str(image_path),
            'lang': lang,
            'image_mtime': Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        }
        
        return self.cache.get(self.prefix, key_data)
    
    def set_ocr_result(self, image_path: str, lang: str, result: Dict[str, Any]) -> None:
        """Set cached OCR result.
        
        Args:
            image_path: Path to image file
            lang: OCR language
            result: OCR result to cache
        """
        key_data = {
            'image_path': str(image_path),
            'lang': lang,
            'image_mtime': Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        }
        
        self.cache.set(self.prefix, key_data, result)
    
    def clear_ocr_cache(self) -> int:
        """Clear OCR cache."""
        return self.cache.clear(self.prefix)


class VLMCache:
    """Specialized cache for VLM results."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize VLM cache."""
        self.cache = PersistentCache(cache_dir)
        self.prefix = "vlm"
    
    def get_vlm_result(self, image_path: str, prompt: str, model: str) -> Optional[Dict[str, Any]]:
        """Get cached VLM result.
        
        Args:
            image_path: Path to image file
            prompt: VLM prompt
            model: VLM model name
            
        Returns:
            Cached VLM result or None
        """
        key_data = {
            'image_path': str(image_path),
            'prompt': prompt,
            'model': model,
            'image_mtime': Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        }
        
        return self.cache.get(self.prefix, key_data)
    
    def set_vlm_result(self, image_path: str, prompt: str, model: str, result: Dict[str, Any]) -> None:
        """Set cached VLM result.
        
        Args:
            image_path: Path to image file
            prompt: VLM prompt
            model: VLM model name
            result: VLM result to cache
        """
        key_data = {
            'image_path': str(image_path),
            'prompt': prompt,
            'model': model,
            'image_mtime': Path(image_path).stat().st_mtime if Path(image_path).exists() else 0
        }
        
        self.cache.set(self.prefix, key_data, result)
    
    def clear_vlm_cache(self) -> int:
        """Clear VLM cache."""
        return self.cache.clear(self.prefix)


def main():
    """Command line interface for cache management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage persistent cache")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument("--clear", help="Clear cache (ocr, vlm, or all)")
    parser.add_argument("--max-size", type=int, default=1000, help="Maximum cache size in MB")
    
    args = parser.parse_args()
    
    cache = PersistentCache(args.cache_dir, args.max_size)
    
    if args.stats:
        stats = cache.get_stats()
        print("Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Total size: {stats['total_size_mb']:.1f} MB")
        print(f"  Average access count: {stats['avg_access_count']:.1f}")
        print(f"  Max access count: {stats['max_access_count']}")
        
        print("\nSize by prefix:")
        for prefix, info in stats['size_by_prefix'].items():
            print(f"  {prefix}: {info['count']} entries, {info['size'] / (1024*1024):.1f} MB")
    
    if args.clear:
        if args.clear == "all":
            cleared = cache.clear()
        elif args.clear == "ocr":
            ocr_cache = OCRCache(args.cache_dir)
            cleared = ocr_cache.clear_ocr_cache()
        elif args.clear == "vlm":
            vlm_cache = VLMCache(args.cache_dir)
            cleared = vlm_cache.clear_vlm_cache()
        else:
            print(f"Unknown cache type: {args.clear}")
            return
        
        print(f"Cleared {cleared} cache entries")


if __name__ == "__main__":
    main()
