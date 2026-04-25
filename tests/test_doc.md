Test Coverage Summary
✅ TESTED
RecordingEntry: Path parsing (root & sample-level), validation, immutability, cache key generation
JsonCacheStore: Save/load operations, atomic writes, file persistence, empty cache
Manager Core: Initialization, recordings retrieval, layout detection, refresh
Filtering: All operators (==, !=, <, >, <=, >=, contain, not contain)
Data Validation: Date format (6-digit only), missing data files ignored
Cache Persistence: Cross-instance persistence, incremental scanning
❌ NOT TESTED
Concurrent Scanning: ThreadPoolExecutor behavior with max_workers parameter
Error Handling: Partial failures during scanning, file permission errors, inaccessible directories
Cleanup: Removal of cached entries when directories are deleted from disk
Performance: Behavior with large datasets or deeply nested structures
Logging: Log output verification
Edge Cases: Special characters in paths, symlinks, corrupted H5 files
Custom Cache Stores: Only JsonCacheStore tested; BaseCacheStore subclasses not covered
Disk I/O Errors: NAS timeouts, write failures, race conditions