# src/data_processors/system_entity_manager.py

import logging
from typing import Dict, List, Any, Optional
from .entity_manager import EntityManager # Inherit from base EntityManager

class SystemEntityManager(EntityManager):
    """
    Base class for system-level entity management.
    Extends generic EntityManager to handle specialized vocabularies
    for processes, files, syscalls, registry keys, and network endpoints.
    """
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__) # Ensure logger is initialized
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO) # Fallback if no external logging setup
            
        self.process_vocab: Dict[str, int] = {}
        self.file_vocab: Dict[str, int] = {}
        self.syscall_vocab: Dict[str, int] = {}
        self.registry_vocab: Dict[str, int] = {}
        self.network_endpoint_vocab: Dict[str, int] = {} # For specific IP:Port endpoints derived from system events

        self._next_process_id = 0
        self._next_file_id = 0
        self._next_syscall_id = 0
        self._next_registry_id = 0
        self._next_network_endpoint_id = 0 # Renamed for clarity vs base EntityManager's _next_entity_id

    # Specialized getters for system-specific entities
    def get_process_id(self, process_name: str, pid: Optional[Any] = None) -> int:
        """Map process to numerical ID, optionally including PID for uniqueness."""
        key = f"PROCESS::{process_name}::{pid}" if pid else f"PROCESS::{process_name}"
        if key not in self.process_vocab:
            self.process_vocab[key] = self._next_process_id
            self._next_process_id += 1
        return self.process_vocab[key]

    def get_file_id(self, file_path: str) -> int:
        """Map file path to numerical ID."""
        key = f"FILE::{file_path}"
        if key not in self.file_vocab:
            self.file_vocab[key] = self._next_file_id
            self._next_file_id += 1
        return self.file_vocab[key]

    def get_syscall_id(self, syscall_name: str) -> int:
        """Map system call to numerical ID."""
        key = f"SYSCALL::{syscall_name}"
        if key not in self.syscall_vocab:
            self.syscall_vocab[key] = self._next_syscall_id
            self._next_syscall_id += 1
        return self.syscall_vocab[key]

    def get_registry_id(self, registry_key: str) -> int:
        """Map registry key to numerical ID."""
        key = f"REGISTRY::{registry_key}"
        if key not in self.registry_vocab:
            self.registry_vocab[key] = self._next_registry_id
            self._next_registry_id += 1
        return self.registry_vocab[key]

    def get_network_endpoint_id(self, endpoint: str) -> int:
        """Map network endpoint (IP:Port) to numerical ID for system-level network entities."""
        key = f"ENDPOINT::{endpoint}"
        if key not in self.network_endpoint_vocab:
            self.network_endpoint_vocab[key] = self._next_network_endpoint_id
            self._next_network_endpoint_id += 1
        return self.network_endpoint_vocab[key]

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Return vocabulary sizes for all entity types (including specialized system vocabs)."""
        base_sizes = super().get_vocab_sizes() # From inherited EntityManager
        system_sizes = {
            'process_vocab_size': len(self.process_vocab),
            'file_vocab_size': len(self.file_vocab),
            'syscall_vocab_size': len(self.syscall_vocab),
            'registry_vocab_size': len(self.registry_vocab),
            'network_endpoint_vocab_size': len(self.network_endpoint_vocab)
        }
        # Merge dictionaries. Note: 'action_vocab_size' will still come from base's get_action_id calls.
        return {**base_sizes, **system_sizes}
