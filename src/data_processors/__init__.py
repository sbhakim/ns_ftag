# src/data_processors/__init__.py


from .base_processor import BaseSecurityEventProcessor
from .cicids2017_processor import CICIDS2017Processor


# === NEW: OpTC Components ===
from .optc_processor import OpTCProcessor
from .optc_entity_manager import OpTCEntityManager
from .optc_relationship_extractor import OpTCRelationshipExtractor
from .optc_feature_extractor import OpTCFeatureExtractor
from .optc_label_extractor import OpTCLabelExtractor
from .system_entity_manager import SystemEntityManager # New base class

# Existing imports
from .entity_manager import EntityManager
from .relationship_extractor import RelationshipExtractor
from .feature_extractor import FeatureExtractor
from .label_extractor import LabelExtractor
from .dataset import SecurityEventDataset
from .batch_collator import custom_collate_fn