import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from functools import wraps
import hashlib
import secrets
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import uuid

# Security and encryption
from cryptography.fernet import Fernet
import jwt


class EventType(Enum):
    """Event types for user interactions"""
    PAGE_VIEW = "page_view"
    CLICK = "click"
    PURCHASE = "purchase"
    SEARCH = "search"
    CONTENT_VIEW = "content_view"
    AB_TEST = "ab_test"


class SecurityLevel(Enum):
    """Security levels for data classification"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class UserEvent:
    """User interaction event"""
    user_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    session_id: str
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'session_id': self.session_id,
            'security_level': self.security_level.value
        }


@dataclass
class UserProfile:
    """Dynamic user profile with adaptive features"""
    user_id: str
    preferences: Dict[str, float]
    segments: List[str]
    behavioral_patterns: Dict[str, Any]
    last_updated: datetime
    version: int = 1
    confidence_scores: Dict[str, float] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}
    
    def update_preference(self, key: str, value: float, weight: float = 0.1):
        """Update preference with exponential moving average"""
        current = self.preferences.get(key, 0.0)
        self.preferences[key] = current * (1 - weight) + value * weight
        self.confidence_scores[key] = min(1.0, self.confidence_scores.get(key, 0.0) + 0.1)
        self.last_updated = datetime.now()
        self.version += 1


class SecurityManager:
    """Handles encryption, authentication, and authorization"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.cipher = Fernet(Fernet.generate_key())
        self.rate_limits = defaultdict(lambda: deque())
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token with permissions"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            return jwt.decode(token, self.secret_key, algorithms=['HS256'])
        except jwt.InvalidTokenError:
            return None
    
    def rate_limit(self, identifier: str, max_requests: int = 100, window: int = 3600) -> bool:
        """Rate limiting implementation"""
        now = time.time()
        self.rate_limits[identifier] = deque([
            req_time for req_time in self.rate_limits[identifier]
            if now - req_time < window
        ])
        
        if len(self.rate_limits[identifier]) >= max_requests:
            return False
        
        self.rate_limits[identifier].append(now)
        return True


class DataStore(ABC):
    """Abstract data store interface"""
    
    @abstractmethod
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        pass
    
    @abstractmethod
    async def save_user_profile(self, profile: UserProfile) -> bool:
        pass
    
    @abstractmethod
    async def store_event(self, event: UserEvent) -> bool:
        pass


class InMemoryDataStore(DataStore):
    """In-memory data store with thread safety"""
    
    def __init__(self):
        self.profiles: Dict[str, UserProfile] = {}
        self.events: List[UserEvent] = []
        self.lock = threading.RLock()
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        with self.lock:
            return self.profiles.get(user_id)
    
    async def save_user_profile(self, profile: UserProfile) -> bool:
        with self.lock:
            self.profiles[profile.user_id] = profile
            return True
    
    async def store_event(self, event: UserEvent) -> bool:
        with self.lock:
            self.events.append(event)
            return True


class LLMAdapter:
    """Adapter for LLM integration"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.cache = {}
    
    async def generate_personalized_content(self, 
                                          user_profile: UserProfile,
                                          content_type: str,
                                          context: Dict[str, Any]) -> str:
        """Generate personalized content using LLM"""
        cache_key = self._generate_cache_key(user_profile, content_type, context)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simulate LLM API call
        prompt = self._build_prompt(user_profile, content_type, context)
        
        # Mock response - in real implementation, call actual LLM API
        response = f"Personalized {content_type} for user {user_profile.user_id}"
        
        self.cache[cache_key] = response
        return response
    
    def _build_prompt(self, profile: UserProfile, content_type: str, context: Dict[str, Any]) -> str:
        """Build LLM prompt from user profile and context"""
        return f"""
        User Profile:
        - Preferences: {profile.preferences}
        - Segments: {profile.segments}
        - Patterns: {profile.behavioral_patterns}
        
        Generate {content_type} content considering:
        {json.dumps(context, indent=2)}
        """
    
    def _generate_cache_key(self, profile: UserProfile, content_type: str, context: Dict[str, Any]) -> str:
        """Generate cache key for content"""
        data = f"{profile.user_id}_{profile.version}_{content_type}_{hash(str(context))}"
        return hashlib.md5(data.encode()).hexdigest()


class ABTestManager:
    """A/B testing framework with segment-aware testing"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    
    def create_experiment(self, 
                         experiment_id: str,
                         variants: List[str],
                         target_segments: List[str],
                         traffic_split: Dict[str, float]):
        """Create new A/B test experiment"""
        self.experiments[experiment_id] = {
            'variants': variants,
            'target_segments': target_segments,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'active': True
        }
    
    def assign_variant(self, user_id: str, experiment_id: str, user_segments: List[str]) -> Optional[str]:
        """Assign user to experiment variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        if not experiment['active']:
            return None
        
        # Check if user is in target segments
        if not any(segment in experiment['target_segments'] for segment in user_segments):
            return None
        
        # Consistent assignment based on user_id hash
        hash_input = f"{user_id}_{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Determine variant based on traffic split
        cumulative = 0.0
        normalized_hash = (hash_value % 1000) / 1000.0
        
        for variant, split in experiment['traffic_split'].items():
            cumulative += split
            if normalized_hash <= cumulative:
                self.assignments[user_id][experiment_id] = variant
                return variant
        
        return experiment['variants'][0]  # Default variant


class PersonalizationEngine:
    """Core personalization engine with real-time processing"""
    
    def __init__(self, data_store: DataStore, llm_adapter: LLMAdapter, 
                 ab_test_manager: ABTestManager, security_manager: SecurityManager):
        self.data_store = data_store
        self.llm_adapter = llm_adapter
        self.ab_test_manager = ab_test_manager
        self.security_manager = security_manager
        self.event_queue = asyncio.Queue()
        self.profile_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def process_event(self, event: UserEvent) -> bool:
        """Process incoming user event"""
        try:
            # Store event
            await self.data_store.store_event(event)
            
            # Update user profile
            await self._update_user_profile(event)
            
            # Queue for further processing
            await self.event_queue.put(event)
            
            return True
        except Exception as e:
            logging.error(f"Error processing event: {e}")
            return False
    
    async def get_personalized_content(self, 
                                     user_id: str,
                                     content_type: str,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized content for user"""
        try:
            profile = await self._get_cached_profile(user_id)
            if not profile:
                profile = await self._create_default_profile(user_id)
            
            # Get A/B test variants
            experiments = {}
            for exp_id in context.get('experiments', []):
                variant = self.ab_test_manager.assign_variant(user_id, exp_id, profile.segments)
                if variant:
                    experiments[exp_id] = variant
            
            # Generate content
            content = await self.llm_adapter.generate_personalized_content(
                profile, content_type, {**context, 'experiments': experiments}
            )
            
            return {
                'content': content,
                'profile_version': profile.version,
                'experiments': experiments,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logging.error(f"Error generating personalized content: {e}")
            return {'error': str(e)}
    
    async def _update_user_profile(self, event: UserEvent):
        """Update user profile based on event"""
        profile = await self._get_cached_profile(event.user_id)
        if not profile:
            profile = await self._create_default_profile(event.user_id)
        
        # Update based on event type
        if event.event_type == EventType.CLICK:
            category = event.data.get('category', 'general')
            profile.update_preference(f"click_{category}", 1.0, weight=0.2)
        
        elif event.event_type == EventType.PURCHASE:
            price = event.data.get('price', 0)
            profile.update_preference('purchase_intent', 1.0, weight=0.3)
            profile.update_preference('price_sensitivity', 1.0 / (1.0 + price / 100), weight=0.1)
        
        elif event.event_type == EventType.CONTENT_VIEW:
            duration = event.data.get('duration', 0)
            engagement = min(1.0, duration / 300)  # Normalize to 5 minutes
            content_type = event.data.get('content_type', 'general')
            profile.update_preference(f"content_{content_type}", engagement, weight=0.15)
        
        # Update segments based on preferences
        self._update_user_segments(profile)
        
        # Save updated profile
        await self.data_store.save_user_profile(profile)
        self._cache_profile(profile)
    
    def _update_user_segments(self, profile: UserProfile):
        """Update user segments based on preferences and behavior"""
        segments = []
        
        # Engagement-based segments
        avg_engagement = sum(
            score for key, score in profile.preferences.items() 
            if key.startswith('content_')
        ) / max(1, len([k for k in profile.preferences.keys() if k.startswith('content_')]))
        
        if avg_engagement > 0.7:
            segments.append('high_engagement')
        elif avg_engagement > 0.3:
            segments.append('medium_engagement')
        else:
            segments.append('low_engagement')
        
        # Purchase intent segments
        if profile.preferences.get('purchase_intent', 0) > 0.5:
            segments.append('buyer')
        
        # Price sensitivity
        if profile.preferences.get('price_sensitivity', 0.5) > 0.7:
            segments.append('price_sensitive')
        
        profile.segments = segments
    
    async def _get_cached_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from cache or data store"""
        cache_key = f"profile_{user_id}"
        
        if cache_key in self.profile_cache:
            cached_profile, timestamp = self.profile_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_profile
        
        profile = await self.data_store.get_user_profile(user_id)
        if profile:
            self._cache_profile(profile)
        
        return profile
    
    def _cache_profile(self, profile: UserProfile):
        """Cache user profile"""
        cache_key = f"profile_{profile.user_id}"
        self.profile_cache[cache_key] = (profile, time.time())
    
    async def _create_default_profile(self, user_id: str) -> UserProfile:
        """Create default user profile"""
        profile = UserProfile(
            user_id=user_id,
            preferences={},
            segments=['new_user'],
            behavioral_patterns={},
            last_updated=datetime.now()
        )
        await self.data_store.save_user_profile(profile)
        return profile


class PersonalizationAPI:
    """REST API for personalization system"""
    
    def __init__(self, engine: PersonalizationEngine, security_manager: SecurityManager):
        self.engine = engine
        self.security = security_manager
    
    def auth_required(self, permissions: List[str] = None):
        """Authentication decorator"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract token from request (simplified)
                token = kwargs.get('token')
                if not token:
                    return {'error': 'Authorization required', 'status': 401}
                
                payload = self.security.validate_token(token)
                if not payload:
                    return {'error': 'Invalid token', 'status': 401}
                
                if permissions:
                    user_permissions = payload.get('permissions', [])
                    if not any(perm in user_permissions for perm in permissions):
                        return {'error': 'Insufficient permissions', 'status': 403}
                
                kwargs['user_payload'] = payload
                return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    @auth_required(['read_content'])
    async def get_personalized_content(self, 
                                     user_id: str,
                                     content_type: str,
                                     context: Dict[str, Any],
                                     token: str = None,
                                     user_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for personalized content"""
        if not self.security.rate_limit(user_id):
            return {'error': 'Rate limit exceeded', 'status': 429}
        
        return await self.engine.get_personalized_content(user_id, content_type, context)
    
    @auth_required(['write_events'])
    async def track_event(self,
                         event_data: Dict[str, Any],
                         token: str = None,
                         user_payload: Dict[str, Any] = None) -> Dict[str, Any]:
        """API endpoint for event tracking"""
        try:
            event = UserEvent(
                user_id=event_data['user_id'],
                event_type=EventType(event_data['event_type']),
                timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
                data=event_data['data'],
                session_id=event_data['session_id'],
                security_level=SecurityLevel(event_data.get('security_level', 'internal'))
            )
            
            success = await self.engine.process_event(event)
            return {'success': success, 'status': 200 if success else 500}
        
        except Exception as e:
            return {'error': str(e), 'status': 400}


class SystemHealthMonitor:
    """System health and performance monitoring"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
    
    def record_metric(self, name: str, value: float, timestamp: datetime = None):
        """Record system metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.metrics[name].append((timestamp, value))
        
        # Keep only last 1000 metrics per type
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        summary = {}
        for name, values in self.metrics.items():
            recent_values = [v for t, v in values if datetime.now() - t < timedelta(minutes=5)]
            if recent_values:
                summary[name] = {
                    'avg': sum(recent_values) / len(recent_values),
                    'min': min(recent_values),
                    'max': max(recent_values),
                    'count': len(recent_values)
                }
        return summary


# Advanced Testing Framework
class TestScenario:
    """Test scenario for AI adaptability"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.test_steps = []
        self.expected_outcomes = []
    
    def add_step(self, action: Callable, expected_result: Any):
        """Add test step"""
        self.test_steps.append((action, expected_result))
    
    async def run(self) -> Dict[str, Any]:
        """Run test scenario"""
        results = {'name': self.name, 'passed': True, 'details': []}
        
        for i, (action, expected) in enumerate(self.test_steps):
            try:
                result = await action() if asyncio.iscoroutinefunction(action) else action()
                passed = self._compare_results(result, expected)
                
                results['details'].append({
                    'step': i + 1,
                    'passed': passed,
                    'expected': expected,
                    'actual': result
                })
                
                if not passed:
                    results['passed'] = False
            
            except Exception as e:
                results['passed'] = False
                results['details'].append({
                    'step': i + 1,
                    'passed': False,
                    'error': str(e)
                })
        
        return results
    
    def _compare_results(self, actual: Any, expected: Any) -> bool:
        """Compare test results"""
        if callable(expected):
            return expected(actual)
        return actual == expected


# Example usage and system initialization
async def main():
    """Main system initialization and example usage"""
    # Initialize components
    security = SecurityManager("your-secret-key-here")
    data_store = InMemoryDataStore()
    llm_adapter = LLMAdapter()
    ab_test_manager = ABTestManager()
    
    # Create personalization engine
    engine = PersonalizationEngine(data_store, llm_adapter, ab_test_manager, security)
    
    # Create API
    api = PersonalizationAPI(engine, security)
    
    # Initialize monitoring
    monitor = SystemHealthMonitor()
    
    # Example: Create A/B test
    ab_test_manager.create_experiment(
        'homepage_layout',
        ['variant_a', 'variant_b'],
        ['new_user', 'returning_user'],
        {'variant_a': 0.5, 'variant_b': 0.5}
    )
    
    # Example: Process user events
    test_user_id = "user_123"
    
    # Generate token for user
    token = security.generate_token(test_user_id, ['read_content', 'write_events'])
    
    # Track events
    events = [
        {
            'user_id': test_user_id,
            'event_type': 'page_view',
            'session_id': 'session_456',
            'data': {'page': '/homepage', 'duration': 45},
            'timestamp': datetime.now().isoformat()
        },
        {
            'user_id': test_user_id,
            'event_type': 'click',
            'session_id': 'session_456',
            'data': {'element': 'product_card', 'category': 'electronics'},
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    for event_data in events:
        result = await api.track_event(event_data, token=token)
        print(f"Event tracking result: {result}")
    
    # Get personalized content
    content_result = await api.get_personalized_content(
        test_user_id,
        'product_recommendations',
        {
            'category': 'electronics',
            'experiments': ['homepage_layout'],
            'context': 'homepage'
        },
        token=token
    )
    
    print(f"Personalized content: {content_result}")
    
    # Health monitoring
    monitor.record_metric('response_time', 0.150)
    monitor.record_metric('cache_hit_rate', 0.85)
    
    print(f"System metrics: {monitor.get_metrics_summary()}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the system
    asyncio.run(main())