"""
ConversationMemory module for storing and managing multi-turn conversation history.

This module provides functionality to:
- Store conversation history with user and agent messages
- Retrieve recent conversation context
- Detect contradictions in factual user statements
"""

from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
import re


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'agent'
    text: str
    timestamp: datetime = field(default_factory=datetime.now)
    facts: Dict[str, Any] = field(default_factory=dict)  # Extracted facts from this message


@dataclass
class Fact:
    """Represents a factual statement extracted from user messages."""
    key: str  # e.g., 'income', 'age', 'location'
    value: Any
    message_index: int  # Index in conversation history
    timestamp: datetime


class ConversationMemory:
    """
    Manages conversation history and detects contradictions in user statements.
    
    This class stores multi-turn conversations in memory and provides methods
    to add messages, retrieve context, and detect contradictions in factual
    statements like income, age, location, etc.
    
    Attributes:
        messages: List of Message objects representing the conversation history
        facts: Dictionary mapping fact keys to Fact objects
        fact_patterns: Dictionary of regex patterns for extracting common facts
    """
    
    def __init__(self):
        """Initialize an empty conversation memory."""
        self.messages: List[Message] = []
        self.facts: Dict[str, Fact] = {}
        
        # Patterns for extracting common factual information
        self.fact_patterns: Dict[str, List[re.Pattern]] = {
            'income': [
                re.compile(r'(?:income|salary|earn|earning).*?(?:is|of|₹|rs\.?|rupees?)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:lakh|lac|thousand|k|crore)?', re.IGNORECASE),
                re.compile(r'(?:₹|rs\.?|rupees?)\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:lakh|lac|thousand|k|crore)?.*?(?:income|salary|earn)', re.IGNORECASE),
            ],
            'age': [
                re.compile(r'(?:age|aged|old).*?(?:is|of|am|years?)\s*(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)\s*(?:years?\s*old|years?\s*of\s*age)', re.IGNORECASE),
            ],
            'location': [
                re.compile(r'(?:from|live|located|reside|staying).*?in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
                re.compile(r'(?:city|state|place).*?(?:is|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', re.IGNORECASE),
            ],
            'occupation': [
                re.compile(r'(?:work|job|profession|occupation).*?(?:as|is|a|an)\s+([a-z]+(?:\s+[a-z]+)*)', re.IGNORECASE),
                re.compile(r'(?:am|is|are)\s+(?:a|an)\s+([a-z]+(?:\s+[a-z]+)*)', re.IGNORECASE),
            ],
        }
    
    def add_user_message(self, text: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            text: The user's message text
        """
        message = Message(role='user', text=text)
        message.facts = self._extract_facts(text, len(self.messages))
        self.messages.append(message)
        
        # Update facts dictionary with newly extracted facts
        for key, value in message.facts.items():
            self.facts[key] = Fact(
                key=key,
                value=value,
                message_index=len(self.messages) - 1,
                timestamp=message.timestamp
            )
    
    def add_agent_message(self, text: str) -> None:
        """
        Add an agent message to the conversation history.
        
        Args:
            text: The agent's message text
        """
        message = Message(role='agent', text=text)
        self.messages.append(message)
    
    def detect_contradiction(self, new_fact: Dict[str, Any]) -> Optional[Tuple[str, Fact, Any]]:
        """
        Detect if a new fact contradicts previously stated facts.
        
        Args:
            new_fact: Dictionary with fact key-value pairs to check
                Example: {'income': 50000, 'age': 30}
        
        Returns:
            Tuple of (contradicted_key, existing_fact, new_value) if contradiction found,
            None otherwise
        """
        for key, new_value in new_fact.items():
            if key in self.facts:
                existing_fact = self.facts[key]
                
                # Normalize values for comparison
                normalized_existing = self._normalize_value(existing_fact.value)
                normalized_new = self._normalize_value(new_value)
                
                # Check for contradiction
                if self._is_contradiction(normalized_existing, normalized_new):
                    return (key, existing_fact, new_value)
        
        return None
    
    def get_context(self, last_n_turns: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve the last N conversation turns as context.
        
        Args:
            last_n_turns: Number of recent messages to retrieve (default: 10)
        
        Returns:
            List of dictionaries with 'role' and 'text' keys representing
            the conversation context
        """
        recent_messages = self.messages[-last_n_turns:] if len(self.messages) > last_n_turns else self.messages
        
        return [
            {
                'role': msg.role,
                'text': msg.text,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in recent_messages
        ]
    
    def _extract_facts(self, text: str, message_index: int) -> Dict[str, Any]:
        """
        Extract factual information from user message text.
        
        Args:
            text: The message text to analyze
            message_index: Index of this message in conversation history
        
        Returns:
            Dictionary of extracted facts (key-value pairs)
        """
        facts = {}
        
        for fact_key, patterns in self.fact_patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    value = match.group(1)
                    # Normalize and store the value
                    facts[fact_key] = self._normalize_value(value)
                    break  # Use first match found
        
        return facts
    
    def _normalize_value(self, value: Any) -> Any:
        """
        Normalize fact values for comparison.
        
        Args:
            value: The value to normalize
        
        Returns:
            Normalized value
        """
        if isinstance(value, str):
            # Remove common formatting
            value = value.strip().lower()
            
            # Handle numeric strings with units
            if 'lakh' in value or 'lac' in value:
                numeric_part = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', value)
                if numeric_part:
                    return float(numeric_part.group(1).replace(',', '')) * 100000
            elif 'crore' in value:
                numeric_part = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', value)
                if numeric_part:
                    return float(numeric_part.group(1).replace(',', '')) * 10000000
            elif 'thousand' in value or 'k' in value:
                numeric_part = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', value)
                if numeric_part:
                    return float(numeric_part.group(1).replace(',', '')) * 1000
            
            # Try to convert to number if possible
            numeric_match = re.search(r'(\d+(?:,\d{3})*(?:\.\d+)?)', value)
            if numeric_match:
                try:
                    return float(numeric_match.group(1).replace(',', ''))
                except ValueError:
                    pass
        
        return value
    
    def _is_contradiction(self, existing_value: Any, new_value: Any) -> bool:
        """
        Determine if two values contradict each other.
        
        Args:
            existing_value: Previously stated value
            new_value: Newly stated value
        
        Returns:
            True if values contradict, False otherwise
        """
        # If both are numeric, check if they differ significantly (>10% difference)
        if isinstance(existing_value, (int, float)) and isinstance(new_value, (int, float)):
            if existing_value == 0:
                return new_value != 0
            difference_ratio = abs(existing_value - new_value) / abs(existing_value)
            return difference_ratio > 0.1  # More than 10% difference
        
        # For strings, check exact match (case-insensitive)
        if isinstance(existing_value, str) and isinstance(new_value, str):
            return existing_value.lower() != new_value.lower()
        
        # Type mismatch might indicate contradiction
        if type(existing_value) != type(new_value):
            return True
        
        # Default: exact match check
        return existing_value != new_value
    
    def get_all_facts(self) -> Dict[str, Fact]:
        """
        Get all stored facts from the conversation.
        
        Returns:
            Dictionary mapping fact keys to Fact objects
        """
        return self.facts.copy()
    
    def clear(self) -> None:
        """Clear all conversation history and facts."""
        self.messages.clear()
        self.facts.clear()

