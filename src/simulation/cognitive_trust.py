"""
Module: cognitive_trust.py
Purpose: Implements the Cognitive Trust Metric (CTM) and Shadow IT decisions.
Author: CoderReproAgent (via Antigravity)
Date: 2026-01-11
Dependencies: dataclasses, typing

Description:
    This module implements the mathematical model for Cognitive Trust Dynamics.
    It simulates how a clinician's trust in the security system evolves over time
    based on their interactions (Success, Friction, Explainability).
"""

from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrustConfig:
    alpha: float = 0.1  # Weight for Explainability
    beta: float  = 0.2  # Weight for Successful Validation
    gamma: float = 0.3  # Penalty for Friction/Latency
    delta: float = 0.5  # Penalty for False Positives (Blocking legit access)
    decay: float = 0.95 # Natural trust decay over time (forgetting)


class CognitiveTrustMetric:
    """
    Models the dynamic trust state T(t) of a clinician.
    """
    def __init__(self, initial_trust: float = 0.5, config: TrustConfig = TrustConfig()):
        self.trust = initial_trust
        self.config = config
        self.history = [initial_trust]

    def update(self, explainability: float, is_valid_success: bool, friction_seconds: float, is_false_positive: bool):
        """
        Update trust based on a single interaction event t -> t+1.
        
        Args:
            explainability: 0.0 to 1.0 (How well system explained why it worked/failed)
            is_valid_success: True if the ZK-Twin was useful/valid.
            friction_seconds: Latency experienced (seconds).
            is_false_positive: True if legitimate access was wrongly blocked.
        """
        # Calculate delta T
        delta_t = 0.0
        
        # + Trust Factors
        delta_t += self.config.alpha * explainability
        if is_valid_success:
            delta_t += self.config.beta
            
        # - Distrust Factors
        # Normalize friction (assume >5s is bad)
        norm_friction = min(friction_seconds / 5.0, 1.0) 
        delta_t -= self.config.gamma * norm_friction
        
        if is_false_positive:
            delta_t -= self.config.delta

        # Update State with limits [0, 1]
        self.trust = self.trust * self.config.decay + delta_t
        self.trust = max(0.0, min(1.0, self.trust))
        
        self.history.append(self.trust)
        logger.debug(f"Trust updated: {self.trust:.4f} (Delta: {delta_t:.4f})")

    def decide_action(self, urgency: float) -> str:
        """
        Decide whether to use Secure System or Shadow IT.
        Decision Logic:
            If Trust >= Urgency: Use Secure System
            If Trust < Urgency:  Use Shadow IT (Bypass)
            
        Args:
            urgency: 0.0 to 1.0 (Clinical urgency of the request)
            
        Returns:
            "SECURE" or "SHADOW_IT"
        """
        threshold = urgency
        if self.trust >= threshold:
            return "SECURE"
        else:
            return "SHADOW_IT"

if __name__ == "__main__":
    # Test Scenario
    ctm = CognitiveTrustMetric(initial_trust=0.6)
    
    print(f"Initial Trust: {ctm.trust}")
    
    # Event 1: Good system behavior (Fast, Valid, Explained)
    ctm.update(explainability=0.8, is_valid_success=True, friction_seconds=0.5, is_false_positive=False)
    print(f"After Success: {ctm.trust} -> Action (Urg=0.8): {ctm.decide_action(0.8)}")
    
    # Event 2: Bad system behavior (Slow, False Positive)
    ctm.update(explainability=0.0, is_valid_success=False, friction_seconds=6.0, is_false_positive=True)
    print(f"After Failure: {ctm.trust} -> Action (Urg=0.8): {ctm.decide_action(0.8)}")
