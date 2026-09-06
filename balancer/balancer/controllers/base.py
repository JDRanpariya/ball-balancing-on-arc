"""Base controller interface for ball-balancer."""

from abc import ABC, abstractmethod

class BaseController(ABC):
    """
    Abstract base class for all ball-balancer controllers.
    
    Defines the interface that all control algorithms must implement. Controllers
    receive the current state and compute control actions to balance the ball.
    
    Attributes:
        action_type (str): "discrete" or "cont" action space
        model_name (str): Identifier for logging and tracking
        
    Example:
        >>> class MyController(BaseController):
        ...     def step(self, state, cart_limit):
        ...         return 0.0  # control action
        ...     def reset(self):
        ...         pass
    """
    
    @abstractmethod
    def step(self, state, cart_limit):
        """
        Compute control action for current state.
        
        Args:
            state (np.ndarray): Current state [cart_pos, cart_vel, ball_pos, ball_vel]
            cart_limit (float): Maximum cart position limit for boundary handling
            
        Returns:
            float: Control action (discrete in {-1,0,1} or continuous in [[-1, 1])
            
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        """
        Reset controller internal state.
        
        Called at the beginning of each episode to clear any accumulated
        state (e.g., integral terms, trajectory history).
        """
        raise NotImplementedError

