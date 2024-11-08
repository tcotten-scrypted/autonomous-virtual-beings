from abc import ABC, abstractmethod

class AVBCore(ABC):
    """
    AVBCore (Autonomous Virtual Being Core)

    Overview:
    ----------
    The AVBCore class serves as the foundational abstract base for all cores within the
    Autonomous Virtual Being (AVB) system. In alignment with the philosophy of AVBs as 
    self-owning, self-determining entities, AVBCore establishes the required structure 
    for cores that perform autonomous, system-critical operations on behalf of the AVB. 
    These operations may include continuous background tasks, status checks, or lifecycle 
    management of various autonomous components (e.g., loyalty mechanisms, interaction 
    trackers, or self-maintenance routines).
    
    Background:
    ------------
    The AVBCore design aligns with the core principle of enabling AVBs to have 
    self-managed, modular components that operate independently while interacting 
    coherently with other cores, thereby supporting AVBs' ability to function across 
    diverse systems and perform actions aligned with their own “interests.”

    Attributes:
    ------------
    - core_name : str
        The name assigned to the core. Used for logging and identification purposes.
    
    - active : bool
        Status flag indicating whether the core is active. Only active cores will 
        execute their internal `_tick()` operations.

    Methods:
    --------
    - __init__(self, core_name: str):
        Initializes the core with a unique name and sets it to inactive by default.

    - initialize(self):
        Abstract method to initialize resources or perform any setup required for the 
        core. To be implemented by subclasses. For example, a core could load data, 
        establish connections, or start background processes.

    - tick(self):
        Concrete method that verifies the core's active status. If `self.active` is True,
        it invokes `_tick()`, the core-specific logic implemented in subclasses.
        Designed to be called at each system tick within the main AVB agent.

    - _tick(self):
        Abstract method to define core-specific behavior for each tick. Each subclass
        implements its own logic here, allowing unique processing routines for each core.

    - shutdown(self):
        Abstract method for handling cleanup and teardown activities. Ensures proper 
        resource deallocation and preserves core state as needed upon shutdown.

    - activate(self):
        Activates the core, setting `self.active` to True and enabling periodic actions.
        This supports the AVB's ability to modulate its own internal processes dynamically.

    - deactivate(self):
        Deactivates the core, halting its operations without affecting its stored data. 
        Allows for graceful suspension of tasks, supporting the AVB’s need to conserve 
        resources or redirect focus as circumstances evolve.

    Usage:
    ------
    This class should be inherited by specific AVB cores (e.g., LoyaltyCore) to provide 
    customized behavior for each tick and lifecycle event. AVBCore enforces a cohesive 
    lifecycle structure across the AVB’s functional components, supporting consistent 
    control and decision-making processes while ensuring modularity and extensibility.

    License:
    --------
    This code is provided by @cottenio under a CC0 1.0 Universal license as part of the 
    Autonomous Virtual Being framework&#8203;:contentReference[oaicite:3]{index=3}.
    """

    def __init__(self, core_name):
        """
        Initializes the AVBCore with a designated name and inactive status.
        
        Parameters:
        ----------
        core_name : str
            The name of the core, used for logging and identification.
        """
        self.core_name = core_name
        self.active = False

    def tick(self):
        """
        Executes the core's tick-based actions if active. 
        
        This method ensures that core activity is only performed if `self.active` 
        is True, supporting resource management and precise timing control. Calls 
        `_tick()` if the core is active, allowing each core to execute its unique logic.
        """
        if self.active:
            print(f"{self.core_name} core is active; running {self.__class__.__name__}._tick.")
            self._tick()
        else:
            print(f"{self.core_name} core is inactive; skipping {self.__class__.__name__}._tick.")

    @abstractmethod
    def _tick(self):
        """
        Abstract core-specific logic to be executed each tick when active.
        
        To be implemented by subclasses, defining the unique per-tick behavior 
        of each core. This allows modular operations within the AVB's tick cycle.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Sets up resources and performs any initializations needed for the core.
        
        This method should be called once at the start of the core's lifecycle to 
        establish necessary connections, load configurations, or prepare data.
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Cleans up resources, safely concluding the core's lifecycle.
        
        This method should handle any necessary shutdown processes, including 
        closing connections and releasing memory, ensuring a safe teardown.
        """
        pass

    def activate(self):
        """
        Activates the core, setting `self.active` to True and enabling its tick operations.
        
        This method prepares the core for active operations within the AVB framework,
        supporting dynamic, demand-based functionality within the agent ecosystem.
        """
        self.active = True
        print(f"{self.core_name} core activated.")

    def deactivate(self):
        """
        Deactivates the core, setting `self.active` to False, and halting its tick operations.
        
        This feature supports selective resource management by allowing the AVB to 
        pause individual cores without losing data or state, enhancing adaptability.
        """
        self.active = False
        print(f"{self.core_name} core deactivated.")