class AVBCoreHeartbeatError(RuntimeError):
    """Raised when a heartbeat file already exists and the agent cannot start."""
    pass

class AVBCoreRegistryFileError(RuntimeError):
    """Raised when there is an issue with the core registry file, such as missing or invalid JSON."""
    pass

class AVBCoreLoadingError(RuntimeError):
    """Raised when a core cannot be imported or initialized properly."""
    pass
