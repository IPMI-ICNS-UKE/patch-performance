class NoPatchesError(RuntimeError):
    def __init__(self, patch_performance):
        message = (
            f"Cannot calculate performance. "
            f"{patch_performance.__class__.__name__} "
            f"never has been called!"
        )
        super().__init__(message)
