class ModelNeededDecorators:
    """
    Decorator. This enforces a specified Variable to be defined, before a Function is run
    Example:
        @PARAMETER_NEEDED("target_model")
        def do_sth(self, a, b):
            return a+b
    """
    @staticmethod
    def PARAMETER_NEEDED(parameter):
        def ENFORCE_PARAMETER(func):
            def wrap_func(self, *args, **kwargs):
                if parameter in locals() or getattr(self, parameter, None) is not None:
                    return func(self, *args, **kwargs)
                raise ValueError(f"Parameter {parameter} is needed, but not loaded! Check your Parameters.")

            return wrap_func
        return ENFORCE_PARAMETER
