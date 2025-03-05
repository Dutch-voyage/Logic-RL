def get_stop_criteria(stop_criteria):
    if stop_criteria == "repetition":
        from .repetition import RepetitionCheck
        return RepetitionCheck()
    else:
        raise ValueError(f"Stop criteria {stop_criteria} not found")
