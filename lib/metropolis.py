import math

def metropolisAcceptanceCriterion(iteration: int, new_score: float, old_score: float, initial_temperature: float=1)-> float:
    x = -1 * (iteration/initial_temperature) *((old_score-new_score)/old_score)
    return math.exp(x)


