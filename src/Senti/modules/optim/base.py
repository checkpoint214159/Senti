from abc import ABC


class BaseLoss(ABC):
    """
    ABC to internalize loss calling and logging.
    Main flow is that user can provide all the required losses he wants to calculate, we cache them
    And only when he calls compute, we execute all loss calls
    Once done, call clear cache.
    """

    def __init__(self):
        self.types = []
        self.calls = []

    def supports_type(self, t) -> bool:
        raise NotImplementedError(f"{type(self)} did not implement supports_type().")
    
    def include(self, t: tuple):
        return self.calls.append(t)

    def compute(self):
        all = 0
        for call in self.calls:
            log, func, a, b = call  # TODO log somewhere
            result = a.compute_energy(b, func)
            # print('log:', log, 'result', result)
            # if 'h' in log:
            #     print('log:', log, )
            #     print('func', func,)
            #     print('result', result)
            #     print('a.mean?', a.mean, 'b.mean?', b.mean)
            all += result

        self.calls.clear()
        return all
