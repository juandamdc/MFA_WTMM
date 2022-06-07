from interval import interval

class CantorSet:
    def __init__(self, p = 0.5) -> None:
        self._levels = {}
        self.p = p

        self._levels[0] = ([interval([0,1])], [1])


    def _getLevel(self, level):
        if level in self._levels.keys():
            return self._levels[level]
        else:
            prev_level_int, prev_level_msr = self._getLevel(level - 1)
            self._levels[level] = self._makeNextLevel(prev_level_int, prev_level_msr)
            return self._levels[level]


    def _makeNextLevel(self, intervals, intervals_measures):
        next_intervals = list()
        next_intervals_measure = list()

        for intr, measure in zip(intervals, intervals_measures):
            left = interval([intr[0].inf, intr[0].inf + 1/3 * (intr[0].sup - intr[0].inf)])
            left_measure = measure * self.p

            right = interval([intr[0].inf + 2/3 * (intr[0].sup - intr[0].inf), intr[0].sup])
            right_measure = measure * (1 - self.p)

            next_intervals.append(left)
            next_intervals.append(right)

            next_intervals_measure.append(left_measure)
            next_intervals_measure.append(right_measure)

        return (next_intervals, next_intervals_measure)


    def eval(self, level, point):
        intervals, measures = self._getLevel(level)

        for interval, measure in zip(intervals, measures):
            if point in interval:
                return measure
            elif point < interval[0].inf:
                break 
        
        return 0


    def printLevel(self, level):
        intrs, measures = self._getLevel(level)

        if self.p == 0.5:
            toPrint = ', '.join([f'{intr}' for intr in intrs])
        else:
            toPrint = ', '.join([f'{intr} -- {measure}' for intr, measure in zip(intrs, measures)])
        
        print(f'Level: {level}, Intervals: {toPrint}')



if __name__=='__main__':
    cs = CantorSet()
    cs.printLevel(3)
