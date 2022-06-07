import numpy as np


class DisjointSet:
    def __init__(self, elements):
        self.elements = elements
        self.cant = dict()
        self.parents = dict()

        for element in elements:
            self.parents[element] = element
            self.cant[element] = 1
        
    def getParent(self, element):
        if self.parents[element] == element:
            return element
        else:
            return self.getParent(self.parents[element])

    def joint(self, elementA, elementB):
        parent1 = self.getParent(elementA)
        parent2 = self.getParent(elementB)

        if parent1 != parent2:
            if self.cant[parent1] >= self.cant[parent2]:
                self.parents[parent2] = parent1
                self.cant[parent1] += self.cant[parent2]
            else:
                self.parents[parent1] = parent2
                self.cant[parent2] += self.cant[parent1]