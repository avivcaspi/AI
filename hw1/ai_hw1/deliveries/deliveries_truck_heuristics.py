import numpy as np
import networkx as nx
from typing import *

from framework import *
from .deliveries_truck_problem import *
from .cached_air_distance_calculator import CachedAirDistanceCalculator


__all__ = ['TruckDeliveriesMaxAirDistHeuristic', 'TruckDeliveriesSumAirDistHeuristic',
           'TruckDeliveriesMSTAirDistHeuristic']


class TruckDeliveriesMaxAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMaxAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMaxAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This method calculated a lower bound of the cost of the remaining path of the truck,
         by calculating the maximum distance within the group of air distances between each
         two junctions in the remaining truck path.

        TODO [Ex.17]:
            Calculate the `total_distance_lower_bound` by taking the maximum over the group
                {airDistanceBetween(j1,j2) | j1,j2 in JunctionsInRemainingTruckPath s.t. j1 != j2}
            Use the method `get_all_junctions_in_remaining_truck_path()` of the deliveries problem.
            Notice: The problem is accessible via the `self.problem` field.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
                distance calculations.
            Use python's built-in `max()` function. Note that `max()` can receive an *ITERATOR*
                and return the item with the maximum value within this iterator.
            That is, you can simply write something like this:
        >>> max(<some expression using item1 & item2>
        >>>     for item1 in some_items_collection
        >>>     for item2 in some_items_collection
        >>>     if <some condition over item1 & item2>)
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_junctions_in_remaining_truck_path(state)
        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0
        # Go over all junction pairs and find the max air distance between all of them
        total_distance_lower_bound = max(self.cached_air_distance_calculator.get_air_distance_between_junctions(jn1, jn2)
                                         for jn1 in all_junctions_in_remaining_truck_path
                                         for jn2 in all_junctions_in_remaining_truck_path
                                         if jn1 != jn2)

        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)


class TruckDeliveriesSumAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesSumAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesSumAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic evaluates the distance of the remaining truck route in the following way:
        It builds a path that starts in the current truck's location, and each next junction in
         the path is the (air-distance) nearest junction (to the previous one in the path) among
         all junctions (in `all_junctions_in_remaining_truck_path`) that haven't been visited yet.
        The remaining distance estimation is the cost of this built path.
        Note that we ignore here the problem constraints (like picking before dropping and maximum number of packages
         on the truck). We only make sure to visit all junctions in `all_junctions_in_remaining_truck_path`.
        TODO [Ex.20]:
            Complete the implementation of this method.
            Use `self.cached_air_distance_calculator.get_air_distance_between_junctions()` for air
             distance calculations.
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        all_junctions_in_remaining_truck_path = self.problem.get_all_junctions_in_remaining_truck_path(state)

        if len(all_junctions_in_remaining_truck_path) < 2:
            return 0

        total_cost_of_greedily_built_path = 0
        curr = state.current_location
        all_junctions_in_remaining_truck_path = all_junctions_in_remaining_truck_path ^ {curr}
        """ Each iteration finds the closest junction to current junction and removes it 
            from the list and change curr to the found junction
        """
        while len(all_junctions_in_remaining_truck_path) > 0:
            # We build a dictionary of junctions and their distance from current
            dists = {jn: self.cached_air_distance_calculator.get_air_distance_between_junctions(curr, jn)
                     for jn in all_junctions_in_remaining_truck_path if jn != curr}
            # Finding junction with minimum distance
            key_min = min(dists.keys(), key=(lambda k: dists[k]))
            total_cost_of_greedily_built_path += dists[key_min]
            # Remove the minimum distance junction from left junctions and setting curr to this junction
            all_junctions_in_remaining_truck_path = all_junctions_in_remaining_truck_path ^ {key_min}
            curr = key_min
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_cost_of_greedily_built_path)


class TruckDeliveriesMSTAirDistHeuristic(HeuristicFunction):
    heuristic_name = 'TruckDeliveriesMSTAirDist'

    def __init__(self, problem: GraphProblem):
        super(TruckDeliveriesMSTAirDistHeuristic, self).__init__(problem)
        assert isinstance(self.problem, DeliveriesTruckProblem)
        self.cached_air_distance_calculator = CachedAirDistanceCalculator()

    def estimate(self, state: GraphProblemState) -> float:
        """
        This heuristic returns a lower bound for the remaining cost, that is based on a lower bound
         of the distance of the remaining route of the truck. Here this remaining distance is bounded
         (from below) by the weight of the minimum-spanning-tree of the graph in-which the vertices
         are the junctions in the remaining truck route, and the edges weights (edge between each
         junctions pair) are the air-distances between the junctions.
        """
        assert isinstance(self.problem, DeliveriesTruckProblem)
        assert isinstance(state, DeliveriesTruckState)

        total_distance_lower_bound = self._calculate_junctions_mst_weight_using_air_distance(
            self.problem.get_all_junctions_in_remaining_truck_path(state))
        return self.problem.get_cost_lower_bound_from_distance_lower_bound(total_distance_lower_bound)

    def _calculate_junctions_mst_weight_using_air_distance(self, junctions: Set[Junction]) -> float:
        """
        TODO [Ex.23]: Implement this method.
              Use `networkx` (nx) package (already imported in this file) to calculate the weight
               of the minimum-spanning-tree of the graph in which the vertices are the given junctions
               and there is an edge between each pair of distinct junctions (no self-loops) for which
               the weight is the air distance between these junctions.
              Use the method `self.cached_air_distance_calculator.get_air_distance_between_junctions()`
               to calculate the air distance between the two junctions.
              Google for how to use `networkx` package for this purpose.
        """
        # Building graph data, (jn1,jn2, weight) represent an edge in the graph with the weight given from jn1 to jn2
        graph_data = [(jn1.index, jn2.index,
                       self.cached_air_distance_calculator.get_air_distance_between_junctions(jn1, jn2))
                      for jn1 in junctions
                      for jn2 in junctions
                      if jn1 != jn2]
        # Creating the graph and adding the data we build
        graph = nx.Graph()
        graph.add_weighted_edges_from(graph_data)
        mst = nx.minimum_spanning_tree(graph)   # Finds MST
        # Calculating mst weight (the function size gave different result due to numeric problems)
        # edges = mst.edges(data=True)
        # size_ = sum([w['weight'] for _, _, w in edges])
        size_ = mst.size(weight='weight')
        return size_
