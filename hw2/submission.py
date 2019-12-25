from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
import time


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    closest_apple = _find_nearest_apple_dist(state, player_index)
    if closest_apple > 0:
        return state.snakes[player_index].length + 1 / closest_apple
    return state.snakes[player_index].length


def _find_nearest_apple_dist(state: GameState, player_index: int) -> float:
    head = state.snakes[player_index].head
    dists = [np.linalg.norm(np.array(head) - np.array(apple)) for apple in state.fruits_locations]
    if len(dists) > 0:
        return np.min(dists)
    return 0


def _find_nearest_apple_num(state: GameState, pos: tuple, r: int) -> float:
    dists = [np.linalg.norm(np.array(pos) - np.array(apple)) for apple in state.fruits_locations]
    r_dists = [dist for dist in dists if dist <= r]
    return len(r_dists)


def _find_cluster_around_apples(state: GameState, player_index: int, n=3) -> float:
    head = state.snakes[player_index].head
    closest_apples = {apple: np.linalg.norm(np.array(head) - np.array(apple)) for apple in state.fruits_locations}
    closest_apples = sorted(closest_apples.items(), key=lambda k: k[1])
    num_of_apple = min(len(closest_apples), n)
    return sum([(_find_nearest_apple_num(state, apple[0], 3)/apple[1]) for apple in closest_apples[:num_of_apple]])


def _get_weight_sum_dist_from_fruits(state: GameState, player_index: int) -> float:
    head = state.snakes[player_index].head
    dists = np.sort([np.linalg.norm(np.array(head) - np.array(apple)) for apple in state.fruits_locations])
    sum = 0
    for i, dist in enumerate(dists):
        sum += 1 / dist
    return sum


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """
        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        start_time = time.time()
        depth = 3
        turn_state = self.TurnBasedGameState(state, agent_action=None)
        _, action = self.rb_minimax(turn_state, self.player_index, depth)
        self.turn_times.append(time.time() - start_time)
        print(self.turn_times[-1])
        return action

    def rb_minimax(self, state: TurnBasedGameState, player_index: int, depth: int):
        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            if state.game_state.turn_number == state.game_state.game_duration_in_turns or not state.game_state.snakes[player_index].alive:
                return state.game_state.snakes[player_index].length, state.agent_action
            if depth == 0:
                return heuristic(state.game_state, player_index), state.agent_action
            best_action = None
            max_value = -np.inf
            for action in state.game_state.get_possible_actions(player_index=player_index):
                turn_state = self.TurnBasedGameState(state.game_state, action)
                next_state_value, _ = self.rb_minimax(turn_state, player_index, depth)
                if next_state_value > max_value:
                    best_action = action
                    max_value = next_state_value
            return max_value, best_action
        else:
            best_action = None
            min_value = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                   player_index=self.player_index):
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_state = self.TurnBasedGameState(next_state, None)
                next_state_value, action = self.rb_minimax(turn_state, player_index, depth - 1)
                if next_state_value < min_value:
                    best_action = action
                    min_value = next_state_value
            return min_value, best_action


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        start_time = time.time()
        depth = 2
        turn_state = self.TurnBasedGameState(state, agent_action=None)
        _, action = self.alphabeta(turn_state, self.player_index, depth, -np.inf, np.inf)
        self.turn_times.append(time.time() - start_time)
        return action

    def alphabeta(self, state: MinimaxAgent.TurnBasedGameState, player_index: int, depth: int, alpha: float, beta: float):
        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            if state.game_state.turn_number == state.game_state.game_duration_in_turns or not state.game_state.snakes[player_index].alive:
                return state.game_state.snakes[player_index].length, state.agent_action
            if depth == 0:
                return heuristic(state.game_state, player_index), state.agent_action
            best_action = None
            max_value = -np.inf
            for action in state.game_state.get_possible_actions(player_index=player_index):
                turn_state = self.TurnBasedGameState(state.game_state, action)
                next_state_value, _ = self.alphabeta(turn_state, player_index, depth, alpha, beta)
                if next_state_value > max_value:
                    best_action = action
                    max_value = next_state_value
                alpha = max(max_value, alpha)
                if max_value >= beta:
                    return np.inf, best_action
            return max_value, best_action
        else:
            best_action = None
            min_value = np.inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                   player_index=self.player_index):
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_state = self.TurnBasedGameState(next_state, None)
                next_state_value, action = self.alphabeta(turn_state, player_index, depth - 1, alpha, beta)
                if next_state_value < min_value:
                    best_action = action
                    min_value = next_state_value
                beta = min(min_value, beta)
                if min_value <= alpha:
                    return -np.inf, best_action
            return min_value, best_action


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    n = 50
    actions = [GameAction.STRAIGHT for _ in range(n)]
    for i in range(n):
        best_action = []
        best_value = -np.inf
        for action in list(GameAction):
            actions[i] = action
            fitness = get_fitness(tuple(actions))
            if fitness > best_value:
                best_value = fitness
                best_action = [action]
            elif fitness == best_value:
                best_action.append(action)
        actions[i] = np.random.choice(best_action)

    print(actions)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    l = 50
    n = 30
    pc = 0.5
    pm = 0.05
    num_of_generations = 10
    generation = np.random.choice(list(GameAction), (n, l))
    for i in range(num_of_generations):

        fitness_list = [get_fitness(tuple(actions)) for actions in generation]
        total_fitness = np.sum(fitness_list)
        fitness_list = fitness_list / total_fitness
        next_gen = []
        for _ in range(n//2):
            parents = np.random.choice(n, 2, p=fitness_list)
            par1 = generation[parents[0]]
            par2 = generation[parents[1]]
            kid1 = par1
            kid2 = par2
            if np.random.uniform() > pc:
                crossover_index = np.random.choice(l)
                kid1[:crossover_index] = par1[:crossover_index]
                kid1[crossover_index:] = par2[crossover_index:]
                kid2[:crossover_index] = par2[:crossover_index]
                kid2[crossover_index:] = par1[crossover_index:]
            for i in range(l):
                if np.random.uniform() < pm:
                    kid1[i] = np.random.choice(list(GameAction))
                if np.random.uniform() < pm:
                    kid2[i] = np.random.choice(list(GameAction))
            next_gen.extend([kid1, kid2])
        generation = next_gen
    fitness_list = [get_fitness(tuple(actions)) for actions in generation]
    print(generation[np.argmax(np.array(fitness_list))])
    print(np.max(np.array(fitness_list)))


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        pass


if __name__ == '__main__':
    SAHC_sideways()
    local_search()

