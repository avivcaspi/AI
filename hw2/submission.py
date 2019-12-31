from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """
    # Insert your code here...
    # if the snake is dead the state value is the snake`s length
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    # calculate the area the snake can reach in current state
    reachable_area = _get_reachable_area(state, player_index)
    # finding closest apple
    closest_apple = _find_nearest_apple_dist(state, player_index)
    # finding enemy distance
    enemy_dist = _get_dist_from_enemy(state, player_index)
    # Indicators for enemy distance and reachable area
    enemy_dist_indicator = 0 if enemy_dist <= 2 else 1
    reachable_area_indicator = 0 if reachable_area < state.snakes[player_index].length + 5 else 2
    return state.snakes[player_index].length + 1 / closest_apple + reachable_area_indicator + enemy_dist_indicator


def _get_dist_from_enemy(state: GameState, player_index: int) -> float:
    head = state.snakes[player_index].head
    # finds all distances from other snakes and finding the closest one to player index snake
    dists = [np.linalg.norm(np.array(head) - np.array(enemy.head)) for enemy in state.snakes if enemy.index != player_index and enemy.alive]
    if len(dists) > 0:
        return np.min(dists)
    return np.inf


def _find_nearest_apple_dist(state: GameState, player_index: int) -> float:
    head = state.snakes[player_index].head
    # finds all distances from all the apples remaining and finding the closest one to player index snake
    dists = [np.linalg.norm(np.array(head) - np.array(apple)) for apple in state.fruits_locations]
    if len(dists) > 0:
        return np.min(dists)
    # if there are no apples left (end game) we return infinity
    return np.inf


def _get_reachable_area(state: GameState, player_index: int) -> float:
    snake = state.snakes[player_index]
    # gets the boards of player index snake positions
    snake_board = state.get_board(player_index)[0]
    # dfs will find the area reachable
    return _dfs(snake_board, snake.head, True)


def _dfs(board, pos, start: bool, depth=0):
    # block depth from reaching 30 (not necessary)
    if len(board) > 25 and depth > 30:
        return 0
    area = 0
    # if the curresnt pos is outside of the board or there is 1 in the pos we retrun 0
    if not (0 <= pos[0] < len(board) and 0 <= pos[1] < len(board[0])) or (board[pos[0]][pos[1]] > 0 and not start):
        return 0
    # add the current pos to the area and call all neighbors pos
    area += 1
    # mark current pos as checked
    board[pos[0]][pos[1]] = 1
    area += _dfs(board, [pos[0] + 1, pos[1]], False, depth + 1)  # down
    area += _dfs(board, [pos[0], pos[1] + 1], False, depth + 1)  # right
    area += _dfs(board, [pos[0] - 1, pos[1]], False, depth + 1)  # up
    area += _dfs(board, [pos[0], pos[1] - 1], False, depth + 1)  # left
    return area


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
        depth = 2
        turn_state = self.TurnBasedGameState(state, agent_action=None)
        _, action = self.rb_minimax(turn_state, self.player_index, depth)
        return action

    def rb_minimax(self, state: TurnBasedGameState, player_index: int, depth: int):
        # check if we are at maximum node
        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            # if we finished the game or our snake is dead we return the snake`s length
            if state.game_state.turn_number == state.game_state.game_duration_in_turns or not state.game_state.snakes[
                player_index].alive:
                return state.game_state.snakes[player_index].length, state.agent_action
            # using heuristic if we reached depth of 0
            if depth == 0:
                return heuristic(state.game_state, player_index), state.agent_action
            best_action = None
            max_value = -np.inf
            # going over our player possible actions and returning the action with max value
            for action in state.game_state.get_possible_actions(player_index=player_index):
                # creates turn state with the action we are checking
                turn_state = self.TurnBasedGameState(state.game_state, action)
                next_state_value, _ = self.rb_minimax(turn_state, player_index, depth)
                if next_state_value > max_value:
                    best_action = action
                    max_value = next_state_value
            return max_value, best_action
        # minimum node
        else:
            best_action = None
            min_value = np.inf
            # going over all opponents possible actions, and returning the actions with minimum value
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                # building next state
                next_state = get_next_state(state.game_state, opponents_actions)
                turn_state = self.TurnBasedGameState(next_state, None)
                # getting the value for the state we build (calling minimax with depth smaller by 1)
                next_state_value, action = self.rb_minimax(turn_state, player_index, depth - 1)
                if next_state_value < min_value:
                    best_action = action
                    min_value = next_state_value
            return min_value, best_action


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        depth = 3
        turn_state = self.TurnBasedGameState(state, agent_action=None)
        _, action = self.alphabeta(turn_state, self.player_index, depth, -np.inf, np.inf)
        return action

    def alphabeta(self, state: MinimaxAgent.TurnBasedGameState, player_index: int, depth: int, alpha: float,
                  beta: float):
        # check if we are at max node
        if state.turn == MinimaxAgent.Turn.AGENT_TURN:
            if state.game_state.turn_number == state.game_state.game_duration_in_turns or not state.game_state.snakes[
                player_index].alive:
                return state.game_state.snakes[player_index].length, state.agent_action
            if depth == 0:
                return heuristic(state.game_state, player_index), state.agent_action
            best_action = None
            max_value = -np.inf
            for action in state.game_state.get_possible_actions(player_index=player_index):
                turn_state = self.TurnBasedGameState(state.game_state, action)
                # passing alpha and beta to the next node
                next_state_value, _ = self.alphabeta(turn_state, player_index, depth, alpha, beta)
                if next_state_value > max_value:
                    best_action = action
                    max_value = next_state_value
                # determine alpha according to the max value we currently have
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
                # passing alpha and beta to the next node
                next_state_value, action = self.alphabeta(turn_state, player_index, depth - 1, alpha, beta)
                if next_state_value < min_value:
                    best_action = action
                    min_value = next_state_value
                # determine beta according to the min value we currently have
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
    # size of our state
    n = 50
    # starting state (STRAIGHT, STRAIGHT ,..., STRAIGHT)
    actions = [GameAction.STRAIGHT for _ in range(n)]
    # going over each action in current state and checking which action gives the best results
    for i in range(n):
        best_action = []
        best_value = -np.inf
        # checking all actions possible
        for action in list(GameAction):
            actions[i] = action
            # getting fitness of current state with current action changed
            fitness = get_fitness(tuple(actions))
            if fitness > best_value:
                best_value = fitness
                best_action = [action]
            elif fitness == best_value:
                best_action.append(action)
        # picking best action
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
    # size of state
    l = 50
    # size of population
    n = 100
    # probability for crossover
    pc = 0.5
    # probability for mutate
    pm = 0.05
    num_of_generations = 30
    # creating first generation with random actions
    generation = np.random.choice(list(GameAction), (n, l))
    for i in range(num_of_generations):
        # finds fitness for all snakes in the current generation
        fitness_list = [get_fitness(tuple(actions)) for actions in generation]
        total_fitness = np.sum(fitness_list)
        # calculating probability for each snake according to it`s score
        probability_list = fitness_list / total_fitness
        next_gen = []
        # building new generation, picking 2 parents -> crossover -> mutate
        for _ in range(n // 2):
            parents = np.random.choice(n, 2, p=probability_list)
            par1 = generation[parents[0]]
            par2 = generation[parents[1]]
            kid1 = par1
            kid2 = par2
            # crossover
            if np.random.uniform() > pc:
                crossover_index = np.random.choice(l)
                kid1[:crossover_index] = par1[:crossover_index]
                kid1[crossover_index:] = par2[crossover_index:]
                kid2[:crossover_index] = par2[:crossover_index]
                kid2[crossover_index:] = par1[crossover_index:]
            # mutate
            for i in range(l):
                if np.random.uniform() < pm:
                    kid1[i] = np.random.choice(list(GameAction))
                if np.random.uniform() < pm:
                    kid2[i] = np.random.choice(list(GameAction))
            # adding kids to new generation
            next_gen.extend([kid1, kid2])
        generation = next_gen
    fitness_list = [get_fitness(tuple(actions)) for actions in generation]
    print(generation[np.argmax(np.array(fitness_list))])


def _heuristic_for_tournament(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return: heuristic value for the given state
    """
    # Insert your code here...
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length
    # we assume that our snake won`t kill itself until it reach length of 6, so to save calculation time at the
    # beginning we start check reachable area later on
    if state.snakes[player_index].length > 6:
        reachable_area = _get_reachable_area(state, player_index)
    else:
        reachable_area = 0
    closest_apple = _find_nearest_apple_dist(state, player_index)
    enemy_dist = _get_dist_from_enemy(state, player_index)
    enemy_dist_factor = 0 if enemy_dist <= 2 else 1
    reachable_area_factor = 0 if reachable_area < state.snakes[player_index].length + 5 else 2
    return state.snakes[player_index].length + 1 / closest_apple + reachable_area_factor + enemy_dist_factor


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:

        # Very similar to greedy agent get action, but now instead of picking an action with highest value we pick
        # action with highest avg value, so in avg our snake will do good
        best_actions = state.get_possible_actions(player_index=self.player_index)
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            avg_value = 0
            actions_len = 0
            for opponents_actions in state.get_possible_actions_dicts_given_action(action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                h_value = _heuristic_for_tournament(next_state, self.player_index)
                avg_value += h_value
                actions_len += 1
                if len(state.opponents_alive) > 2:
                    # consider only 1 possible opponents actions to reduce time & memory:
                    break
            avg_value /= actions_len
            # choosing action according to the avg value we got preforming this action
            if avg_value > best_value:
                best_value = avg_value
                best_actions = [action]
            elif avg_value == best_value:
                best_actions.append(action)

        return np.random.choice(best_actions)


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
