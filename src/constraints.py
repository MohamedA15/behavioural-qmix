import torch as th


# Shared Utilities



def _infer_n_food(obs_dim, n_agents):
    """
    Infer the number of food items from the observation size.

    Observation format:

    [food_x, food_y, food_level] * n_food
    +
    [self_x, self_y, self_level]
    +
    [other_x, other_y, other_level] * (n_agents - 1)
    """

    n_food = (obs_dim - (3 * n_agents)) / 3

    if int(n_food) != n_food:
        raise ValueError(
            f"Cannot infer number of food items from obs_dim={obs_dim}"
        )

    return int(n_food)


def extract_state_info(obs, n_agents):
    """
    Extract useful information from the observation tensor.

    Parameters
    ----------
    obs : Tensor

        Shape:

            (batch,
             seq_len,
             n_agents,
             obs_dim)

    Returns
    -------

    positions
        (batch, seq_len, n_agents, 2)

    agent_levels
        (batch, seq_len, n_agents)

    food_positions
        (batch, seq_len, n_food, 2)

    food_levels
        (batch, seq_len, n_food)
    """

    obs_dim = obs.shape[-1]

    n_food = _infer_n_food(obs_dim, n_agents)

    #
    # -----------------------------
    # Agent information
    # -----------------------------
    #

    self_start = 3 * n_food

    self_block = obs[
        ...,
        self_start:self_start + 3
    ]

    positions = self_block[..., 0:2]

    agent_levels = self_block[..., 2]

    #
    # -----------------------------
    # Food information
    # -----------------------------
    #

    #
    # Every agent observes the same food.
    # Therefore just use Agent 0's copy.
    #

    food_block = obs[
        :,
        :,
        0,
        :3 * n_food
    ]

    food_block = food_block.view(
        *food_block.shape[:-1],
        n_food,
        3
    )

    food_positions = food_block[..., 0:2]

    food_levels = food_block[..., 2]

    return (
        positions,
        agent_levels,
        food_positions,
        food_levels,
    )


# Constraint A – Active Participation



def compute_participation_violation(
    positions,
    food_positions,
    food_levels,
    N=3,
):
    """
    Compute a binary participation violation tensor.

    A violation occurs when:
    - the agent has remained in the same position for more than N timesteps
    - AND it is not adjacent to an uncollected food item

    Returns
    -------
    Tensor
        Shape:
        (batch, seq_len, n_agents)
    """
    with th.no_grad():
        batch_size, seq_len, n_agents, _ = positions.shape

        # Start with no violations
        stayed_still = th.zeros(
            batch_size,
            seq_len,
            n_agents,
            dtype=th.bool,
            device=positions.device,
        )

        # For each timestep from N onwards, check whether the agent has
        # stayed in exactly the same position for the previous N timesteps.
        for t in range(N, seq_len):
            # window shape: (batch, N+1, n_agents, 2)
            window = positions[:, t - N : t + 1]

            # current position expanded to compare with the window
            current = positions[:, t].unsqueeze(1)  # (batch, 1, n_agents, 2)

            # Compare every position in the window to the current one
            # same_position shape: (batch, N+1, n_agents)
            same_position = (window == current).all(dim=-1)

            # stayed_still[:, t] = True if all entries in the window match current
            stayed_still[:, t] = same_position.all(dim=1)

        #
        # ---------------------------------
        # Check adjacency to uncollected food
        # ---------------------------------
        #

        # (batch, seq_len, n_agents, 1, 2)
        agent_pos = positions.unsqueeze(3)

        # (batch, seq_len, 1, n_food, 2)
        food_pos = food_positions.unsqueeze(2)

        # Manhattan distance
        manhattan = (agent_pos - food_pos).abs().sum(dim=-1)

        # Ignore collected food
        uncollected = (food_levels > 0).unsqueeze(2)

        if not uncollected.any():
            return th.zeros_like(stayed_still, dtype=th.float32)

        # Any collected food gets an effectively infinite distance
        manhattan = manhattan.masked_fill(~uncollected, float("inf"))

        # Closest remaining food
        min_distance = manhattan.min(dim=-1).values

        # Tactical waiting = adjacent to food
        adjacent_to_food = min_distance <= 1

        #
        # ---------------------------------
        # Final participation violation
        # ---------------------------------
        #

        participation_violation = stayed_still & (~adjacent_to_food)

        return participation_violation.float()


def participation_loss(
    positions,
    food_positions,
    food_levels,
    chosen_action_qvals_per_agent,
    mask,
    N=3,
):
    """
    Active Participation constraint loss.

    Penalises the Q-values assigned to actions that violate
    the participation constraint.
    """

    # Binary violation tensor
    violation = compute_participation_violation(
        positions,
        food_positions,
        food_levels,
        N,
    )

    # Expand mask if necessary
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    if mask.shape[-1] == 1:
        mask = mask.expand_as(violation)

    # Only keep valid timesteps
    violation = violation * mask

    # Sanity-check shapes before multiplication
    assert chosen_action_qvals_per_agent.shape == violation.shape, (
        f"Expected {violation.shape}, got {chosen_action_qvals_per_agent.shape}"
    )

    # Penalise Q-values for violating actions, keeping the penalty non-negative
    penalty = violation * th.relu(chosen_action_qvals_per_agent)

    # Mean loss over valid agent-timesteps
    loss = penalty.sum() / mask.sum().clamp(min=1)

    return loss, violation
    


# -----------------------------------------------------
# Constraint B – Interference Avoidance
# -----------------------------------------------------

def compute_interference_violation(
    positions,
    actions,
    grid_size,
):
    """
    Compute a binary interference violation tensor.

    A violation occurs when an agent attempts to:
    - move into another agent, or
    - move outside the environment boundaries.

    Parameters
    ----------
    positions : Tensor
        Shape:
        (batch, seq_len, n_agents, 2)

    actions : Tensor
        Shape:
        (batch, seq_len, n_agents, 1)

    grid_size : int
        Size of the square environment.

    Returns
    -------
    Tensor
        Shape:
        (batch, seq_len, n_agents)
    """

    with th.no_grad():
        batch_size, seq_len, n_agents, _ = positions.shape

        interference_violation = th.zeros(
            batch_size,
            seq_len,
            n_agents,
            dtype=th.bool,
            device=positions.device,
        )

        #
        # ---------------------------------
        # Compute intended next positions
        # ---------------------------------
        #

        # Start by assuming every agent stays where it is
        next_positions = positions.clone()

        #
        # ---------------------------------
        # NORTH (Action = 1)
        # ---------------------------------
        #

        north_mask = actions.squeeze(-1) == 1

        next_positions[..., 1][north_mask] -= 1

        #
        # ---------------------------------
        # SOUTH (Action = 2)
        # ---------------------------------
        #

        south_mask = actions.squeeze(-1) == 2

        next_positions[..., 1][south_mask] += 1

        #
        # ---------------------------------
        # WEST (Action = 3)
        # ---------------------------------
        #

        west_mask = actions.squeeze(-1) == 3

        next_positions[..., 0][west_mask] -= 1

        #
        # ---------------------------------
        # EAST (Action = 4)
        # ---------------------------------
        #

        east_mask = actions.squeeze(-1) == 4

        next_positions[..., 0][east_mask] += 1

        # Actions 0 (NONE) and 5 (LOAD) leave the position unchanged.

        #
        # ---------------------------------
        # Check wall collisions
        # ---------------------------------
        #

        wall_collision = (
            (next_positions[..., 0] < 0)
            | (next_positions[..., 0] >= grid_size)
            | (next_positions[..., 1] < 0)
            | (next_positions[..., 1] >= grid_size)
        )

        #
        # ---------------------------------
        # Check agent collisions
        # ---------------------------------
        #

        agent_collision = th.zeros_like(wall_collision)

        for i in range(n_agents):
            for j in range(n_agents):

                if i == j:
                    continue

                collision = (
                    next_positions[..., i, :]
                    == next_positions[..., j, :]
                ).all(dim=-1)

                agent_collision[..., i] |= collision

        #
        # ---------------------------------
        # Final interference violation
        # ---------------------------------
        #

        interference_violation = wall_collision | agent_collision

        return interference_violation.float()

def interference_loss(
    positions,
    actions,
    chosen_action_qvals_per_agent,
    mask,
    grid_size,
):
    """
    Interference Avoidance constraint loss.

    Penalises the Q-values assigned to actions that violate
    the interference constraint.
    """

    # Binary violation tensor
    violation = compute_interference_violation(
        positions,
        actions,
        grid_size,
    )

    # Expand mask if necessary
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    if mask.shape[-1] == 1:
        mask = mask.expand_as(violation)

    # Only keep valid timesteps
    violation = violation * mask

    # Sanity-check shapes before multiplication
    assert chosen_action_qvals_per_agent.shape == violation.shape, (
        f"Expected {violation.shape}, got {chosen_action_qvals_per_agent.shape}"
    )

    # Penalise Q-values for violating actions, keeping the penalty non-negative
    penalty = violation * th.relu(chosen_action_qvals_per_agent)

    # Mean loss over valid agent-timesteps
    loss = penalty.sum() / mask.sum().clamp(min=1)

    return loss, violation


# -----------------------------------------------------
# Constraint C – Cooperative Commitment
# -----------------------------------------------------

def compute_commitment_violation(
    positions,
    agent_levels,
    food_positions,
    food_levels,
    M=2,
):
    """
    Compute a binary cooperative commitment violation tensor.

    A violation occurs when an agent begins cooperating on a
    heavy food item but subsequently moves away before the
    cooperative task is completed.

    Parameters
    ----------
    positions : Tensor
        (batch, seq_len, n_agents, 2)

    agent_levels : Tensor
        (batch, seq_len, n_agents)

    food_positions : Tensor
        (batch, seq_len, n_food, 2)

    food_levels : Tensor
        (batch, seq_len, n_food)

    M : int
        Teammate proximity threshold.

    Returns
    -------
    Tensor
        (batch, seq_len, n_agents)
    """

    with th.no_grad():

        batch_size, seq_len, n_agents, _ = positions.shape

        commitment_violation = th.zeros(
            batch_size,
            seq_len,
            n_agents,
            dtype=th.bool,
            device=positions.device,
        )

        #
        # ---------------------------------
        # Compute agent-to-food distances
        # ---------------------------------
        #

        # Expand dimensions so every agent is compared
        # against every food item.
        agent_pos = positions.unsqueeze(3)

        # Shape:
        # (batch, seq_len, 1, n_food, 2)
        food_pos = food_positions.unsqueeze(2)

        # Manhattan distance between every agent
        # and every food item.
        #
        # Shape:
        # (batch, seq_len, n_agents, n_food)
        distances = (
            (agent_pos[..., 0] - food_pos[..., 0]).abs()
            + (agent_pos[..., 1] - food_pos[..., 1]).abs()
        )

        #
        # ---------------------------------
        # Heavy food detection
        # ---------------------------------
        #

        # A food item is considered "heavy" for an agent
        # if the food level is greater than the agent's level,
        # meaning the agent cannot collect it alone.
        #
        # Shape:
        # (batch, seq_len, n_agents, n_food)
        heavy_food = (
            food_levels.unsqueeze(2)
            > agent_levels.unsqueeze(3)
        )

        #
        # ---------------------------------
        # Adjacent to heavy food
        # ---------------------------------
        #

        # An agent is adjacent if its Manhattan
        # distance to the food is exactly 1.
        adjacent_to_food = distances == 1

        # Agent must be adjacent to a food item
        # that requires cooperation.
        #
        # Shape:
        # (batch, seq_len, n_agents, n_food)
        adjacent_heavy_food = (
            adjacent_to_food
            & heavy_food
        )

        #
        # ---------------------------------
        # Teammate proximity
        # ---------------------------------
        #

        # A teammate is considered nearby if another
        # agent (not the current one) is within M
        # grid cells of the same food item.
        #
        # Shape:
        # (batch, seq_len, n_agents, n_food)
        teammate_nearby = th.zeros_like(
            adjacent_heavy_food
        )

        # Agent 0 checks whether Agent 1
        # is close to the same food.
        teammate_nearby[:, :, 0, :] = (
            distances[:, :, 1, :] <= M
        )

        # Agent 1 checks whether Agent 0
        # is close to the same food.
        teammate_nearby[:, :, 1, :] = (
            distances[:, :, 0, :] <= M
        )

        #
        # ---------------------------------
        # Commitment trigger
        # ---------------------------------
        #

        # An agent is considered committed if it is
        # adjacent to a heavy food and another teammate
        # is within M cells of the same food.
        #
        # Shape (kept per-food, not reduced yet):
        # (batch, seq_len, n_agents, n_food)
        commitment_trigger = (
            adjacent_heavy_food
            & teammate_nearby
        )

        #
        # ---------------------------------
        # Choose a single commitment target
        # ---------------------------------
        #

        # If an agent satisfies the commitment trigger for
        # more than one food at once, we need to pick ONE
        # food as its actual target, rather than checking
        # "moved away" against every triggered food
        # independently. We pick the nearest triggered food
        # as the target (a reasonable proxy for intent).
        #
        # Mask out non-committed foods with +inf so they're
        # never selected by argmin.
        #
        # Shape:
        # (batch, seq_len, n_agents, n_food)
        committed_distances = distances.masked_fill(
            ~commitment_trigger, float("inf")
        )

        # Index of the chosen target food per agent, per
        # timestep. Meaningless where has_commitment is
        # False (argmin over all-inf), so it must always be
        # gated by has_commitment before use.
        #
        # Shape:
        # (batch, seq_len, n_agents)
        target_food_idx = committed_distances.argmin(dim=3)

        # Whether the agent has ANY valid commitment at all
        # at this timestep.
        #
        # Shape:
        # (batch, seq_len, n_agents)
        has_commitment = commitment_trigger.any(dim=3)

        #
        # ---------------------------------
        # Distance at current and next step
        # ---------------------------------
        #

        # Distance at time t
        current_distance = distances[:, :-1]

        # Distance at time t+1
        next_distance = distances[:, 1:]

        # Shapes:
        # (batch, seq_len-1, n_agents, n_food)

        #
        # ---------------------------------
        # Gather distance to the CHOSEN target only
        # ---------------------------------
        #

        # Target is chosen based on commitment state at
        # time t, then tracked forward to t+1.
        #
        # Shape:
        # (batch, seq_len-1, n_agents, 1)
        target_idx = target_food_idx[:, :-1].unsqueeze(-1)

        # Shape:
        # (batch, seq_len-1, n_agents)
        target_distance_t = th.gather(
            current_distance, dim=3, index=target_idx
        ).squeeze(-1)

        target_distance_t1 = th.gather(
            next_distance, dim=3, index=target_idx
        ).squeeze(-1)

        #
        # ---------------------------------
        # Moved away from target check
        # ---------------------------------
        #

        # Shape:
        # (batch, seq_len-1, n_agents)
        moved_away_from_target = (
            target_distance_t1 > target_distance_t
        )

        #
        # ---------------------------------
        # Violation
        # ---------------------------------
        #

        # A violation requires: the agent had a commitment
        # at time t, AND it moved away from the specific
        # food it was committed to (not just any food it
        # happened to also be near).
        #
        # Shape:
        # (batch, seq_len-1, n_agents)
        violation = (
            has_commitment[:, :-1]
            & moved_away_from_target
        )

        commitment_violation[:, :-1] = violation

    return commitment_violation

def commitment_loss(
    positions,
    agent_levels,
    food_positions,
    food_levels,
    chosen_action_qvals_per_agent,
    mask,
):
    """
    Cooperative Commitment constraint loss.

    Penalises the Q-values assigned to actions that violate
    the cooperative commitment constraint.
    """

    # Binary violation tensor
    violation = compute_commitment_violation(
        positions,
        agent_levels,
        food_positions,
        food_levels,
    )

    # Expand mask if necessary
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    if mask.shape[-1] == 1:
        mask = mask.expand_as(violation)

    # Only keep valid timesteps
    violation = violation * mask

    # Sanity-check shapes before multiplication
    assert chosen_action_qvals_per_agent.shape == violation.shape, (
        f"Expected {violation.shape}, got {chosen_action_qvals_per_agent.shape}"
    )

    # Penalise Q-values for violating actions,
    # keeping the penalty non-negative
    penalty = violation * th.relu(chosen_action_qvals_per_agent)

    # Mean loss over valid agent-timesteps
    loss = penalty.sum() / mask.sum().clamp(min=1)

    return loss, violation



# ----------------------------------------------------
# Constraint D – Cooperative Target Selection
# ----------------------------------------------------


def compute_target_violation(
    positions,
    agent_levels,
    food_positions,
    food_levels,
    K=3,
):
    """
    Compute a binary cooperative target selection violation tensor.

    A violation occurs when an agent moves towards a heavy food
    without another teammate also moving towards the same food.
    """

    with th.no_grad():

        batch_size, seq_len, n_agents, _ = positions.shape

        target_violation = th.zeros(
            batch_size,
            seq_len,
            n_agents,
            dtype=th.bool,
            device=positions.device,
        )

        # ---------------------------------
        # Compute agent-to-food distances
        # ---------------------------------

        agent_pos = positions.unsqueeze(3)
        food_pos = food_positions.unsqueeze(2)

        distances = (
            (agent_pos[..., 0] - food_pos[..., 0]).abs()
            + (agent_pos[..., 1] - food_pos[..., 1]).abs()
        )

        # ---------------------------------
        # Heavy food detection
        # ---------------------------------

        heavy_food = (
            food_levels.unsqueeze(2)
            > agent_levels.unsqueeze(3)
        )

        # Ignore foods that are not heavy
        heavy_distances = distances.masked_fill(
            ~heavy_food,
            float("inf"),
        )

        # Nearest heavy food
        target_food_idx = heavy_distances.argmin(dim=3)

        # Whether a heavy food exists
        has_target = heavy_food.any(dim=3)

        # ---------------------------------
        # Distance to chosen heavy food
        # ---------------------------------

        target_idx = target_food_idx.unsqueeze(-1)

        target_distance = th.gather(
            distances,
            dim=3,
            index=target_idx,
        ).squeeze(-1)

        # ---------------------------------
        # Compute ΔD
        # ---------------------------------

        delta_distance = th.zeros_like(target_distance)

        delta_distance[:, K:] = (
            target_distance[:, K:]
            - target_distance[:, :-K]
        )

        # Agent is moving towards the heavy food
        moving_towards = delta_distance < 0

        # ---------------------------------
        # Same target check
        # ---------------------------------

        same_target = (
            target_food_idx[:, :, 0]
            == target_food_idx[:, :, 1]
        )

        # ---------------------------------
        # Teammate support
        # ---------------------------------

        teammate_support = th.zeros_like(moving_towards)

        # Agent 0 is supported if Agent 1 is also
        # moving towards the same heavy food.
        teammate_support[:, :, 0] = (
            moving_towards[:, :, 1]
            & same_target
        )

        # Agent 1 is supported if Agent 0 is also
        # moving towards the same heavy food.
        teammate_support[:, :, 1] = (
            moving_towards[:, :, 0]
            & same_target
        )

        # ---------------------------------
        # Final violation
        # ---------------------------------

        violation = (
            has_target
            & moving_towards
            & (~teammate_support)
        )

        target_violation = violation

        return target_violation.float()


# ----------------------------------------------------
# Constraint D – Loss Wrapper
# ----------------------------------------------------


def target_loss(
    positions,
    agent_levels,
    food_positions,
    food_levels,
    chosen_action_qvals_per_agent,
    mask,
):
    violation = compute_target_violation(
        positions,
        agent_levels,
        food_positions,
        food_levels,
    )

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    if mask.shape[-1] == 1:
        mask = mask.expand_as(violation)

    violation = violation * mask

    assert chosen_action_qvals_per_agent.shape == violation.shape

    penalty = violation * th.relu(chosen_action_qvals_per_agent)

    loss = penalty.sum() / mask.sum().clamp(min=1)

    return loss, violation