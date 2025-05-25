# Normalize hero_holding to canonical form
def canonical_hand(hand):
    """Convert hand like 'AsKs' or 'KcKh' to canonical form like 'AKs' or 'KK'."""
    if not isinstance(hand, str) or len(hand) != 4:
        return hand

    rank_order = '23456789TJQKA'
    ranks = [hand[0], hand[2]]
    suits = [hand[1], hand[3]]

    # Sort cards by rank
    if rank_order.index(ranks[0]) < rank_order.index(ranks[1]):
        ranks = [ranks[1], ranks[0]]
        suits = [suits[1], suits[0]]

    suited = suits[0] == suits[1]
    if ranks[0] == ranks[1]:
        return f"{ranks[0]}{ranks[1]}"
    return f"{ranks[0]}{ranks[1]}{'s' if suited else 'o'}"

def parse_prev_line(prev_line, hero_pos):
    """
    Parses PokerBench-style action lines up to the hero's first action.
    Returns: facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, num_players_still_in
    """
    facing_raise = False
    num_raises = 0
    last_raiser_pos = None
    estimated_pot = 1.5  # start with SB + BB
    last_raise_size = 0.0
    positions_in = set()

    if not isinstance(prev_line, str):
        return facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, 1  # only hero still in

    position_order = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    tokens = prev_line.split('/')
    i = 0
    # Find the last action index for hero_pos to stop there
    hero_indices = [j for j in range(0, len(tokens), 2) if tokens[j] == hero_pos]
    stop_index = hero_indices[-1] if hero_indices else len(tokens)

    while i < len(tokens) - 1 and i < stop_index:
        pos = tokens[i]
        action = tokens[i + 1]

        if 'bb' in action.lower():  # raise or bet
            try:
                amount = float(action.lower().replace('bb', ''))
                estimated_pot += amount
                facing_raise = True
                num_raises += 1
                last_raiser_pos = pos
                last_raise_size = amount
                positions_in.add(pos)
            except:
                pass
        elif action.lower() == 'call':
            estimated_pot += last_raise_size
            positions_in.add(pos)
        elif action.lower() == 'allin':
            estimated_pot += last_raise_size
            positions_in.add(pos)
        elif action.lower() == 'fold':
            pass  # explicitly folded
        else:
            pass  # ignore unknowns

        i += 2

    # Infer folded positions
    hero_index = position_order.index(hero_pos)
    positions_before_hero = position_order[:hero_index]
    for pos in positions_before_hero:
        if pos not in positions_in:
            continue  # assumed to have folded

    num_players_still_in = len(positions_in) + 1  # include hero

    return facing_raise, num_raises, last_raiser_pos, estimated_pot, last_raise_size, num_players_still_in