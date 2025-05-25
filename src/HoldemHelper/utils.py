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
    Returns a dict with parsed features.
    """
    facing_raise = False
    num_raises = 0
    last_raiser_pos = None
    estimated_pot = 1.5  # SB + BB
    last_raise_size = 0.0
    positions_in = set()
    hero_acted_before = False
    to_call = 1.0  # default is BB size

    if not isinstance(prev_line, str):
        return {
            'facing_raise': False,
            'num_raises': 0,
            'last_raiser_pos': None,
            'estimated_pot': 1.5,
            'last_raise_size': 0.0,
            'num_players_still_in': 1,
            'to_call': 1.0,
            'pot_odds': 1.5,
            'is_3bet_plus': False,
            'hero_acted_before': False
        }

    position_order = ['UTG', 'HJ', 'CO', 'BTN', 'SB', 'BB']
    tokens = prev_line.split('/')
    i = 0

    hero_indices = [j for j in range(0, len(tokens), 2) if tokens[j] == hero_pos]
    stop_index = hero_indices[-1] if hero_indices else len(tokens)

    while i < len(tokens) - 1 and i < stop_index:
        pos = tokens[i]
        action = tokens[i + 1]

        if pos == hero_pos:
            hero_acted_before = True

        if 'bb' in action.lower():
            try:
                amount = float(action.lower().replace('bb', ''))
                estimated_pot += amount
                facing_raise = True
                num_raises += 1
                last_raiser_pos = pos
                last_raise_size = amount
                to_call = amount
                positions_in.add(pos)
            except:
                pass
        elif action.lower() == 'call':
            estimated_pot += to_call
            positions_in.add(pos)
        elif action.lower() == 'allin':
            estimated_pot += to_call
            positions_in.add(pos)
        elif action.lower() == 'fold':
            pass
        else:
            pass

        i += 2

    hero_index = position_order.index(hero_pos)
    positions_before_hero = position_order[:hero_index]
    for pos in positions_before_hero:
        if pos not in positions_in:
            continue

    num_players_still_in = len(positions_in) + 1

    pot_odds = estimated_pot / to_call if to_call > 0 else 0.0
    is_3bet_plus = num_raises >= 2

    return {
        'facing_raise': facing_raise,
        'num_raises': num_raises,
        'last_raiser_pos': last_raiser_pos,
        'estimated_pot': estimated_pot,
        'last_raise_size': last_raise_size,
        'num_players_still_in': num_players_still_in,
        'to_call': to_call,
        'pot_odds': pot_odds,
        'is_3bet_plus': is_3bet_plus,
        'hero_acted_before': hero_acted_before
    }