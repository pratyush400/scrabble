from location import *
from board import *
from move import *
from itertools import combinations, permutations
from string import ascii_uppercase

ALL_TILES = [True] * 7
EMPTY_VALUES = {None, "", " "}

PREMIUM_VALUES = {
    TRIPLE_WORD_SCORE: 100,
    DOUBLE_WORD_SCORE: 50,
    TRIPLE_LETTER_SCORE: 20,
    DOUBLE_LETTER_SCORE: 10,
}


class SuperiorBot:
    def __init__(self):
        self._gatekeeper = None

    def set_gatekeeper(self, gatekeeper):
        self._gatekeeper = gatekeeper

    # ----------------------------------------------------------------------
    # MAIN ENTRY
    # ----------------------------------------------------------------------
    def choose_move(self):
        # Opening move must go on center
        if self._gatekeeper.get_square(CENTER) == DOUBLE_WORD_SCORE:
            move = self._search_moves(force_center=True)
        else:
            move = self._search_moves(force_center=False)

        # If we found nothing good → exchange as LAST resort
        if move is None:
            return ExchangeTiles(ALL_TILES)

        word, loc, direction, score = move
        if score < 4:
            # Don't exchange unless the move is really terrible
            # EXCEPT if no move exists
            if score <= 0:
                return ExchangeTiles(ALL_TILES)

        return PlayWord(word, loc, direction)

    # ----------------------------------------------------------------------
    # Tile + anchor helpers
    # ----------------------------------------------------------------------
    def _is_tile(self, loc):
        sq = self._gatekeeper.get_square(loc)
        return (
            isinstance(sq, str)
            and len(sq) == 1
            and sq.isalpha()
        )

    def _is_adjacent_to_tile(self, loc):
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            r, c = loc.r + dr, loc.c + dc
            if 0 <= r < WIDTH and 0 <= c < WIDTH:
                if self._is_tile(Location(r, c)):
                    return True
        return False

    def _premium_value(self, loc):
        sq = self._gatekeeper.get_square(loc)
        return PREMIUM_VALUES.get(sq, 0)

    # ----------------------------------------------------------------------
    # Find anchors (much faster)
    # ----------------------------------------------------------------------
    def _find_anchors(self):
        anchors = []

        for r in range(WIDTH):
            for c in range(WIDTH):
                loc = Location(r, c)
                sq = self._gatekeeper.get_square(loc)

                if self._is_tile(loc):
                    anchors.append(loc)
                    continue

                if sq in EMPTY_VALUES and self._is_adjacent_to_tile(loc):
                    anchors.append(loc)

        # Prefer premium squares
        anchors.sort(key=self._premium_value, reverse=True)
        return anchors

    # ----------------------------------------------------------------------
    # Efficient blank expansion
    # ----------------------------------------------------------------------
    def _expand_blanks(self, word):
        if "_" not in word:
            return [word]
        idx = word.index("_")
        return [word[:idx] + L + word[idx+1:] for L in ascii_uppercase]

    # ----------------------------------------------------------------------
    # MAIN MOVE SEARCH
    # ----------------------------------------------------------------------
    def _search_moves(self, force_center=False):
        hand = self._gatekeeper.get_hand()

        anchors = (
            [(CENTER, HORIZONTAL)]
            if force_center else
            self._find_anchors()
        )

        # Tile sequences 2–5 tiles (skip many permutations)
        sequences = []
        for n in (5,4,3,2):    # strongest first
            for combo in combinations(hand, n):
                for perm in permutations(combo):
                    w = "".join(perm)
                    # Skip too many blanks
                    if w.count("_") > 1:
                        continue
                    sequences.extend(self._expand_blanks(w))

        best = None
        best_score = -1

        # SEARCH
        for anchor in anchors:

            if force_center:
                loc, direction = anchor
                test_positions = [(loc, direction)]
            else:
                a = anchor
                test_positions = [(a, HORIZONTAL), (a, VERTICAL)]

            for loc, direction in test_positions:

                board_sq = self._gatekeeper.get_square(loc)
                board_letter = board_sq if isinstance(board_sq, str) and board_sq.isalpha() else None

                for w in sequences:

                    # If square has tile, word must match that letter at that position
                    if board_letter and w[0] != board_letter and w[-1] != board_letter:
                        continue  # simple prune

                    try:
                        self._gatekeeper.verify_legality(w, loc, direction)
                        score = self._gatekeeper.score(w, loc, direction)
                        # Premium prioritization bonus (lightweight)
                        score += self._premium_value(loc) // 10

                        if score > best_score:
                            best_score = score
                            best = (w, loc, direction, score)

                    except:
                        pass

        return best
