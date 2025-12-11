from location import *
from board import *
from move import *

ALL_TILES = [True] * 7


class BetterIncrementalist:
    """AI that:
      • On the first turn: tries 4-tile, then 3-tile, then 2-tile, then 1-tile words.
      • Otherwise: tries 3-tile, then 2-tile, then 1-tile.
      • If no moves available: exchanges all tiles."""

    def __init__(self):
        self._gatekeeper = None

    def set_gatekeeper(self, gatekeeper):
        self._gatekeeper = gatekeeper

    def choose_move(self):
        # maybe add in something to make bot prefer double/triple word/letter squares
        first_move = (self._gatekeeper.get_square(CENTER) == DOUBLE_WORD_SCORE)
        if first_move:
            move = (
                self._find_seven_tile_move() or
                self._find_six_tile_move() or
                self._find_five_tile_move() or
                self._find_four_tile_move() or
                self._find_three_tile_move() or
                self._find_two_tile_move() or
                self._find_one_tile_move()
            )
        else:
            move = (
                self._find_seven_tile_move() or
                self._find_six_tile_move() or
                self._find_five_tile_move() or
                self._find_four_tile_move() or
                self._find_three_tile_move() or
                self._find_two_tile_move() or
                self._find_one_tile_move()
            )
        return move or ExchangeTiles(ALL_TILES)

    def _find_seven_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                for k in range(len(hand)):
                    for m in range(len(hand)):
                        for n in range(len(hand)):
                            for o in range(len(hand)):
                                for p in range(len(hand)):
                                    if len({i, j, k, m, n, o, p}) != 7:
                                        continue  # don't use same tile twice
                                    raw = hand[i] + hand[j] + hand[k] + hand[m] + hand[n] + hand[o] + hand[p]
                                    word = self._replace_blank(raw)
                                    try:
                                        self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                                        score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                                        if score > best_score:
                                            best_score = score
                                            best_word = word
                                    except:
                                        pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_six_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                for k in range(len(hand)):
                    for m in range(len(hand)):
                        for n in range(len(hand)):
                            for o in range(len(hand)):
                                if len({i, j, k, m, n, o}) != 6:
                                    continue  # don't use same tile twice
                                raw = hand[i] + hand[j] + hand[k] + hand[m] + hand[n] + hand[o]
                                word = self._replace_blank(raw)
                                try:
                                    self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                                    score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                                    if score > best_score:
                                        best_score = score
                                        best_word = word
                                except:
                                    pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_five_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                for k in range(len(hand)):
                    for m in range(len(hand)):
                        for n in range(len(hand)):
                            if len({i, j, k, m, n}) != 5:
                                continue  # don't use same tile twice
                            raw = hand[i] + hand[j] + hand[k] + hand[m] + hand[n]
                            word = self._replace_blank(raw)
                            try:
                                self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                                score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                                if score > best_score:
                                    best_score = score
                                    best_word = word
                            except:
                                pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_four_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                for k in range(len(hand)):
                    for m in range(len(hand)):
                        if len({i, j, k, m}) != 4:
                            continue  # don't use same tile twice
                        raw = hand[i] + hand[j] + hand[k] + hand[m]
                        word = self._replace_blank(raw)
                        try:
                            self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                            score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                            if score > best_score:
                                best_score = score
                                best_word = word
                        except:
                            pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_three_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                for k in range(len(hand)):
                    if len({i, j, k}) != 3:
                        continue  # don't use same tile twice
                    raw = hand[i] + hand[j] + hand[k]
                    word = self._replace_blank(raw)
                    try:
                        self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                        score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                        if score > best_score:
                            best_score = score
                            best_word = word
                    except:
                        pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_two_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_word = None
        for i in range(len(hand)):
            for j in range(len(hand)):
                if i == j:
                    continue
                raw = hand[i] + hand[j]
                word = self._replace_blank(raw)
                try:
                    self._gatekeeper.verify_legality(word, CENTER, HORIZONTAL)
                    score = self._gatekeeper.score(word, CENTER, HORIZONTAL)
                    if score > best_score:
                        best_score = score
                        best_word = word
                except:
                    pass
        if best_word:
            return PlayWord(best_word, CENTER, HORIZONTAL)
        return None

    def _find_one_tile_move(self):
        hand = self._gatekeeper.get_hand()
        best_score = -1
        best_move = None
        for tile in hand:
            tile = self._replace_blank(tile)
            candidates = [tile + ' ', ' ' + tile]
            for word in candidates:
                for row in range(WIDTH):
                    for col in range(WIDTH):
                        location = Location(row, col)
                        for direction in (HORIZONTAL, VERTICAL):
                            try:
                                self._gatekeeper.verify_legality(word, location, direction)
                                score = self._gatekeeper.score(word, location, direction)
                                if score > best_score:
                                    best_score = score
                                    best_move = PlayWord(word, location, direction)
                            except:
                                pass
        return best_move

    def _replace_blank(self, letters):
        # try to replace if have time
        return letters.replace('_', 'E')
