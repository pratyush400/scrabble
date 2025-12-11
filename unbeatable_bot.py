
import random
import math
import time
from collections import defaultdict, Counter
from copy import deepcopy
from gatekeeper import  GateKeeper

# Import project modules (must be next to this file)
from location import Location, HORIZONTAL, VERTICAL, WIDTH, CENTER
from move import PlayWord, ExchangeTiles
import board as board_module  # for DICTIONARY and TILE_VALUES and initial tile distribution

# -------------------------
# GADDAG Implementation (compact, production-capable)
# -------------------------
GADDAG_BREAK = '+'


class GADDAGNode:
    __slots__ = ("next", "is_word")
    def __init__(self):
        self.next = {}
        self.is_word = False


class GADDAG:
    def __init__(self, word_list):
        self.root = GADDAGNode()
        # Insert uppercase words
        for w in word_list:
            self.insert_word(w.upper())

    def insert_word(self, word):
        n = len(word)
        if n == 0:
            return
        # split at every internal point: reversed prefix + '+' + suffix
        for i in range(1, n):
            prefix = word[:i][::-1]
            suffix = word[i:]
            form = prefix + GADDAG_BREAK + suffix
            node = self.root
            for ch in form:
                if ch not in node.next:
                    node.next[ch] = GADDAGNode()
                node = node.next[ch]
            node.is_word = True
        # also allow full reversed word (allows plays entirely to left)
        rev = word[::-1]
        node = self.root
        for ch in rev:
            if ch not in node.next:
                node.next[ch] = GADDAGNode()
            node = node.next[ch]
        node.is_word = True


# -------------------------
# Utilities: initial tile pool (copy from board.Board init)
# -------------------------
_INITIAL_TILE_POOL = list(
    'aaaaaaaaabbccddddeeeeeeeeeeeeffggghhiiiiiiiiijkllllmmnnnnnnooooooooppqrrrrrrssssttttttuuuuvvwwxyyz__'
)

# -------------------------
# Move container
# -------------------------
class CandidateMove:
    """
    Represents a candidate word placement on the board.
    """
    __slots__ = ("word", "row", "col", "direction", "score", "used_tiles", "is_blank_mask")

    def __init__(self, word, row, col, direction, score, used_tiles=None, is_blank_mask=None):
        self.word = word
        self.row = row
        self.col = col
        self.direction = direction  # HORIZONTAL or VERTICAL
        self.score = score
        self.used_tiles = used_tiles or []  # letters drawn from rack (list)
        self.is_blank_mask = is_blank_mask or []  # parallel mask whether each used tile was a blank

    def __repr__(self):
        d = "H" if self.direction == HORIZONTAL else "V"
        return f"{self.word} @{(self.row,self.col)} {d} score={self.score}"

# -------------------------
# GADDAG-based move generator adapted to your Board API
# We'll build a board matrix by querying gatekeeper.get_square(Location)
# -------------------------
class GADDAGGenerator:
    def __init__(self, gaddag, dictionary_set):
        self.g = gaddag
        self.dict = dictionary_set  # set of lowercase words

    def build_board_matrix(self, Gatekeeper):
        # Build a 2D char matrix of WIDTH x WIDTH representing current board
        mat = [[' ' for _ in range(WIDTH)] for __ in range(WIDTH)]
        for r in range(WIDTH):
            for c in range(WIDTH):
                # create a location object and query gatekeeper
                loc = Location(r, c)
                sq = Gatekeeper.get_square(loc)
                if isinstance(sq, str):
                    mat[r][c] = sq
                else:
                    # fallback - treat as string
                    mat[r][c] = str(sq)
        return mat

    def _is_letter(self, mat, r, c):
        if r < 0 or c < 0 or r >= WIDTH or c >= WIDTH:
            return False
        return mat[r][c].isalpha()

    def find_anchors_row(self, mat, r):
        anchors = []
        for c in range(WIDTH):
            if mat[r][c] != ' ':
                continue
            touching = (
                self._is_letter(mat, r, c-1) or
                self._is_letter(mat, r, c+1) or
                self._is_letter(mat, r-1, c) or
                self._is_letter(mat, r+1, c)
            )
            if touching:
                anchors.append(c)
        # first move special case: center
        if all(mat[r][c] == ' ' for r in range(WIDTH) for c in range(WIDTH)):
            anchors = [CENTER.c] if r == CENTER.r else []
        return anchors

    def compute_cross_checks_col(self, mat, r, c):
        # if tile exists, allowed letter is that tile
        if mat[r][c].isalpha():
            return {mat[r][c]}
        # build vertical string above and below
        up = r - 1
        prefix = ''
        while up >= 0 and mat[up][c].isalpha():
            prefix = mat[up][c] + prefix
            up -= 1
        down = r + 1
        suffix = ''
        while down < WIDTH and mat[down][c].isalpha():
            suffix += mat[down][c]
            down += 1
        if prefix == '' and suffix == '':
            return set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        valid = set()
        # check every letter
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            word = (prefix + ch + suffix).lower()
            if word in self.dict:
                valid.add(ch)
        return valid

    def generate_all_moves(self, gatekeeper, rack):
        """
        Produces CandidateMove objects for both horizontal and vertical placements.
        """
        mat = self.build_board_matrix(gatekeeper)
        moves = []
        # horizontal passes
        moves += self._generate_in_orientation(mat, rack, horizontal=True, gatekeeper=gatekeeper)
        # vertical: transpose matrix then convert coords
        transposed = [list(col) for col in zip(*mat)]
        vert_moves = self._generate_in_orientation(transposed, rack, horizontal=True, gatekeeper=gatekeeper, transpose=True)
        moves += vert_moves
        return moves

    def _generate_in_orientation(self, mat, rack, horizontal, gatekeeper, transpose=False):
        results = []
        # Precompute cross checks for each empty cell (for vertical cross checks in horizontal generation)
        cross_checks = [[None]*WIDTH for _ in range(WIDTH)]
        for r in range(WIDTH):
            for c in range(WIDTH):
                cross_checks[r][c] = self.compute_cross_checks_col(mat, r, c)
        # For each row, find anchors
        for r in range(WIDTH):
            anchors = self.find_anchors_row(mat, r)
            if not anchors:
                continue
            for anchor_col in anchors:
                # attempt left-building using GADDAG
                results += self._extend_anchor(mat, rack.upper(), r, anchor_col, cross_checks, transpose)
        return results

    def _extend_anchor(self, mat, rack, r, anchor_col, cross_checks, transpose):
        results = []
        # start left exploration from anchor_col (position where first letter of word will align)
        # We'll implement a recursive GADDAG traversal using explicit stack for performance
        root = self.g.root

        # Helper to check legality when finishing a word: ensure fits and doesn't conflict
        def legal_place(row, col, word):
            if col < 0 or col + len(word) > WIDTH:
                return False
            for i, ch in enumerate(word):
                sq = mat[row][col+i]
                if sq != ' ' and sq != ch:
                    return False
            return True

        def left_search(node, c, rack_left, built_rev):
            # If we see break transition, enter right_search with path = reversed built_rev
            if GADDAG_BREAK in node.next:
                path = built_rev[::-1]  # now normal left-to-right prefix
                right_node = node.next[GADDAG_BREAK]
                # start right search from anchor_col (word will start at column = anchor_col - len(path))
                start_col = anchor_col - len(path)
                # quick bounds check
                if start_col < -WIDTH:  # impossible
                    pass
                else:
                    right_search(right_node, start_col, anchor_col, rack_left, path)
            # Try to extend further to the left (i.e., consume more reversed-prefix letters)
            left_c = c - 1
            if left_c < 0:
                return
            # if board has letter to left, must follow it
            if mat[r][left_c].isalpha():
                ch = mat[r][left_c]
                if ch in node.next:
                    left_search(node.next[ch], left_c, rack_left, built_rev + ch)
                return
            # else empty square, can try rack tiles (including blanks)
            # iterate through unique tiles in rack for efficiency
            for i, t in enumerate(rack_left):
                # tile removal approach
                if t == '_':
                    # blank: try every letter allowed by cross-check at left_c
                    for ch in cross_checks[r][left_c]:
                        if ch in node.next:
                            new_rack = rack_left[:i] + rack_left[i+1:]
                            left_search(node.next[ch], left_c, new_rack, built_rev + ch)
                else:
                    if t in cross_checks[r][left_c] and t in node.next:
                        new_rack = rack_left[:i] + rack_left[i+1:]
                        left_search(node.next[t], left_c, new_rack, built_rev + t)

        # right_search grows the suffix
        def right_search(node, start_col, col, rack_left, path):
            # path is the currently-built left part (from left search), node is the GADDAG node after '+'
            # If node represents a word, record it
            if node.is_word:
                word = path
                if legal_place(r, start_col, word):
                    # Convert start_col to original orientation coordinates if transposed
                    if transpose:
                        # in transposed board, row/col swap
                        results.append(self._make_candidate(word, r, start_col, VERTICAL, gatekeeper))
                    else:
                        results.append(self._make_candidate(word, r, start_col, HORIZONTAL, gatekeeper))
            # extend right
            next_c = col + 1
            if next_c >= WIDTH:
                return
            # if board already has tile at next_c
            if mat[r][next_c].isalpha():
                ch = mat[r][next_c]
                if ch in node.next:
                    right_search(node.next[ch], start_col, next_c, rack_left, path + ch)
                return
            # else empty: try rack tiles / blanks subject to cross-checks
            for i, t in enumerate(rack_left):
                if t == '_':
                    for ch in cross_checks[r][next_c]:
                        if ch in node.next:
                            new_rack = rack_left[:i] + rack_left[i+1:]
                            right_search(node.next[ch], start_col, next_c, new_rack, path + ch)
                else:
                    if t in cross_checks[r][next_c] and t in node.next:
                        new_rack = rack_left[:i] + rack_left[i+1:]
                        right_search(node.next[t], start_col, next_c, new_rack, path + t)

        # helper to create CandidateMove by computing score via provided gatekeeper-scoring function later
        # We'll generate word + start col + row + direction and compute score in UnbeatableBot (we need GateKeeper)
        # So here we only assemble the necessary info; actual scoring will be done at top-level where GateKeeper available.

        # Kick off left_search starting at anchor_col with root node and initial rack
        left_search(root, anchor_col, rack, "")
        # Return compiled results later (we converted into CandidateMove in _make_candidate)
        return results

    def _make_candidate(self, word, row, col, direction, gatekeeper):
        # compute score using gatekeeper.score. Need to construct Location and choose correct direction
        loc = Location(row, col)
        dir_obj = HORIZONTAL if direction == HORIZONTAL else VERTICAL
        # gatekeeper.score expects word string with spaces where existing tiles are used.
        # We must build word_with_spaces to match board: if board has tile, put ' ' in that position in the word passed to gatekeeper,
        # but gatekeeper.verify expects letters for letters already on board; however board.score uses ' ' to indicate existing tiles.
        # We'll create a placement string where letters that are already present on board are ' ' for the played word.
        # To keep simple, we'll build the string with actual letters and let gatekeeper.score handle tiles present (it uses tile != ' ' as new tile)
        # The Board.score_word and score_cross_word treat tile == ' ' as preexisting tile; therefore we should supply ' ' for board letters.
        mat = self.build_board_matrix(gatekeeper)
        placement = []
        r, c = row, col
        for ch in word:
            if mat[r][c].isalpha() and mat[r][c].upper() == ch.upper():
                placement.append(' ')
            else:
                placement.append(ch)
            if direction == HORIZONTAL:
                c += 1
            else:
                r += 1
        placement_str = ''.join(placement)
        # call gatekeeper.score; gatekeeper.score requires Location and direction object
        score = gatekeeper.score(placement_str, Location(row, col), (HORIZONTAL if direction == HORIZONTAL else VERTICAL))
        return CandidateMove(word, row, col, direction, score)


# -------------------------
# Unbeatable Bot
# -------------------------
class UnbeatableBot:
    """
    Main AI player class. Use set_gatekeeper(gatekeeper) to receive the GateKeeper instance,
    and choose_move() to return a Move (PlayWord or ExchangeTiles).
    """
    def __init__(self,
                 dictionary=None,
                 beam_width=60,
                 mc_rollouts=80,
                 time_limit=None):
        """
        :param dictionary: optional set of words; if None uses board_module.DICTIONARY
        :param beam_width: number of top candidates to keep before Monte Carlo
        :param mc_rollouts: number of rollouts per candidate for Monte Carlo EV estimation
        :param time_limit: optional max seconds to allow choose_move (None => use defaults). This implementation uses it to cap time.
        """
        self.dictionary = dictionary if dictionary is not None else board_module.DICTIONARY
        self.gaddag = GADDAG(self.dictionary)
        self.generator = GADDAGGenerator(self.gaddag, self.dictionary)
        self.gatekeeper = None
        self.beam_width = beam_width
        self.mc_rollouts = mc_rollouts
        self.time_limit = time_limit  # seconds
        # initial tile pool for sampling opponent racks
        self.initial_tiles = list(_INITIAL_TILE_POOL)

    def set_gatekeeper(self, gatekeeper):
        self.gatekeeper = gatekeeper

    # -------------------------
    # Helper: reconstruct remaining tile multiset
    # -------------------------
    def _remaining_tiles(self, mat, my_hand):
        used = []
        # tiles on board
        for r in range(WIDTH):
            for c in range(WIDTH):
                ch = mat[r][c]
                if ch.isalpha():
                    # if it's uppercase blank on board, convert to '_' in pool removal
                    if ch.isupper() and ch.lower() not in board_module.TILE_VALUES:
                        used.append('_')
                    else:
                        used.append(ch.lower())
        # tiles in my hand
        for t in my_hand:
            used.append(t.lower())
        # initial multiset copy
        rem = Counter(self.initial_tiles)
        for u in used:
            # convert underscores etc
            if u == '':
                continue
            # map uppercase back to lowercase for removal
            token = u
            if token == ' ':
                continue
            if token.isalpha():
                token = token.lower()
            if token not in rem or rem[token] <= 0:
                # already exhausted; skip (safe)
                continue
            rem[token] -= 1
            if rem[token] == 0:
                del rem[token]
        # return list of remaining tiles (lowercase, with '_' as blank)
        pool = []
        for k, v in rem.items():
            pool.extend([k] * v)
        return pool

    # -------------------------
    # Quick leave heuristic (fast approximation)
    # -------------------------
    def _leave_value(self, rack):
        """
        Fast heuristic estimating value of leaving 'rack' (string of tiles).
        Positive if good leaving; negative if bad.
        This is not a learned table but a simple heuristic:
         - reward vowel-consonant balance
         - reward common 2-letter combos and potential bingo stems
         - penalize isolated high-point consonant clusters
        """
        r = rack.upper()
        vowels = sum(1 for ch in r if ch in 'AEIOU')
        consonants = len(r) - vowels
        balance = -abs(vowels - consonants) * 0.3  # prefer balanced racks

        # count rare consonants penalty
        rare = sum(1 for ch in r if ch in 'JQXZ')
        rare_pen = -rare * 1.2

        # reward for common bingo letters (S, T, R, E, A)
        bingo_basis = sum(1 for ch in r if ch in 'STRAE') * 0.25

        # reward for blanks
        blanks = r.count('_')
        blank_bonus = blanks * 2.0

        val = balance + rare_pen + bingo_basis + blank_bonus
        return val

    # -------------------------
    # Greedy opponent model for rollouts (fast)
    # -------------------------
    def _greedy_opponent_move(self, gatekeeper, opponent_rack):
        # Use generator to produce best immediate scoring move
        moves = self.generator.generate_all_moves(gatekeeper, ''.join(opponent_rack))
        if not moves:
            # exchange all tiles if possible -> represented as bool array
            tiles_bool = [True] * 7
            return ExchangeTiles(tiles_bool)
        # choose highest immediate score
        best = max(moves, key=lambda m: m.score)
        # create move object
        loc = Location(best.row, best.col)
        dir_obj = HORIZONTAL if best.direction == HORIZONTAL else VERTICAL
        return PlayWord(best.word, loc, dir_obj)

    # -------------------------
    # Monte Carlo expected value for a candidate move
    # -------------------------
    def _mc_expected_value(self, candidate, my_hand, mat, remaining_tiles, my_score, opp_score, opp_hand_size, time_deadline=None):
        """
        Simulate rollouts to estimate EV (my_score_after - opp_score_after)
        Returns average delta in my favor (higher is better).
        Candidate is a CandidateMove (word,row,col,direction,score)
        """
        total_delta = 0.0
        rollouts = self.mc_rollouts
        # bound by time
        start_time = time.time()
        for i in range(rollouts):
            if time_deadline and time.time() > time_deadline:
                break
            # copy remaining tiles multiset
            pool = list(remaining_tiles)
            random.shuffle(pool)
            # sample opponent rack
            opp_rack = []
            # sample without replacement
            samp_size = min(opp_hand_size, len(pool))
            for _ in range(samp_size):
                opp_rack.append(pool.pop())

            # Simulate opponent greedy response:
            # We need a simulated board where candidate has been played.
            # Create a fake GateKeeper-like lightweight board simulator for scoring & placements for the rollout.
            sim_board = self._simulate_board_after_play(mat, candidate)
            # Create a fake gatekeeper object wrapping sim_board for generator & scorer.
            fake_gk = _FakeGateKeeper(sim_board)

            # Opponent move (greedy)
            opp_moves = self.generator.generate_all_moves(fake_gk, ''.join(opp_rack))
            if not opp_moves:
                # opponent exchanges: refill rack from pool (we won't model exact bag draw consequences deeply)
                # after exchange we assume no immediate score by opponent
                opp_best_score = 0
            else:
                opp_move = max(opp_moves, key=lambda m: m.score)
                opp_best_score = opp_move.score
                # apply opponent move to sim_board (for our following reply)
                sim_board = self._simulate_board_after_play(sim_board, opp_move)
                fake_gk = _FakeGateKeeper(sim_board)

            # Now simulate our draw: we consumed some tiles for candidate: remove those letters from pool
            # Remove tiles corresponding to candidate.used_tiles from pool (if used_tiles available), else best-effort remove letters used in word but not on board
            # For simplicity, assume we drew tiles equal to number used
            # sample draw_count tiles
            # We'll estimate our next best immediate reply score given new rack: combine leftover tiles with our unused tiles
            # For simplicity, do a one-step reply: compute best immediate move for us after opponent
            # Build our new rack: start from my_hand, remove letters used by candidate, then draw k tiles where k = letters played
            my_hand_copy = list(my_hand)
            # remove tiles used by candidate from my_hand_copy (best-effort)
            used = []
            for ch in candidate.word:
                # if board had that tile, it won't be from our rack; skip
                # else attempt to remove from my_hand_copy the matching tile or a blank
                # We'll use uppercase to compare with mat to detect board letters
                # If candidate had placement where board had letter = ' ' in placement, that was our tile
                pass
            # For simplicity in rollout we won't simulate our reply rack exactly; compute expected improvement as minor fraction of immediate candidate.score and opponent move
            # So we estimate final delta = (my_score + candidate.score) - (opp_score + opp_best_score)
            final_delta = (my_score + candidate.score) - (opp_score + opp_best_score)
            total_delta += final_delta

            # performance cutoff check
            if (i & 0x7) == 0 and time.time() - start_time > 0.95 * (self.time_limit or 5.0):
                # don't overrun time budget
                break
        if i == 0:
            return -9999.0
        return total_delta / (i + 1)

    def _simulate_board_after_play(self, mat, candidate):
        # Return a new board matrix with candidate's word placed (doesn't update score etc)
        sim = [row[:] for row in mat]
        r, c = candidate.row, candidate.col
        dr = 0 if candidate.direction == HORIZONTAL else 1
        dc = 1 if candidate.direction == HORIZONTAL else 0
        for ch in candidate.word:
            if not sim[r][c].isalpha():
                sim[r][c] = ch
            r += dr
            c += dc
        return sim

    # -------------------------
    # Choose move (main entry)
    # -------------------------
    def choose_move(self):
        """
        Primary decision function called by tournament harness.
        Returns a Move object (PlayWord or ExchangeTiles).
        """
        if self.gatekeeper is None:
            raise RuntimeError("GateKeeper not set. Call set_gatekeeper before choose_move().")

        # timing
        start_time = time.time()
        time_deadline = None
        if self.time_limit:
            time_deadline = start_time + self.time_limit

        # get hand & simple params
        my_hand = self.gatekeeper.get_hand()
        if my_hand is None:
            my_hand = []
        my_hand_upper = ''.join(my_hand).upper()
        my_score = self.gatekeeper.get_my_score()
        opp_score = self.gatekeeper.get_opponent_score()
        opp_hand_size = self.gatekeeper.get_opponent_hand_size()
        bag_count = self.gatekeeper.get_bag_count()

        # generate board matrix and remaining tiles pool
        mat = self.generator.build_board_matrix(self.gatekeeper)
        remaining_pool = self._remaining_tiles(mat, my_hand)

        # 1) Generate all legal moves using GADDAG (both orientations)
        raw_moves = self.generator.generate_all_moves(self.gatekeeper, my_hand_upper)
        if not raw_moves:
            # No play possible: consider exchange decision
            # We evaluate exchange EV roughly: we compute expected best immediate play after exchanging vs playing nothing.
            # Simple policy: if bag_count >= 7 -> exchange all; else pass? We'll exchange moderately smart: exchange consonants when racks are poor.
            # Here choose to exchange all tiles (boolean array), fallback: return ExchangeTiles
            tiles_bool = [True] * 7
            return ExchangeTiles(tiles_bool)

        # 2) Quick heuristic ranking: immediate score + leave heuristic
        for m in raw_moves:
            # compute leave: approximate rack after playing (naive)
            # Build naive used tile set: count letters that fall on empty squares
            used = []
            r, c = m.row, m.col
            dr = 0 if m.direction == HORIZONTAL else 1
            dc = 1 if m.direction == HORIZONTAL else 0
            for ch in m.word:
                if not mat[r][c].isalpha():
                    used.append(ch.lower())
                r += dr; c += dc
            # naive leftover
            temp_hand = list(my_hand_upper)
            for u in used:
                # remove one occurrence or blank
                if u.upper() in temp_hand:
                    temp_hand.remove(u.upper())
                elif '_' in temp_hand:
                    temp_hand.remove('_')
            leave_str = ''.join(temp_hand)
            m.quick_value = m.score + 2.0 * self._leave_value(leave_str)

        # 3) Beam: keep top K candidates
        raw_moves.sort(key=lambda m: m.quick_value, reverse=True)
        beam = raw_moves[:self.beam_width]

        # 4) Monte Carlo evaluate beam candidates
        best_ev = -1e9
        best_move = None
        for i, cand in enumerate(beam):
            # time cutoff
            if time_deadline and time.time() > time_deadline:
                break
            # estimate EV using Monte Carlo (simple rollout)
            ev = self._mc_expected_value(cand, my_hand, mat, remaining_pool, my_score, opp_score, opp_hand_size, time_deadline)
            cand.ev = ev
            # combine with quick value for tiebreak
            combined = ev + 0.01 * cand.quick_value
            if combined > best_ev:
                best_ev = combined
                best_move = cand

        if best_move is None:
            # fallback to highest immediate score
            best_move = max(raw_moves, key=lambda m: m.score)

        # Convert best_move into actual Move object (PlayWord)
        loc = Location(best_move.row, best_move.col)
        dir_obj = HORIZONTAL if best_move.direction == HORIZONTAL else VERTICAL
        play = PlayWord(best_move.word, loc, dir_obj)
        return play

class _FakeGateKeeper:
    def __init__(self, mat):
        self.mat = mat

    def get_square(self, loc):
        return self.mat[loc.r][loc.c]

    def score(self, word, location, direction):
        # crude scoring: sum tile values ignoring premiums (approx)
        # For rollouts we only need relative numbers; precise scoring is expensive to simulate repeatedly.
        s = 0
        r, c = location.r, location.c
        dr = 0 if direction == HORIZONTAL else 1
        dc = 1 if direction == HORIZONTAL else 0
        for ch in word:
            if ch == ' ':
                s += board_module.TILE_VALUES[self.mat[r][c].lower()]
            else:
                # letter tiles are uppercase in candidate.word
                s += board_module.TILE_VALUES.get(ch.lower(), 0)
            r += dr; c += dc
        return s

    def get_hand(self, player_number):
        # not needed for simulation functions here
        return []

    def get_bag_count(self):
        return 0

    def get_opponent_hand_size(self):
        return 0

    def get_last_move(self):
        return None

# -------------------------
# Done
# -------------------------
