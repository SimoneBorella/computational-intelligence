from typing import Tuple, List
from enum import Enum
import math
import numpy as np
from bitarray import bitarray
from tabulate import tabulate
from queue import Queue 
from threading import Thread, Event
from time import time



N = 4
if N < 5:
    SHIFT = 16
else:
    SHIFT = 32



# Render
    
def print_state(state: np.array) -> None:
    board = np.zeros((N, N), dtype=str)
    for r in range(N):
        for c in range(N):
            if state[r, c] == -1:
                board[r, c] = "-"
            elif state[r, c] == 0:
                board[r, c] = "X"
            elif state[r, c] == 1:
                board[r, c] = "O"

    board = tabulate(board, tablefmt="fancy_grid")
    print(board)




# Encoding and decoding

def encode_state(state: np.array):
    encoded_state = np.int32(0) if N<5 else np.int64(0)

    for r in range(N):
        for c in range(N):
            pos = rc_to_pos(r, c)
            # X in (r, c)
            if state[r, c] == 0:
                encoded_state = encoded_state | (1<<pos+SHIFT)
            # O in (r, c)
            elif state[r, c] == 1:
                encoded_state = encoded_state | (1<<pos)
    return encoded_state
    
def decode_state(state: np.int64) -> np.array:
    decoded_state = np.ones((N, N), dtype=np.uint8) * -1
    # X writing
    for pos in range(N*N):
        if state & (1<<(pos+SHIFT)) != 0:
            r, c = pos_to_rc(pos)
            decoded_state[r, c] = 0
    
    # O writing
    for pos in range(N*N):
        if state & (1<<pos) != 0:
            r, c = pos_to_rc(pos)
            decoded_state[r, c] = 1
    
    return decoded_state

def rc_to_pos(row: int, col: int) -> int:
    return (N-1-row)*N + (N-1-col)

def pos_to_rc(pos: int) -> Tuple[int, int]:
    return (N-1-pos//N), (N-1-pos%N)









# Moves

class Move(Enum):
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


def swap(state):
    mask = 2**(N*N) - 1
    return (state & mask) << SHIFT | ((state & (mask << SHIFT)) >> SHIFT)




def move(state, action: Tuple[Tuple[int, int], Move], player: int):
    (r, c), move = action

    mask = 2**(N*N) - 1
    o = state & mask
    x = (state & (mask<<SHIFT)) >> SHIFT

    if move == Move.RIGHT:
        final_pos = rc_to_pos(r, N-1-c)
        if player == 0:
            final_pos += SHIFT

        r_mask = np.int32(0) if N<5 else np.int64(0)

        for j in range(N):
            r_mask |= 1 << rc_to_pos(r, j)

        moved_x = (((x & r_mask) << 1) & r_mask) | (x & ~r_mask)
        moved_o = (((o & r_mask) << 1) & r_mask) | (o & ~r_mask)

        new_state = moved_x << SHIFT | moved_o | 1 << final_pos

    elif move == Move.LEFT:
        final_pos = rc_to_pos(r, N-1-c)
        if player == 0:
            final_pos += SHIFT

        r_mask = np.int32(0) if N<5 else np.int64(0)

        for j in range(N):
            r_mask |= 1 << rc_to_pos(r, j)

        moved_x = (((x & r_mask) >> 1) & r_mask) | (x & ~r_mask)
        moved_o = (((o & r_mask) >> 1) & r_mask) | (o & ~r_mask)

        new_state = moved_x << SHIFT | moved_o | 1 << final_pos

    elif move == Move.TOP:
        final_pos = rc_to_pos(N-1-r, c)
        if player == 0:
            final_pos += SHIFT

        c_mask = np.int32(0) if N<5 else np.int64(0)

        for i in range(N):
            c_mask |= 1 << rc_to_pos(i, c)

        moved_x = (((x & c_mask) >> N) & c_mask) | (x & ~c_mask)
        moved_o = (((o & c_mask) >> N) & c_mask) | (o & ~c_mask)

        new_state = moved_x << SHIFT | moved_o | 1 << final_pos
        
    elif move == Move.BOTTOM:
        final_pos = rc_to_pos(N-1-r, c)
        if player == 0:
            final_pos += SHIFT

        c_mask = np.int32(0) if N<5 else np.int64(0)

        for i in range(N):
            c_mask |= 1 << rc_to_pos(i, c)

        moved_x = (((x & c_mask) << N) & c_mask) | (x & ~c_mask)
        moved_o = (((o & c_mask) << N) & c_mask) | (o & ~c_mask)

        new_state = moved_x << SHIFT | moved_o | 1 << final_pos
            
    return new_state



def get_available_moves(state, player_id):

    state = decode_state(state)
    
    available_moves = []
    for i in range(N):
        if state[N-1][i] == player_id or state[N-1][i] == -1:
            available_moves.append(((N-1, i), Move.TOP))

    for i in range(N):
        if state[0][i] == player_id or state[0][i] == -1:
            available_moves.append(((0, i), Move.BOTTOM))

    for i in range(N):
        if state[i][N-1] == player_id or state[i][N-1] == -1:
            available_moves.append(((i, N-1), Move.LEFT))

    for i in range(N):
        if state[i][0] == player_id or state[i][0] == -1:
            available_moves.append(((i, 0), Move.RIGHT))

    return available_moves


def check_winner(state: np.int64, player: int) -> bool:
    for r in range(N):
        row_mask = np.int32(0) if N<5 else np.int64(0)

        for c in range(N):
            row_mask |= 1 << rc_to_pos(r, c)

        if player == 0:
            row_mask = row_mask << SHIFT

        if state & row_mask == row_mask:
            return True
        
    for c in range(N):
        col_mask = np.int32(0) if N<5 else np.int64(0)

        for r in range(N):
            col_mask |= 1 << rc_to_pos(r, c)
        
        if player == 0:
            col_mask = col_mask << SHIFT
            
        if state & col_mask == col_mask:
            return True
    
    diag_mask = np.int32(0) if N<5 else np.int64(0)
    for i in range(N):
        diag_mask |= 1 << rc_to_pos(i, i)

    if player == 0:
        diag_mask = diag_mask << SHIFT
        
    if state & diag_mask == diag_mask:
        return True
    
    rev_diag_mask = np.int32(0) if N<5 else np.int64(0)
    for i in range(N):
        rev_diag_mask |= 1 << rc_to_pos(i, N-1-i)

    if player == 0:
        rev_diag_mask = rev_diag_mask << SHIFT
        
    if state & rev_diag_mask == rev_diag_mask:
        return True

    return False













# Bijections

def compute_pop(i):
    pop = 0
    for pos in range(N*N):
        if i & (1 << pos) == (1 << pos):
            pop += 1
    return pop


def compute_pop_ord_table():
    pop_ord_table = {}
    pop_count = {}

    for i in range(2**(N*N)):
        pop = compute_pop(i)

        if pop in pop_count:
            pop_count[pop] += 1
        else:
            pop_count[pop] = 1
        
        ord = pop_count[pop] -1

        pop_ord_table[i] = (pop, ord)

    return pop_ord_table


pop_ord_table = compute_pop_ord_table()


def get_state_class_id(state: np.int64) -> int:
    mask = 2**(N*N) - 1
    o = state & mask
    x = (state & (mask<<SHIFT)) >> SHIFT

    shifted_o = o

    shift_count = 0
    shift_amount = 0
    for pos in range(N*N):
        if (x & 1 << pos) == 1 << pos:
            shift_count += 1
        elif shift_count > 0:
            shift_mask = 0b0
            for p in range(pos-shift_count-shift_amount):
                shift_mask |= (1 << p)
            shift_mask = ~shift_mask & mask

            shifted_o = ((shifted_o >> shift_count) & shift_mask) | (shifted_o & ~shift_mask)

            shift_amount += shift_count
            shift_count = 0

    global pop_ord_table

    cx = pop_ord_table[x][0]
    co = pop_ord_table[o][0]
    
    sx = x
    so = shifted_o

    ord_sx = pop_ord_table[sx][1]
    ord_so = pop_ord_table[so][1]

    return ord_sx * math.comb((N*N)-cx, co) + ord_so

def get_class(state: np.int64) -> Tuple[int, int]:
    mask = 2**(N*N) - 1
    o = state & mask
    x = (state & (mask<<SHIFT)) >> SHIFT

    cx = pop_ord_table[x][0]
    co = pop_ord_table[o][0]

    return cx, co

def states_in_class(cx: int, co: int) -> int:
    return math.comb(N*N, cx) * math.comb(N*N-cx, co)











# Class states computations

def compute_class_states(cmd_queue, states_queue, event, cx, co):

    def dispositions(pos, sol_x, mark_x, sol_o):
        if pos >= N*N:
            sol = (sol_x<<SHIFT) | sol_o

            event.wait()

            if not cmd_queue.empty():
                return

            states_queue.put(sol)
            event.clear()

            return
        
        for i in range(2):
            if mark_x[i] > 0:
                if i == 0:
                    if (sol_o & 1<<pos)!=0:
                        continue
                    sol_x |= 1<<pos
                else:
                    sol_x &= ~(1<<pos)

                mark_x[i] -= 1
                dispositions(pos+1, sol_x, mark_x, sol_o)
                if not cmd_queue.empty():
                    return
                mark_x[i] += 1


    def state_dispositions(pos, cx, co, sol_o, mark_o):
        if pos >= N*N:
            sol_x = np.int16(0) if N<5 else np.int32(0)
            mark_x = [cx, N*N-cx]
            dispositions(0, sol_x, mark_x, sol_o)
            return
        
        for i in range(2):
            if mark_o[i] > 0:
                if i == 0:
                    sol_o |= 1<<pos
                else:
                    sol_o &= ~(1<<pos)

                mark_o[i] -= 1
                state_dispositions(pos+1, cx, co, sol_o, mark_o)
                if not cmd_queue.empty():
                    return
                mark_o[i] += 1

    sol_o = np.int16(0) if N<5 else np.int32(0)
    mark_o = [co, N*N-co]
    state_dispositions(0, cx, co, sol_o, mark_o)
    if not cmd_queue.empty():
        cmd_queue.get()
    else:
        event.wait()
        if not cmd_queue.empty():
            return
        states_queue.put(None)
        event.clear()


class ClassStateIterator:
    def __init__(self, cx: int, co: int):
        self.cx = cx
        self.co = co
        self.cmd_queue = Queue()
        self.states_queue = Queue()
        self.event = Event()

        self.class_states_thread = Thread(target=compute_class_states, args=(self.cmd_queue, self.states_queue, self.event, self.cx, self.co))
        self.class_states_thread.start()


    def __iter__(self):
        return self
    
    def __next__(self):
        self.event.set()
        state = self.states_queue.get()
        if state is None:
            raise StopIteration
        return state
    





# Outcomes structure
if N<5:
    MAX_STATES_NUMBER = states_in_class(5, 6)
else:
    MAX_STATES_NUMBER = states_in_class(8, 8)


class Outcomes:
    def __init__(self) -> None:
        self.outcomes = bitarray('0'* 2*MAX_STATES_NUMBER)
                
    def get_state_outcome(self, id):
        return self.outcomes[2*id], self.outcomes[2*id + 1]
    
    def set_state_outcome(self, id, val):
        self.outcomes[2*id] = val[0]
        self.outcomes[2*id + 1] = val[1]
    
    def init_values(self, cx, co):
        for id in range(states_in_class(cx, co)):
            self.set_state_outcome(id, (0, 0))

    # Check state
    def is_draw(self, state):
        id = get_state_class_id(state)
        val = self.get_state_outcome(id)
        return val[0] == 0 and val[1] == 0
    
    def is_win(self, state):
        id = get_state_class_id(state)
        val = self.get_state_outcome(id)
        return val[0] == 0 and val[1] == 1
    
    def is_loss(self, state):
        id = get_state_class_id(state)
        val = self.get_state_outcome(id)
        return val[0] == 1 and val[1] == 1
    

    # Update state
    def update_draw(self, state):
        id = get_state_class_id(state)
        self.set_state_outcome(id, (0, 0))

    def update_win(self, state):
        id = get_state_class_id(state)
        self.set_state_outcome(id, (0, 1))

    def update_loss(self, state):
        id = get_state_class_id(state)
        self.set_state_outcome(id, (1, 1))
    
    def read(self, cx, co):
        with open(f'./outcomes_{N}x{N}/c_{cx}_{co}.bin', 'rb') as file:
            byte = file.read(1)
            id = 0
            while byte:
                buffer = bitarray()
                buffer.frombytes(byte)
                i = 0
                while i < 8 and id < states_in_class(cx, co):
                    val = (buffer[i], buffer[i+1])
                    self.set_state_outcome(id, val)
                    id += 1
                    i += 2

                # if id >= MAX_STATES_NUMBER:
                if id >= states_in_class(cx, co):
                    break
                byte = file.read(1)


    def write(self, cx, co):
        with open(f'./outcomes_{N}x{N}/c_{cx}_{co}.bin', 'wb') as file:
            buffer = bitarray()
            bit_count = 0
            for id in range(states_in_class(cx, co)):
                val = self.get_state_outcome(id)
                buffer.append(val[0])
                buffer.append(val[1])
                bit_count += 2
                if len(buffer) >= 8:
                    file.write(buffer.tobytes())
                    buffer = bitarray()
                    bit_count = 0
            if bit_count != 0:
                file.write(buffer.tobytes())






# Value iteration algorithm with backward induction

def compute_all_states_outcome():
    outcomes = Outcomes()
    outcomes_rev = Outcomes()
    outcomes_next = Outcomes()
    outcomes_next_rev = Outcomes()

    # For each number of filled tile
    # for n in range(N*N, -1, -1):
    for n in range(8, -1, -1):
        print(f"Computing class level {n} - Started")
    
        # For each class
        for cx in range(0, (n//2) + 1):
            co = n - cx
            print(f"Computing class ({cx}, {co}) and ({co}, {cx}) - Started")
            

            outcomes.init_values(cx, co)
            if cx != co:
                outcomes_rev.init_values(co, cx)


            if n < N*N:
                outcomes_next_rev.read(co, cx+1)
                if cx != co:
                    outcomes_next.read(cx, co+1)

            # Compute outcome of terminal states
            for state in ClassStateIterator(cx, co):
                # Win outcome
                if check_winner(state, 0):
                    outcomes.update_win(state)
                # Loss outcome
                elif check_winner(state, 1):
                    outcomes.update_loss(state)


            # Reversed
            if cx != co:
                for state in ClassStateIterator(co, cx):
                    # Win outcome
                    if check_winner(state, 0):
                        outcomes_rev.update_win(state)
                    # Loss outcome
                    elif check_winner(state, 1):
                        outcomes_rev.update_loss(state)
                    


            # Compute outcome of draw state until converging (no more update)
            update = True
            while update:
                update = False
                for state in ClassStateIterator(cx, co):
                    if outcomes.is_draw(state):
                        # For each child belonging to the same class of state, if are all Win then state is Loss
                        loss = True
                        win = False
                        for m in get_available_moves(state, 0):
                            child_state = swap(move(state, m, 0))
                            ccx, cco = get_class(child_state)
                            if cx != co:
                                # Child belonging to the same class (update from outcomes)
                                if ccx == co and cco == cx:
                                    if not outcomes_rev.is_win(child_state):
                                        loss = False
                                    if outcomes_rev.is_loss(child_state):
                                        win = True
                                        break
                                # Child belonging to next class (update from next outcomes)
                                else:
                                    if not outcomes_next_rev.is_win(child_state):
                                        loss = False
                                    if outcomes_next_rev.is_loss(child_state):
                                        win = True
                                        break
                            else:
                                # Child belonging to the same class (update from outcomes)
                                if ccx == co and cco == cx:
                                    if not outcomes.is_win(child_state):
                                        loss = False
                                    if outcomes.is_loss(child_state):
                                        win = True
                                        break
                                # Child belonging to next class (update from next outcomes)
                                else:
                                    if not outcomes_next_rev.is_win(child_state):
                                        loss = False
                                    if outcomes_next_rev.is_loss(child_state):
                                        win = True
                                        break
                        
                        if win:
                            # Set Win state
                            outcomes.update_win(state)
                            update = True
                            
                        elif loss:
                            # Set Loss state
                            outcomes.update_loss(state)
                            update = True

                # Reversed
                if cx != co:
                    for state in ClassStateIterator(co, cx):
                        if outcomes_rev.is_draw(state):
                            # For each child belonging to the same class of state, if are all Win then state is Loss
                            loss = True
                            win = False
                            for m in get_available_moves(state, 0):
                                child_state = swap(move(state, m, 0))
                                ccx, cco = get_class(child_state)
                                # Child belonging to the same class (update from outcomes)
                                if ccx == cx and cco == co:
                                    if not outcomes.is_win(child_state):
                                        loss = False
                                    if outcomes.is_loss(child_state):
                                        win = True
                                        break
                                # Child belonging to next class (update from next outcomes)
                                else:
                                    if not outcomes_next.is_win(child_state):
                                        loss = False
                                    if outcomes_next.is_loss(child_state):
                                        win = True
                                        break
                            
                            if win:
                                # Set Win state
                                outcomes_rev.update_win(state)
                                update = True
                                
                            elif loss:
                                # Set Loss state
                                outcomes_rev.update_loss(state)
                                update = True
                
            
            outcomes.write(cx, co)
            if cx != co:
                outcomes_rev.write(co, cx)


if __name__ == "__main__":
    start = time()
    compute_all_states_outcome()
    end = time()

    print(f"Computation time: {end - start} s")

