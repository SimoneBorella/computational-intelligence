#include <string>
#include <tuple>
#include <chrono>
#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <queue>
#include <map>
#include <bitset>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define _N 4
const int N = _N;

#if _N == 4
const long int MAX_STATES_NUMBER = 2018016;
const long int SHIFT = 16;
typedef int32_t state_t;
typedef Eigen::Matrix<int, 4, 4> matrix_state_t;
#else
#if _N == 5
const long int MAX_STATES_NUMBER = 26293088250;
const int SHIFT = 32;
typedef int64_t state_t;
typedef Eigen::Matrix<int, 5, 5> matrix_state_t;
#endif
#endif

const int MUTEX_NUMBER = 1 << 15;
const int THREADS_NUMBER = 8;

enum Move
{
    TOP = 0,
    BOTTOM = 1,
    LEFT = 2,
    RIGHT = 3
};

// Game render, logic and moves

void print_state(const matrix_state_t& state) {
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (state(r, c) == -1) {
                std::cout << "-";
            } else if (state(r, c) == 0) {
                std::cout << "X";
            } else if (state(r, c) == 1) {
                std::cout << "O";
            }

            std::cout << " ";
        }
        std::cout << "\n";
    }
}

int rc_to_pos(int row, int col) {
    return (N - 1 - row) * N + (N - 1 - col);
}

std::tuple<int, int> pos_to_rc(int pos) {
    return std::make_tuple(N - 1 - (pos / N), N - 1 - (pos % N));
}

state_t encode_state(const matrix_state_t& state) {
    state_t encoded_state = 0;

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int pos = rc_to_pos(r, c);

            if (state(r, c) == 0) {
                encoded_state |= (1 << (pos + SHIFT));
            }

            else if (state(r, c) == 1) {
                encoded_state |= (1 << pos);
            }
        }
    }

    return encoded_state;
}


matrix_state_t decode_state(state_t state) {
    matrix_state_t decoded_state;
    decoded_state.fill(-1);

    for (int pos = 0; pos < N * N; ++pos) {
        if ((state & (1 << (pos + SHIFT))) != 0) {
            int r, c;
            std::tie(r, c) = pos_to_rc(pos);
            decoded_state(r, c) = 0;
        }
    }

    for (int pos = 0; pos < N * N; ++pos) {
        if ((state & (1 << pos)) != 0) {
            int r, c;
            std::tie(r, c) = pos_to_rc(pos);
            decoded_state(r, c) = 1;
        }
    }

    return decoded_state;
}


std::vector<std::tuple<std::tuple<int, int>, Move>> get_available_moves(const state_t& state, int player_id) {
    std::vector<std::tuple<std::tuple<int, int>, Move>> available_moves;
    matrix_state_t decoded_state = decode_state(state);

    for (int i = 0; i < N; ++i) {
        if (decoded_state(N - 1, i) == player_id || decoded_state(N - 1, i) == -1) {
            available_moves.push_back(std::make_tuple(std::make_tuple(N - 1, i), Move::TOP));
        }
    }

    for (int i = 0; i < N; ++i) {
        if (decoded_state(0, i) == player_id || decoded_state(0, i) == -1) {
            available_moves.push_back(std::make_tuple(std::make_tuple(0, i), Move::BOTTOM));
        }
    }

    for (int i = 0; i < N; ++i) {
        if (decoded_state(i, N - 1) == player_id || decoded_state(i, N - 1) == -1) {
            available_moves.push_back(std::make_tuple(std::make_tuple(i, N - 1), Move::LEFT));
        }
    }

    for (int i = 0; i < N; ++i) {
        if (decoded_state(i, 0) == player_id || decoded_state(i, 0) == -1) {
            available_moves.push_back(std::make_tuple(std::make_tuple(i, 0), Move::RIGHT));
        }
    }

    return available_moves;
}


state_t move(state_t state, std::tuple<std::tuple<int, int>, Move> action, int player) {
    std::tuple<int, int> pos = std::get<0>(action);
    Move move = std::get<1>(action);

    state_t mask = (1 << (N * N)) - 1;
    state_t o = state & mask;
    state_t x = (state & (mask << SHIFT)) >> SHIFT;

    state_t new_state = 0;

    if (move == Move::RIGHT) {
        int final_pos = rc_to_pos(std::get<0>(pos), N - 1 - std::get<1>(pos));
        if (player == 0) {
            final_pos += SHIFT;
        }

        state_t r_mask = (N < 5) ? static_cast<state_t>(0) : static_cast<state_t>(0);

        for (int j = 0; j < N; ++j) {
            r_mask |= 1 << rc_to_pos(std::get<0>(pos), j);
        }

        state_t moved_x = (((x & r_mask) << 1) & r_mask) | (x & ~r_mask);
        state_t moved_o = (((o & r_mask) << 1) & r_mask) | (o & ~r_mask);

        new_state = (moved_x << SHIFT) | moved_o | (1 << final_pos);
    }
    else if (move == Move::LEFT) {
        int final_pos = rc_to_pos(std::get<0>(pos), N - 1 - std::get<1>(pos));
        if (player == 0) {
            final_pos += SHIFT;
        }

        state_t r_mask = (N < 5) ? static_cast<state_t>(0) : static_cast<state_t>(0);

        for (int j = 0; j < N; ++j) {
            r_mask |= 1 << rc_to_pos(std::get<0>(pos), j);
        }

        state_t moved_x = (((x & r_mask) >> 1) & r_mask) | (x & ~r_mask);
        state_t moved_o = (((o & r_mask) >> 1) & r_mask) | (o & ~r_mask);

        new_state = (moved_x << SHIFT) | moved_o | (1 << final_pos);
    }
    else if (move == Move::TOP) {
        int final_pos = rc_to_pos(N - 1 - std::get<0>(pos), std::get<1>(pos));
        if (player == 0) {
            final_pos += SHIFT;
        }

        state_t c_mask = (N < 5) ? static_cast<state_t>(0) : static_cast<state_t>(0);

        for (int i = 0; i < N; ++i) {
            c_mask |= 1 << rc_to_pos(i, std::get<1>(pos));
        }

        state_t moved_x = (((x & c_mask) >> N) & c_mask) | (x & ~c_mask);
        state_t moved_o = (((o & c_mask) >> N) & c_mask) | (o & ~c_mask);

        new_state = (moved_x << SHIFT) | moved_o | (1 << final_pos);
    }
    else if (move == Move::BOTTOM) {
        int final_pos = rc_to_pos(N - 1 - std::get<0>(pos), std::get<1>(pos));
        if (player == 0) {
            final_pos += SHIFT;
        }

        state_t c_mask = (N < 5) ? static_cast<state_t>(0) : static_cast<state_t>(0);

        for (int i = 0; i < N; ++i) {
            c_mask |= 1 << rc_to_pos(i, std::get<1>(pos));
        }

        state_t moved_x = (((x & c_mask) << N) & c_mask) | (x & ~c_mask);
        state_t moved_o = (((o & c_mask) << N) & c_mask) | (o & ~c_mask);

        new_state = (moved_x << SHIFT) | moved_o | (1 << final_pos);
    }

    return new_state;
}


state_t swap(state_t state) {
    state_t mask = (1 << (N * N)) - 1;
    return ((state & mask) << SHIFT) | (((state & (mask << SHIFT)) >> SHIFT) & mask);
}

bool check_winner(state_t state, int player) {
    for (int r = 0; r < N; ++r) {
        state_t row_mask = 0;

        for (int c = 0; c < N; ++c) {
            row_mask |= 1 << rc_to_pos(r, c);
        }

        if (player == 0) {
            row_mask = row_mask << SHIFT;
        }

        if ((state & row_mask) == row_mask) {
            return true;
        }
    }

    for (int c = 0; c < N; ++c) {
        state_t col_mask = 0;

        for (int r = 0; r < N; ++r) {
            col_mask |= 1 << rc_to_pos(r, c);
        }

        if (player == 0) {
            col_mask = col_mask << SHIFT;
        }

        if ((state & col_mask) == col_mask) {
            return true;
        }
    }

    state_t diag_mask = 0;
    for (int i = 0; i < N; ++i) {
        diag_mask |= 1 << rc_to_pos(i, i);
    }

    if (player == 0) {
        diag_mask = diag_mask << SHIFT;
    }

    if ((state & diag_mask) == diag_mask) {
        return true;
    }

    state_t rev_diag_mask = 0;
    for (int i = 0; i < N; ++i) {
        rev_diag_mask |= 1 << rc_to_pos(i, N - 1 - i);
    }

    if (player == 0) {
        rev_diag_mask = rev_diag_mask << SHIFT;
    }

    if ((state & rev_diag_mask) == rev_diag_mask) {
        return true;
    }

    return false;
}



// Combinations precomputed

std::vector<std::vector<long>> combinations;

void create_combinations()
{
    int comb_size = N * N + 1;
    std::vector<std::vector<long>> comb(comb_size, std::vector<long>(comb_size));
    for (int i = 0; i < comb_size; i++)
    {
        comb.at(i).at(0) = 1;
        comb.at(i).at(i) = 1;
    }
    for (int i = 1; i < comb_size; i++)
    {
        for (int j = 1; j < i; j++)
        {
            comb.at(i).at(j) = comb.at(i - 1).at(j - 1) + comb.at(i - 1).at(j); // nCr = (n-1)C(r-1)+(n-1)Cr
        }
    }
    combinations = comb;
}

long get_combination(int x, int y)
{
    if (x < 0 || x > N * N || y < 0 || y > N * N)
    {
        return 0;
    }
    return combinations[x][y];
}

// Pop ord table precomputed

std::map<int, std::tuple<int, int>> pop_ord_table;

int compute_pop(int i)
{
    return __builtin_popcount(i);
}

void compute_pop_ord_table()
{
    std::map<int, int> pop_count;

    for (int i = 0; i < (1 << (N * N)); i++)
    {
        int pop = compute_pop(i);

        if (pop_count.find(pop) != pop_count.end())
            pop_count[pop]++;
        else
            pop_count[pop] = 1;

        int ord = pop_count[pop] - 1;

        pop_ord_table[i] = std::make_tuple(pop, ord);
    }
}

// Bijection
int get_state_class_id(state_t state)
{
    int mask = (1 << (N * N)) - 1;
    int o = state & mask;
    int x = ((state & (mask << SHIFT)) >> SHIFT) & mask;

    int shifted_o = o;

    int shift_count = 0;
    int shift_amount = 0;

    for (int pos = 0; pos < N * N; ++pos)
    {
        if ((x & (1 << pos)) == (1 << pos))
        {
            shift_count += 1;
        }
        else if (shift_count > 0)
        {
            int shift_mask = 0;
            for (int p = 0; p < pos - shift_count - shift_amount; ++p)
            {
                shift_mask |= (1 << p);
            }
            shift_mask = ~shift_mask & mask;

            shifted_o = ((shifted_o >> shift_count) & shift_mask) | (shifted_o & ~shift_mask);

            shift_amount += shift_count;
            shift_count = 0;
        }
    }

    int cx = std::get<0>(pop_ord_table[x]);
    int co = std::get<0>(pop_ord_table[o]);

    int sx = x;
    int so = shifted_o;

    int ord_sx = std::get<1>(pop_ord_table[sx]);
    int ord_so = std::get<1>(pop_ord_table[so]);
    
    return ord_sx * get_combination((N * N) - cx, co) + ord_so;
}

std::tuple<int, int> get_class(state_t state)
{
    int mask = (1 << (N * N)) - 1;
    int o = state & mask;
    int x = ((state & (mask << SHIFT)) >> SHIFT) & mask;

    int cx = std::get<0>(pop_ord_table[x]);
    int co = std::get<0>(pop_ord_table[o]);

    return std::make_tuple(cx, co);
}

long states_in_class(int cx, int co)
{
    return get_combination(N * N, cx) * get_combination((N * N) - cx, co);
}























// Class state iterator

class ComputeClassStates
{
public:
    static void dispositions(int pos, state_t sol_x, int mark_x[2], state_t sol_o, std::queue<int> &states_queue, std::condition_variable &cvar, std::mutex &mutex, bool &next, bool &stop)
    {
        if (pos >= N * N)
        {
            state_t sol = (sol_x << SHIFT) | sol_o;

            {
                std::unique_lock<std::mutex> lock(mutex);
                cvar.wait(lock, [&next, &stop]
                          { return next || stop; });

                if (stop)
                    return;

                states_queue.push(sol);
                next = false;
            }

            cvar.notify_one();

            return;
        }

        for (int i = 0; i < 2; ++i)
        {
            if (mark_x[i] > 0)
            {
                if (i == 0)
                {
                    if ((sol_o & (1 << pos)) != 0)
                    {
                        continue;
                    }
                    sol_x |= (1 << pos);
                }
                else
                {
                    sol_x &= ~(1 << pos);
                }

                mark_x[i]--;
                dispositions(pos + 1, sol_x, mark_x, sol_o, states_queue, cvar, mutex, next, stop);
                if (stop)
                {
                    return;
                }
                mark_x[i]++;
            }
        }
    }

    static void state_dispositions(int pos, int cx, int co, state_t sol_o, int mark_o[2], std::queue<int> &states_queue, std::condition_variable &cvar, std::mutex &mutex, bool &next, bool &stop)
    {
        if (pos >= N * N)
        {
            state_t sol_x = 0;
            int mark_x[2] = {cx, (N * N) - cx};
            dispositions(0, sol_x, mark_x, sol_o, states_queue, cvar, mutex, next, stop);
            return;
        }

        for (int i = 0; i < 2; ++i)
        {
            if (mark_o[i] > 0)
            {
                if (i == 0)
                {
                    sol_o |= (1 << pos);
                }
                else
                {
                    sol_o &= ~(1 << pos);
                }

                mark_o[i]--;
                state_dispositions(pos + 1, cx, co, sol_o, mark_o, states_queue, cvar, mutex, next, stop);
                if (stop)
                {
                    return;
                }
                mark_o[i]++;
            }
        }
    }

    static void compute_class_states(int cx, int co, std::queue<int> &states_queue, std::condition_variable &cvar, std::mutex &mutex, bool &next, bool &stop)
    {
        state_t sol_o = 0;
        int mark_o[2] = {co, (N * N) - co};
        state_dispositions(0, cx, co, sol_o, mark_o, states_queue, cvar, mutex, next, stop);
    }
};

class ClassStateIterator
{
public:
    ClassStateIterator(int cx, int co) : cx(cx), co(co)
    {
        stop = false;
        next = false;
        count = 0;
        limit_count = states_in_class(cx, co);

        class_states_thread = std::thread(&ComputeClassStates::compute_class_states, cx, co, std::ref(states_queue), std::ref(cvar), std::ref(mutex), std::ref(next), std::ref(stop));
    }

    ~ClassStateIterator()
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            stop = true;
        }
        cvar.notify_one();

        class_states_thread.join();
    }

    state_t get_next()
    {
        std::unique_lock<std::mutex> lock(mutex);

        next = true;
        cvar.notify_one();

        cvar.wait(lock, [this]
                  {  return !next; });

        state_t state = states_queue.front();
        states_queue.pop();

        count++;
        return state;
    }

    bool has_next()
    {
        return count < limit_count;
    }

private:
    int cx;
    int co;
    long count;
    long limit_count;
    std::queue<state_t> states_queue;
    std::thread class_states_thread;
    std::condition_variable cvar;
    std::mutex mutex;
    bool next;
    bool stop;
};















// Outcomes structure

class Outcomes
{
private:
    std::array<std::vector<bool>, MUTEX_NUMBER> outcomes;
    std::array<std::mutex, MUTEX_NUMBER> mutexes;

public:
    Outcomes();
    std::tuple<bool, bool> get_state_outcome(long id);
    void set_state_outcome(long id, std::tuple<bool, bool> value);
    void init_values(int cx, int co);
    bool is_draw(state_t state);
    bool is_win(state_t state);
    bool is_loss(state_t state);
    bool is_draw_lock(state_t state);
    bool is_win_lock(state_t state);
    bool is_loss_lock(state_t state);
    void update_draw_lock(state_t state);
    void update_win_lock(state_t state);
    void update_loss_lock(state_t state);
    void read(int cx, int co);
    void write(int cx, int co);
};

Outcomes::Outcomes()
{
    for (int i = 0; i < MUTEX_NUMBER; i++)
    {
        outcomes[i].resize(((MAX_STATES_NUMBER / MUTEX_NUMBER) + 1) * 2); // round up (+1)
    }
}

std::tuple<bool, bool> Outcomes::get_state_outcome(long id)
{
    bool bit0 = outcomes[id % MUTEX_NUMBER][(id / MUTEX_NUMBER) * 2];
    bool bit1 = outcomes[id % MUTEX_NUMBER][(id / MUTEX_NUMBER) * 2 + 1];
    return std::make_tuple(bit0, bit1);
}

void Outcomes::set_state_outcome(long id, std::tuple<bool, bool> value)
{
    outcomes[id % MUTEX_NUMBER][(id / MUTEX_NUMBER) * 2] = std::get<0>(value);
    outcomes[id % MUTEX_NUMBER][(id / MUTEX_NUMBER) * 2 + 1] = std::get<1>(value);
}

void Outcomes::init_values(int cx, int co)
{
    for(int id=0; id<states_in_class(cx, co); id++)
    {
        set_state_outcome(id, std::make_tuple(0, 0));
    }
}

bool Outcomes::is_draw(state_t state)
{
    long id = get_state_class_id(state);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 0) && (std::get<1>(val) == 0);
}

bool Outcomes::is_win(state_t state)
{
    long id = get_state_class_id(state);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 0) && (std::get<1>(val) == 1);
}

bool Outcomes::is_loss(state_t state)
{
    long id = get_state_class_id(state);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 1) && (std::get<1>(val) == 1);
}

bool Outcomes::is_draw_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 0) && (std::get<1>(val) == 0);
}

bool Outcomes::is_win_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 0) && (std::get<1>(val) == 1);
}

bool Outcomes::is_loss_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    std::tuple val = get_state_outcome(id);
    return (std::get<0>(val) == 1) && (std::get<1>(val) == 1);
}

void Outcomes::update_draw_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    set_state_outcome(id, std::make_tuple(false, false));
}

void Outcomes::update_win_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    set_state_outcome(id, std::make_tuple(false, true));
}

void Outcomes::update_loss_lock(state_t state)
{
    long id = get_state_class_id(state);
    std::lock_guard<std::mutex> lock(mutexes[id % MUTEX_NUMBER]);
    set_state_outcome(id, std::make_tuple(true, true));
}

void Outcomes::write(int cx, int co)
{
    std::string filename = "./outcomes_" + std::to_string(N) + "x" + std::to_string(N) + "/c_" + std::to_string(cx) + "_" + std::to_string(co) + ".bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        char buffer = 0;
        int bit_count = 0;
        for (int id = 0; id < states_in_class(cx, co); ++id) {
            auto val = get_state_outcome(id);
            int bit0 = std::get<0>(val);
            int bit1 = std::get<1>(val);
            
            buffer |= (bit0 << (6 - bit_count + 1));
            buffer |= (bit1 << (6 - bit_count));
            
            bit_count += 2;
            if (bit_count >= 8) {
                file.write(static_cast<const char*>(&buffer), sizeof(buffer));
                buffer = 0;
                bit_count = 0;
            }
        }
        if (bit_count != 0) {
            file.write(static_cast<const char*>(&buffer), sizeof(buffer));
        }
        file.close();
    }
}

void Outcomes::read(int cx, int co)
{
    std::string filename = "./outcomes_" + std::to_string(N) + "x" + std::to_string(N) + "/c_" + std::to_string(cx) + "_" + std::to_string(co) + ".bin";

    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        char byte;
        int id = 0;
        int total_states = states_in_class(cx, co);

        while (file.read(&byte, sizeof(byte)) && id < total_states) {
            int bit_count = 0;
            while (bit_count < 8 && id < total_states) {
                bool bit0 = (byte & (1 << (6 - bit_count + 1))) == (1 << (6 - bit_count + 1));
                bool bit1 = (byte & (1 << (6 - bit_count))) == (1 << (6 - bit_count));
                set_state_outcome(id, std::make_tuple(bit0, bit1));
                id++;
                bit_count += 2;
            }

            if (id >= total_states) {
                break;
            }
        }
        file.close();
    }
}



Outcomes outcomes;
Outcomes outcomes_rev;
Outcomes outcomes_next;
Outcomes outcomes_next_rev;


bool update;
std::mutex update_mutex;











void compute_terminal_states_outcomes_t(int i_start, int i_end, int cx, int co, bool rev)
{
    int i = 0;

    if(!rev)
    {
        ClassStateIterator iterator = ClassStateIterator(cx, co);
        while (iterator.has_next())
        {
            state_t state = iterator.get_next();

            if(i>=i_end)
                break;

            if(i>=i_start)
            {
                if (check_winner(state, 0))
                    outcomes.update_win_lock(state);
                else if (check_winner(state, 1))
                    outcomes.update_loss_lock(state);
            }
            i++;
        }
    }
    else
    {
        ClassStateIterator iterator = ClassStateIterator(co, cx);
        while (iterator.has_next())
        {
            state_t state = iterator.get_next();

            if(i>=i_end)
                break;

            if(i>=i_start)
            {
                if (check_winner(state, 0))
                    outcomes_rev.update_win_lock(state);
                else if (check_winner(state, 1))
                    outcomes_rev.update_loss_lock(state);
            }
            i++;
        }
    }
}





void compute_terminal_states_outcomes(int cx, int co, bool rev)
{
    std::vector<std::thread> threads;
    int states_number = states_in_class(cx, co);
    int states_per_thread = states_number / THREADS_NUMBER;

    for(int i=0; i<THREADS_NUMBER-1; i++)
        threads.push_back(std::thread([i, cx, co, states_per_thread, rev]{compute_terminal_states_outcomes_t(i*states_per_thread, (i+1)*states_per_thread, cx, co, rev);}));

    threads.push_back(std::thread([cx, co, states_number, states_per_thread, rev]{compute_terminal_states_outcomes_t((THREADS_NUMBER-1)*states_per_thread, states_number, cx, co, rev);}));

    for(std::thread &t : threads)
        t.join();           
}













void compute_from_next_states_outcomes_t(int i_start, int i_end, int cx, int co, bool rev)
{
    int i = 0;

    if(!rev)
    {
        ClassStateIterator iterator = ClassStateIterator(cx, co);
        while (iterator.has_next())
        {
            state_t state = iterator.get_next();

            if(i>=i_end)
                break;

            if(i>=i_start)
            {
                if(outcomes.is_draw_lock(state))
                {
                    bool loss = true;
                    bool win = false;

                    for(auto m : get_available_moves(state, 0))
                    {
                        state_t child_state = swap(move(state, m, 0));
                        std::tuple<int, int> child_state_class = get_class(child_state);
                        int ccx = std::get<0>(child_state_class);
                        int cco = std::get<1>(child_state_class);

                        if(cx != co)
                        {
                            if(ccx==co && cco==cx)
                            {
                                if(!outcomes_rev.is_win_lock(child_state))
                                    loss = false;
                                if(outcomes_rev.is_loss_lock(child_state))
                                {
                                    win = true;
                                    break;
                                }
                            }
                            else
                            {
                                if(!outcomes_next_rev.is_win_lock(child_state))
                                    loss = false;
                                if(outcomes_next_rev.is_loss_lock(child_state))
                                {
                                    win = true;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            if(ccx==co && cco==cx)
                            {
                                if(!outcomes.is_win_lock(child_state))
                                    loss = false;
                                if(outcomes.is_loss_lock(child_state))
                                {
                                    win = true;
                                    break;
                                }
                            }
                            else
                            {
                                if(!outcomes_next_rev.is_win_lock(child_state))
                                    loss = false;
                                if(outcomes_next_rev.is_loss_lock(child_state))
                                {
                                    win = true;
                                    break;
                                }
                            }
                        }

                    }

                    if(win)
                    {
                        outcomes.update_win_lock(state);
                        std::lock_guard<std::mutex> lock(update_mutex);
                        update = true;
                    }
                    else if(loss)
                    {
                        outcomes.update_loss_lock(state);
                        std::lock_guard<std::mutex> lock(update_mutex);
                        update = true;
                    }
                }
            }
            i++;
        }
    }
    else
    {
        ClassStateIterator iterator = ClassStateIterator(co, cx);
        while (iterator.has_next())
        {
            state_t state = iterator.get_next();
            
            if(i>=i_end)
                break;

            if(i>=i_start)
            {
                if(outcomes_rev.is_draw_lock(state))
                {
                    bool loss = true;
                    bool win = false;

                    for(auto m : get_available_moves(state, 0))
                    {
                        state_t child_state = swap(move(state, m, 0));
                        std::tuple<int, int> child_state_class = get_class(child_state);
                        int ccx = std::get<0>(child_state_class);
                        int cco = std::get<1>(child_state_class);

                        if(ccx==cx && cco==co)
                        {
                            if(!outcomes.is_win_lock(child_state))
                                loss = false;
                            if(outcomes.is_loss_lock(child_state))
                            {
                                win = true;
                                break;
                            }
                        }
                        else
                        {
                            if(!outcomes_next.is_win_lock(child_state))
                                loss = false;
                            if(outcomes_next.is_loss_lock(child_state))
                            {
                                win = true;
                                break;
                            }
                        }
                    }

                    if(win)
                    {
                        outcomes_rev.update_win_lock(state);
                        std::lock_guard<std::mutex> lock(update_mutex);
                        update = true;
                    }
                    else if(loss)
                    {
                        outcomes_rev.update_loss_lock(state);
                        std::lock_guard<std::mutex> lock(update_mutex);
                        update = true;
                    }
                }
            }
            i++;
        }
    }
}


void compute_from_next_states_outcomes(int cx, int co, bool rev)
{
    std::vector<std::thread> threads;
    int states_number = states_in_class(cx, co);
    int states_per_thread = states_number / THREADS_NUMBER;

    for(int i=0; i<THREADS_NUMBER-1; i++)
        threads.push_back(std::thread([i, cx, co, states_per_thread, rev]{compute_from_next_states_outcomes_t(i*states_per_thread, (i+1)*states_per_thread, cx, co, rev);}));

    threads.push_back(std::thread([cx, co, states_number, states_per_thread, rev]{compute_from_next_states_outcomes_t((THREADS_NUMBER-1)*states_per_thread, states_number, cx, co, rev);}));

    for(std::thread &t : threads)
        t.join();        
}




void compute_all_states_outcome()
{

    for(int n=N*N; n>=0; n--)
    {
        std::cout << "Computing class level " << n << std::endl;
        
        for(int cx=0; cx<=(n/2); cx++)
        {
            int co = n - cx;
            std::cout << "Computing class (" << cx << ", " << co << ") and (" << co << ", " << cx << ")" << std::endl;

            outcomes.init_values(cx, co);
            if(cx != co)
                outcomes_rev.init_values(co, cx);
            
            if(n < N * N)
            {
                outcomes_next_rev.read(co, cx+1);
                if(cx != co)
                    outcomes_next.read(cx, co+1);
            }

            compute_terminal_states_outcomes(cx, co, false);
            if(cx!=co)
                compute_terminal_states_outcomes(cx, co, true);


            update = true;
            while(update)
            {
                update = false;
                compute_from_next_states_outcomes(cx, co, false);
                if(cx != co)
                    compute_from_next_states_outcomes(cx, co, true);
            }

            outcomes.write(cx, co);
            if(cx!=co)
                outcomes_rev.write(co, cx);
            


        }
    }
}






int main()
{
    auto start_time = std::chrono::high_resolution_clock::now();
    create_combinations();
    compute_pop_ord_table();
    compute_all_states_outcome();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Computation Time: " << duration.count() / 1000. << " s" << std::endl;

    return 0;
}



// (No threading) Computation Time: 1590.79 s 