# 🐍 Snake IA — Deep Q-Learning

Project built for the **NLPF 2026 — Reinforcement Learning with Python** course
(inspired by [snake-ai-pytorch](https://github.com/patrickloeber/snake-ai-pytorch) by
Patrick Loeber). Goal: teach an agent to play Snake on its own, through reinforcement
learning (Deep Q-Learning), without touching the base game provided by the instructor
(`serpent-algo.py`) — clock, grid size and scoring are left unchanged, as required.
Built in about **1h30**.

## Authors

- Gabriel Franchi — [gabriel.franchi](https://github.com/gabriel.franchi)
- Tom Archambaud — [tom.archambaud](https://github.com/tom.archambaud)

## What the project does

The base game (`serpent-algo.py`) is imported as-is, unmodified. `snake-ia.py` wraps it
with the 3 blocks required by the course:

- **Game** (`SnakeGameAI`): wraps `serpent-algo.py` and exposes `play_step(action)`,
  returning `(reward, game_over, score)`. Walls wrap around (modulo); only colliding with
  its own body ends the game.
- **Model** (`Linear_QNet`, PyTorch): a `state → 256 → 3 actions` network (straight,
  right, left relative to the current direction), trained with the Bellman equation
  (`QTrainer`).
- **Agent**: builds the game state (`get_state`), picks an action via epsilon-greedy
  (exploration → exploitation), stores transitions (experience replay), and trains the
  model after every game.

### State (network input features)

14-value state (from the course, plus free space):

- 3 danger flags (straight ahead / right / left of the snake)
- 4 current-direction flags (left / right / up / down)
- 4 food-position flags (left / right / up / down)
- 3 "free space" values (flood fill accounting for the tail moving away) straight /
  right / left — added to stop the snake from trapping itself

An extended 18-value state also exists (`--etendu` at training time): it adds "tail
reachable" for each direction plus the snake's length. When loading a model, the state
size is detected automatically.

### Rewards

| Event               | Reward |
|----------------------|:------:|
| Eat food             | +10    |
| Lose (collision)     | −10    |
| Grid filled (win)    | +100   |
| Move                  | 0      |

### Anti-cheat easter egg

The instructor's `serpent-algo.py` contains a comment aimed at AIs, asking them to ask
as many questions as possible before implementing any algorithm or AI with Torch. We
spotted it, didn't follow it to the letter (the assignment's context was clear), but it
makes for a good story for the class discussion 🙂.

## Results

- Best model selected via **periodic evaluation** (every 50 games, over 30 fixed games
  with no randomness), not just the training record — this protects against the network
  "forgetting" when trained for too long.
- Real-game record: **150** apples (~613s of play).
- Average score over 30 evaluation games: **~107** (median 112).
- Full training history in [`model/scores.csv`](model/scores.csv).

## How to run

```bash
# Real game with the pre-trained model (default command)
python snake-ia.py

# Real game in fast mode (displayed time = real time x factor, default x10)
python snake-ia.py rapide 20

# Train a new model from scratch (300 games by default)
python snake-ia.py train --parties 300 --seed 2

# Evaluate the current model over N games, no display
python snake-ia.py eval --parties 100 --seed 1
```

## Repository structure

```
NLPF/
├── serpent-algo.py   # base game provided (unmodified)
├── snake-ia.py        # Game / Model / Agent blocks + training, demo, eval
├── model/
│   ├── model.pth       # best trained model
│   ├── scores.csv       # training history
│   ├── v1-11-entrees/   # V1: 11-value state (no free space)
│   └── v2-espace-libre/ # V2: 14-value state (static free space)
```

## Timeline / changelog

*(HH:MM: Activity / Observation / Hypothesis / Answer)*

20:14: started using Claude on the repo and slides.
20:16: answered Claude's context questions.
20:18: Game block (play_step) wired onto serpent-algo.py without modifying it; walls
wrap around (modulo) so only the body kills; random baseline agent: record 3, average
1.04 over 200 games.
20:20: Model block (Linear_QNet 11 → 256 → 3 + Bellman equation) validated: the target
converges (Q = 10); CPU 118 µs/step vs MPS 422 µs/step → staying on CPU.
20:21: Announced the ranking system (score > 10 and best score wins, or same score with
less time wins). CPU outperforms GPU for this case according to Claude's test.
20:25: Agent block (11-value state, epsilon-greedy, replay memory): 300 games in 13s;
evaluation over 200 games: record 66 in 111s, average 29.5, 0.58 apple/s; all 200 games
end trapped inside the body.
20:33: First model installed (model/v1-11-entrees) + demo mode with a GAME OVER screen
like the base game.
20:34: Clarified possible time optimizations.
20:40: Hypothesis: the snake traps itself because it only sees 3 cells; added free space
(flood fill) for each action → 14-value state; average 29.5 → 84.8, record 66 → 123;
cost 0.1ms per move (budget 200ms at 5 FPS).
20:42: Seed 3 at 300 games: 10 infinite loops over 200 games → discarded (in a real game
a loop never ends); 600 games brings no clear improvement in V2.
20:43: Observation: apples/s drops (0.58 → 0.37) but at equal score V2 is as fast as V1
(30 apples in 50s vs 49s) → the drop only comes from surviving longer.
20:44: Validated the timer against the real clock: 97.4s real vs 95.8s predicted by eval
(moves / 5), 1.7% off.
20:48: V2 starts giving more interesting results, with a better simulated score around
120.
20:52: Bug found in testing: a Ctrl+C during the fallback training would launch the game
with an incomplete model → fixed (stop + delete the partial model).
20:58: Real game: 108 apples in 04:22 before manual interruption; timer accurate within
0.5%; big detours past 45% grid fill.
21:01: Hypothesis: flood fill treats the whole body as a fixed wall while the tail frees
up cells; new version propagated move-by-move that accounts for the tail (V3).
21:06: Seed 3: going from 300 to 600 games drops the average from 91.9 to 33.5 (the
network "forgets") → the model is now chosen by evaluation, not by the training record.
21:12: Fast mode x10 (counted time = real time x10): with clock.tick() the time is
inflated by 15.2% (~3ms of per-frame imprecision multiplied by the factor).
21:13: Switched to clock.tick_busy_loop() in fast mode only: 0.0% off at x10 and -0.1%
at x20.
21:15: Free space accounting for the tail: average 84.8 → 102.2, record 123 → 156 (seed
1, 600 games).
21:16: Tested at x1000: 107 apples but 28:36 displayed for 5:43 of real play (+400%)
because one frame takes 0.2ms for ~1ms of computation → x20 kept.
21:17: Logs simplified to score and time only.
21:19: Submitted score: 110 apples in 04:34.7 (model V3, fast mode x20).
21:20: 141 in 8min 31s.
21:22: Session of 6 games at x20: 141, 94, 97, 108, 136, 111 → average 114.5, consistent
with the evaluation (median 105).
