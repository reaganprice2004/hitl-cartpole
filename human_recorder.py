"""
human_recorder.py - Teleoperate CartPole (or other gym env) with keyboard input and record transitions
into a CSV file 'human_corrections.csv'. Run this in a separate terminal while the agent is training
or running to collect human demonstrations/corrections.
"""

# import libraries
import gymnasium as gym            # OpenAI Gymnasium for environments (install with `pip install gymnasium`)
import csv                         # to write CSV file
import argparse                    # for CLI arguments
import time                        # for small sleeps
import sys                         # for reading keyboard input (cross-platform will differ)
import termios, tty, os            # for simple Unix keyboard capture

def getch():
    # read a single character from stdin (works on Unix-like terminals)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)   # save terminal settings
    try:
        tty.setraw(fd)                     # set raw mode to capture keys instantly
        ch = sys.stdin.read(1)              # read one character
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)  # restore settings
    return ch

def run(env_name="CartPole-v1", out_file="human_corrections.csv", render=True):
    # create environment
    env = gym.make(env_name)               # instantiate environment
    # open CSV file for appending human transitions
    f = open(out_file, "a", newline='')    # append mode so multiple sessions accumulate
    writer = csv.writer(f)                 # CSV writer
    # write header if file empty
    if os.stat(out_file).st_size == 0:
        writer.writerow(["state", "action", "next_state", "reward", "done"])
    # start episodes loop
    try:
        while True:
            state, info = env.reset()      # reset environment and get initial state
            done = False                   # episode done flag
            while not done:
                if render:
                    env.render()           # render to screen (requires display)
                # read a key from user (blocking until key pressed)
                print("Press 'a' for left, 'd' for right, or 'q' to quit and save. Waiting for input...")
                key = getch()              # read single char
                if key == 'q':            # quit command
                    f.close()             # close file cleanly
                    env.close()           # close env
                    print("Exiting human recorder.")
                    return
                # map key to action for CartPole: 0=left,1=right
                if key == 'a':
                    action = 0
                elif key == 'd':
                    action = 1
                else:
                    # ignore other keys and continue loop
                    print("Unrecognized key. Use 'a' or 'd' (or 'q').")
                    continue
                # step environment with human action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # write transition to CSV (store state vectors as JSON-like strings)
                writer.writerow([state.tolist(), action, next_state.tolist(), float(reward), bool(done)])
                f.flush()                 # flush to disk so trainer can read it concurrently
                # advance state for next iteration
                state = next_state
                # tiny sleep to avoid spamming
                time.sleep(0.01)
    except KeyboardInterrupt:
        # handle Ctrl-C gracefully: close file and environment
        f.close()
        env.close()
        print("Interrupted; human recorder closed.")

if __name__ == "__main__":
    # CLI: allow specifying env and output file if desired
    parser = argparse.ArgumentParser(description="Human teleoperation recorder for Gym environments.")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment id (default CartPole-v1)")
    parser.add_argument("--out", type=str, default="human_corrections.csv", help="CSV file to append corrections")
    parser.add_argument("--norender", action="store_true", help="Disable rendering (useful in headless)")
    args = parser.parse_args()
    run(env_name=args.env, out_file=args.out, render=not args.norender)
