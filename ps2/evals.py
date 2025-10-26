from game import Game

FILES = {
    "kuhn": "efgs/kuhn.txt",
    "leduc2": "efgs/leduc2.txt",
    "rock_paper_superscissors": "efgs/rock_paper_superscissors.txt",
}   

if __name__ == "__main__":
    for name, filename in FILES.items():
        g = Game(filename)
        x,y = g.uniform()
        # best response of P1 to uniform P2
        v1, s1_br = g.best_response(1, y)

        # nash gap if both are uniform
        nash_gap = g.nash_gap(x,y)
        
        print(f"{name}: v1 = {v1}, nash_gap = {nash_gap}")