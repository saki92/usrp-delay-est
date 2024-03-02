PSS_LENGTH = 127
def get_x_sequence():

    x = [0,1,1,0,1,1,1]

    for i in range(PSS_LENGTH):
        x.append((x[i + 4] + x[i]) % 2)

    return x

def get_d_sequence(Nid2: int) -> list[int]:
    x = get_x_sequence()
    d = []
    for n in range(PSS_LENGTH):
        m = (n + 43 * Nid2) % PSS_LENGTH
        d.append(1 - 2 * x[m])

    return d
