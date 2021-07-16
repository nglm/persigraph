import numpy as np
import pandas as pd

def clear_double_space(filename):
    # Read in the file
    with open(filename, 'r') as file :
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('  ', ' ')

    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

def extract_from_files(
    filename,

):
    """

    """
    # Find files
    names = [x for x in 'abcdefghi'[:7]]
    df = pd.read_csv(filename, sep=' ', names=names, header=1)
    # Remove ensemble mean rows
    df = df[df['b'] != 'em']
    # Remove useless columns
    df = df[['a', 'c', 'd', 'e']]
    print(df)
    # Time steps
    time = list(set([h for h in df['a']]))
    time.sort()
    time = np.array(time)
    print(time)
    # Re-index time step in the df
    df[['a']] = df[['a']]//24-1
    print(df)
    # Reshape to (N, 2, T) array
    N = int(df[['c']].max())+1
    print(N)
    T = int(df[['a']].max())+1
    print(T)
    members = np.zeros((N, 2, T))
    # Find the right values
    for i in range(N):
        for t in range(T):
            values = df[(df.a == t) & (df.c == i)]
            members[i, 0, t] = float(values.d)
            members[i, 1, t] = float(values.e)
    return members, time

def preprocessing_mjo(filename):
    clear_double_space(filename)
    return extract_from_files(filename)
