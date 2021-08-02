import os

if __name__ == '__main__':
    dialects = [i[:-5] for i in os.listdir('aggregated_city/lm/char')]
    print(dialects, len(dialects))
