def clean_character(character, movie):
    ch_len = len(character)
    with open(movie + '.txt') as file:
        with open(movie + '_' + character + '_clean.txt', 'a') as clean_file:
            for line in file:
                if line[:ch_len] == character:
                    while '[' in line:
                        s, e = line.find('['), line.find(']')
                        line = line[:s] + line[e+2:]
                    clean_file.write(line[ch_len+2:])


def clean_movie(movie):
    with open(movie + '.txt') as file:
        with open(movie +  '_clean.txt', 'a') as clean_file:
            for line in file:
                if line[0] != '\n':
                    clean_file.write(line)




def merge(files, names, character):
    with open('final_data.txt', 'w') as outfile:
        for fname in files:
            with open(fname+'_clean.txt') as infile:
                for line in infile:
                    for name in names:
                        if line[:len(name)] == name:
                            line = character + ': ' +line[len(name)+1:]
                    outfile.write(line)

#clean_character('Tony Stark', 'ironman1')
#clean_movie('spiderman')

f = ['ironman1', 'ironman2', 'avengers', 'ironman3', 'ultron', 'civilwar', 'spiderman', 'infwar', 'endgame']
n = ['Tony:', 'TONY:', 'TONY STARK:']
character = 'Tony Stark'
merge(f, n, character)