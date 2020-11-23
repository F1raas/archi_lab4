# -*- coding: utf-8 -*-
#
# Copyright (c) 2011-2013, Julien-Charles Lévesque <levesque.jc@gmail.com>
#  and contributors.
#
# Distributed under the terms of the MIT license. See the COPYING file at
#  the top-level directory of this project and at
#  https://bitbucket.org/ulaval-gif-3000/mipssim/raw/tip/COPYING

'''
Effectue une comparaison entre deux traces et trouve les différences pour chaque cycle d'horloge.
'''

import argparse
import difflib
from collections import OrderedDict as OD


def find_table_type(lines, i):
    '''Remonte ligne par ligne à partir de `i` pour trouver le type du tableau.'''

    #On regarde le caractère de début de ligne.
    while lines[i][0] in ['|', '+']:
        #La fin de l'entête
        if lines[i][0] == '+' and lines[i-1][0] != '|':
            header_line = lines[i-1]
            header_i = i - 1
            break
        i -= 1

    if header_line.find('Stations') != -1:
        table_type = 'RS'
    elif header_line.find('ROB') != -1:
        table_type = 'ROB'
    elif header_line.find('Registres') != -1:
        table_type = 'Registres'

    return table_type, header_line.strip(':\n')


def find_next_cycle(lines):
    '''Générateur pour parcourir un fichier de trace par coup d'horloge. Retourne l'indice de chq
     ligne débutant par "Cycle:" '''
    i = 0
    while i < len(lines):
        if lines[i].find('Cycle:') != -1:
            yield i
        i += 1


def get_table(lines, i):
    '''Extrait une table débutant à la ligne i et retourne un pointeur vers la ligne à la fin
     de la dite table. La table est stoquée dans un ordered dict avec une entrée par ligne.'''
    headers = [h.strip() for h in lines[i + 1][1:-2].split('|')]

    table = OD()
    table['headers'] = headers
    i = i + 3
    while lines[i][0] != '+':
        elements = [e.strip() for e in lines[i][1:-2].split('|')]
        table[elements[0]] = elements[1:]
        i += 1
    return table, i + 1


def parse_trace(lines):
    '''
    Reconstruit des tableaux à partir d'une trace textuelle.
    '''
    cycles = []
    for i in find_next_cycle(lines):
        cycle = OD()
        cycle['i'] = int(lines[i].split(':')[1].strip('\n'))
        cycle['pc'] = int(lines[i+1].split(':')[1].strip('\n'))
        cycle['RS'], next_i = get_table(lines, i+3)
        cycle['regs'], next_i = get_table(lines, next_i + 1)
        cycle['ROB'], next_i = get_table(lines, next_i + 1)
        cycles.append(cycle)
    return cycles


def compare_dicts(d1, d2):
    '''
    Compare deux cycles et retourne un bool indiquant si ils sont pareils et une liste des
     différences entre les deux cycles.
    '''
    same = True
    diffs = []
    for (key1, val1), (key2, val2) in zip(d1.items(), d2.items()):
        if key1 != key2:
            same = False
            #Peut se produire dans le cas d'entrées différentes se trouvant dans le ROB.
            #import IPython; IPython.embed()

        if isinstance(val1, OD) or isinstance(val1, dict):
            sub_same, sub_diffs = compare_dicts(val1, val2)
            if sub_same != True:
                same = False
                #dans le cas des registres on peut faire un meilleur display (peut-être devrais-je
                #simplement reconstruire une table complète pour les registres?)
                if key1 == 'regs':
                    sub_diffs = [[sd[0] + sd[1], sd[-1]] for sd in sub_diffs]
                diffs.append([key1] + sub_diffs)
        elif isinstance(val1, list):
            i = 0
            assert(len(val1) == len(val2))
            for j, (v1, v2) in enumerate(zip(val1, val2)):
                if v1 != v2:
                    i = j
                    same = False
                    diffs.append([key1, d1['headers'][i + 1], (v1, v2)])
        else:
            if val1 != val2:
                same = False
                diffs.append([key1, (val1, val2)])
    return same, diffs


def main(file_1, file_2):

    f1 = open(args.file_1, 'rt')
    f2 = open(args.file_2, 'rt')

    f1_lines = f1.readlines()
    f2_lines = f2.readlines()

    simulation1 = parse_trace(f1_lines)
    simulation2 = parse_trace(f2_lines)

    all_diffs = []
    for i, (cycle1, cycle2) in enumerate(zip(simulation1, simulation2)):
        same, diffs = compare_dicts(cycle1, cycle2)
        strdiffs = ['\n  '.join([str(sd) for sd in d]) for d in diffs]
        if not same:
            print("Différences au coup d'horloge %i :\n %s" % (i, '\n '.join(strdiffs)))
            all_diffs.append([i, diffs])

    return simulation1, simulation2, all_diffs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Effectue une comparaison entre deux traces textuelles et trouve les différences pour chaque cycle d'horloge.")

    parser.add_argument('file_1', help='Premier fichier.')
    parser.add_argument('file_2', help='Second fichier.')

    args = parser.parse_args()

    sim1, sim2, diffs = main(args.file_1, args.file_2)
