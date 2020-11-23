# -*- coding: utf-8 -*-
#
# Copyright (c) 2011-2014, Julien-Charles Lévesque <levesque.jc@gmail.com>
#  and contributors.
#
# Distributed under the terms of the MIT license. See the COPYING file at
#  the top-level directory of this project and at
#  https://bitbucket.org/ulaval-gif-3000/mipssim/raw/tip/COPYING

import os
from string import Template
from output.prettytable import PrettyTable

noneify = lambda x: x if x != None else ''
rob_states = ['Unused', 'Issue', 'Execute', 'Writeback', 'Commit']
rob_variables = ['i', 'instr.code', 'instr.operands', 'state', 'dest', 'value']
#rob_labels = ['Entrée', 'Instruction', '', 'État', 'Dest.', 'Valeur']
rob_labels = ['Entry', 'Instruction', '', 'State', 'Dest.', 'Value']

rs_variables = ['name', 'instr.code', 'time', 'vj', 'vk', 'qj', 'qk', 'dest', 'A']
#rs_labels = ['Station', 'Op', 'Temps', 'Vj', 'Vk', 'Qj', 'Qk', 'Dest', 'A']
rs_labels = ['Station', 'Op', 'Time', 'Vj', 'Vk', 'Qj', 'Qk', 'Dest', 'A']

def get_vars_and_subvars(var_names, extractee   ):
    row = []
    for v in var_names:
        if '.' in v:
            temp = extractee.__getattribute__(v.split('.')[0])
            try:
                elem = temp.__getattribute__(v.split('.')[1])
            except:
                elem = None
        else:
            elem = extractee.__getattribute__(v)
        elem = str(noneify(elem))
        row.append(elem)
    return row

def rob_fix_row(row):
    #Entry
    row[0] = str(int(row[0]) + 1)
    #État d'exécution
    row[3] = rob_states[int(row[3])]
    return row

def rs_fix_row(row):
    if row[5] != '':
        row[5] = '#' + str(int(row[5]) + 1)
    if row[6] != '':
        row[6] = '#' + str(int(row[6]) + 1)
    if row[7] != '':
        row[7] = '#' + str(int(row[7]) + 1)
    return row

def reg_fix_statrow(row):
    for i in range(1, len(row)):
        if row[i] != '' and row[i] != 'X':
            row[i] = '#' + str(int(row[i]) + 1)
    return row

class TextTrace:
    '''
    Laisse une trace dans un fichier texte.
    '''
    def __init__(self, trace_file):
        self.trace_f = open(trace_file, 'w')

    def __del__(self):
        self.trace_f.close()

    def update(self, simulator):
        '''
        Écrit l'état du ROB, des stations de reservation (des unités fonctionnelles) et de la
         mémoire le fichier `self.trace_f` à chaque itération.
        '''
        #Tampon de réordonnancement


        rob_table = PrettyTable(rob_labels)
        for rob_entry in simulator.ROB:
            row = get_vars_and_subvars(rob_variables, rob_entry)
            #quelques manipulations pour obtenir un résultat plus clair.
            row = rob_fix_row(row)
            rob_table.add_row(row)

        #Table des stations de réservation
        rs_table = PrettyTable(rs_labels)
        for station_type, funits in simulator.RS.items():
            for i, funit in enumerate(funits):
                row = get_vars_and_subvars(rs_variables, funit)
                row = rs_fix_row(row)
                rs_table.add_row(row)

        # Table des registres
        num_regs = int(len(simulator.regs) / 2)
        reg_table = PrettyTable([' '] + [str(a) for a in range(10)])
        for reg_type in ['R', 'F']:
            for row_start in range(0, num_regs, 10):
                remainder = num_regs - row_start
                if remainder < 10:
                    padding = ['X'] * (10 - remainder)
                    row_end = row_start + remainder
                else:
                    padding = []
                    row_end = row_start + 10
                reg_stat_row = ['ROB#'] + [noneify(simulator.regs.stat[reg_type + str(a)])
                    for a in range(row_start, row_end)] + padding
                reg_stat_row = reg_fix_statrow(reg_stat_row)
                reg_table.add_row(reg_stat_row)

                reg_value_row = [reg_type + str(row_start)] +\
                    [simulator.regs[reg_type + str(a)] for a in range(row_start, row_end)] +\
                    padding
                reg_table.add_row(reg_value_row)

        # Affichage des tableaux précédemment créés
        self.trace_f.write('%s\n' % ('=' * 80))
        self.trace_f.write('Cycle: %d\n' % simulator.clock)
        self.trace_f.write('Program Counter : %d\n' % simulator.PC)
        #self.trace_f.write('Stations de réservation:\n%s\n' % str(rs_table))
        self.trace_f.write('Reservation stations:\n%s\n' % str(rs_table))
        self.trace_f.write('ROB: \n%s\n' % str(rob_table))
        #self.trace_f.write('Registres: \n%s\n' % (reg_table))
        self.trace_f.write('Registers: \n%s\n' % (reg_table))

        self.trace_f.flush()


class LaTeXTable:
    def __init__(self, caption, tab_label, first_row, align=[]):
        self.num_rows = len(first_row)

        try:
            if len(align) == self.num_rows:
                ali_str = ''.join(align)
            else:
                ali_str = ''.join(['c'] * self.num_rows)
        except:
            ali_str = ''.join([align] * self.num_rows)

        self.string = Template(r'''
\begin{center}
$caption


%$label
    \begin{tabular}{$align} \toprule
$firstrow \\ \midrule
''').substitute(caption=caption, label=tab_label, align=ali_str, firstrow='&'.join(first_row))

    def add_row(self, row):
        row_str = '&'.join(row) + '\\\\ \n'
        row_str = row_str.replace('#', r'\#')
        self.string += row_str

    def get_table(self):
        #fermer la table
        self.string += r'''
    \bottomrule
    \end{tabular}
\end{center}
'''
        return self.string


class LaTeXTrace:

    '''
    Laisse une trace dans un fichier texte.
    '''
    def __init__(self, trace_file):
        self.trace_f = open(trace_file, 'w')
        self.trace_f.write(r'''\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[french]{babel}
\usepackage{booktabs}
\begin{document}
''')

    def __del__(self):
        self.trace_f.write(r"\end{document}")
        self.trace_f.close()

    def update(self, simulator):
        '''
        Écrit l'état du ROB, des stations de reservation (des unités fonctionnelles) et de la
         mémoire le fichier `self.trace_f` à chaque itération.
        '''
        #Tampon de réordonnancement
        rob_table = LaTeXTable('Tampon de réordonnancement','cycle%i_rob' % simulator.clock, rob_labels)
        for rob_entry in simulator.ROB:
            row = get_vars_and_subvars(rob_variables, rob_entry)
            row = rob_fix_row(row)
            rob_table.add_row(row)

        #Table des stations de réservation
        rs_table = LaTeXTable('Stations de réservation', 'cycle%i_rs' % simulator.clock, rs_labels)
        for station_type, funits in simulator.RS.items():
            for i, funit in enumerate(funits):
                row = get_vars_and_subvars(rs_variables, funit)
                row = rs_fix_row(row)
                rs_table.add_row(row)

        # Table des registres
        num_regs = int(len(simulator.regs) / 2)
        reg_table = LaTeXTable('Registres', 'cycle%i_regs' % simulator.clock,
            [' '] + [str(a) for a in range(10)])
        for reg_type in ['R', 'F']:
            for row_start in range(0, num_regs, 10):
                remainder = num_regs - row_start
                if remainder < 10:
                    padding = ['X'] * (10 - remainder)
                    row_end = row_start + remainder
                else:
                    padding = []
                    row_end = row_start + 10
                reg_stat_row = ['ROB#'] + [noneify(simulator.regs.stat[reg_type + str(a)])
                    for a in range(row_start, row_end)] + padding
                reg_stat_row = reg_fix_statrow(reg_stat_row)
                reg_table.add_row(reg_stat_row)

                reg_value_row = [reg_type + str(row_start)] +\
                    [str(simulator.regs[reg_type + str(a)]) for a in range(row_start, row_end)] +\
                    padding
                reg_table.add_row(reg_value_row)

        # Affichage des tableaux précédemment créés
        self.trace_f.write(r'\hrule \vspace{0.5cm}' + '\n')
        self.trace_f.write('Cycle: %d\n \n' % simulator.clock)
        self.trace_f.write('Program Counter : %d\n \n' % simulator.PC)
        self.trace_f.write('\n \n' + rs_table.get_table() + '\n \n')
        self.trace_f.write('\n \n' + rob_table.get_table() + '\n \n')
        self.trace_f.write('\n \n' + reg_table.get_table() + '\n \n')

        self.trace_f.flush()





