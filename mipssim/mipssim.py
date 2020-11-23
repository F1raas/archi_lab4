# -*- coding: utf-8 -*-
#
# Copyright (c) 2011-2014, Julien-Charles Lévesque <levesque.jc@gmail.com>
#  and contributors.
#
# Distributed under the terms of the MIT license. See the COPYING file at
#  the top-level directory of this project and at
#  https://bitbucket.org/ulaval-gif-3000/mipssim/raw/tip/COPYING

'''

'''

import argparse
import sys
if sys.version_info < (2, 7):
    print('ATTENTION!!!')
    print('Vous tentez de lancer mipssim.py sur une version de Python inférieure à 2.7!')
    print('Python 2.6 et ses versions antérieures ne supportent pas OrderedDict, qui')
    print('est requise pour ce simulateur.')
    sys.exit(1)

import interpreter as interp
import simulator as sim

auteurs = ''

def main(config_file, source_file, trace_file, latex_trace_file, debug):
    # Génération du simulateur
    simulator = sim.Simulator(config_file, source_file, trace_file, latex_trace_file, debug)

    # Affichage de l'état initial de la mémoire et des registre.
    print('État initial des registres: ' + str(simulator.regs))
    print('État initial de la mémoire: ' + str(simulator.mem))

    # Démarrage du simulateur
    print('Démarrage de la simulation.')
    err = simulator.go()
    print('Arrêt de la simulation.')

    # Affichage de l'état final de la mémoire et des registre.
    print('État final des registres : ' + str(simulator.regs))
    print('État final de la mémoire : ' + str(simulator.mem))

    # Affichage de l'état du processus à sa terminaison.
    if err == 0:
        print('Simulation terminée avec succès.')
    else:
        print('Une erreur est survenue lors de l\'exécution du programme.')
    return err, simulator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulateur de MIPS en Python (2.7+). Testé avec\
 Python 2.7 et 3.3.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_file', help="Fichier xml contenant la configuration du simulateur.")
    parser.add_argument('source_file', help="Fichier contenant le code source à exécuter.")
    parser.add_argument('trace_file', nargs='?', help="Ficher dans lequel sera écrit \
l'état du simulateur à tous les pas de temps.")
    parser.add_argument('-L', dest='latex_trace_file', help="Fichier pour écrire une trace sous \
format LaTeX (surtout les tableaux).")
    parser.add_argument('-d', default=False, action='store_true', dest='debug', help="Force \
l'impression de davantage d'information à chaque étape de l'exécution dans la ligne de commande.")

    args = parser.parse_args()

    err, simulator = main(args.config_file, args.source_file, args.trace_file,
        args.latex_trace_file, args.debug)
