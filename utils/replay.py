# -*- coding: utf-8 -*-
#
# Copyright (c) 2011, Yannick Hold-Geoffroy and contributors.
#
# Distributed under the terms of the MIT license. See the COPYING file at
#  the top-level directory of this project and at
#  https://bitbucket.org/ulaval-gif-3000/mipssim/raw/tip/COPYING

import sys

def main(argv=None):
    """
    Permet d'afficher une trace par coup l'horloge.
    """
    argv = sys.argv if argv == None else argv
    try:
        # Buffer file in input
        fichier = open(argv[1], 'r')
        le_fichier = fichier.readlines()
        fichier.close()
        index = 1
        last_screen = 1
        while True:
            if index > len(le_fichier):
                break
            ligne = le_fichier[index]
            print(ligne)
            if ligne.strip()[:3] == '===':
                cmd = input().strip()
                if cmd.lower() == 'a':
                    for buffer_index in range(last_screen - 1, -1, -1):
                        if le_fichier[buffer_index][:3] == '===':
                            print(buffer_index)
                            index = buffer_index
                            break
                last_screen = index
            index += 1
        return 0
    except:
        print("Veuillez spécifier un fichier valide en argument")
        print("Entrez \"a\" pour revenir en arrière de 1 cycle.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
