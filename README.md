# MIPSSIM

Simulateur de code MIPS conçu pour le cours GIF-3000 à l'Université Laval. Testé avec Python 2.7 et 3.3.

Pour vous servir du simulateur, il faut lancer le fichier `mipssim/mipssim.py` avec comme paramètres le nom d'un fichier de configuration (placés dans le dossier `conf/`), suivi d'un fichier de code assembleur (placés dans le dossier `asm/`). Concrètement, l'interface est la suivante :

    :::text
    usage: mipssim.py config_file source_file [trace_file]


Le programme effectuera une simulation du code donné sur un microprocesseur avec une architecture MIPS paramétrisée par le contenu du fichier de configuration (nombre d'unités fonctionnelles, nombre de cycles pour exécuter les opérations à virgule flottante, etc.).

Le simulateur imprimera l'état de la mémoire et des registres au début et à la fin de l'exécution. Pour avoir plus d'information, il faudra donner un troisième paramètre, soit un nom de fichier dans lequel déposer une *trace* de l'exécution. Ce fichier contiendra le contenu du tampon de réordonnancement, des stations de réservations et des registres à chaque cycle de la simulation. Exemple :

    :::text
    Cycle: 11
    Program Counter : 3
    Stations de réservation:
    +---------+-------+-----+-----+----+----+------+---+
    | Station |   Op  |  Vj |  Vk | Qj | Qk | Dest | A |
    +---------+-------+-----+-----+----+----+------+---+
    |  Load1  |  L.D  | 184 |     |    |    |  5   | 0 |
    |  Store1 |  S.D  |     | 184 | 6  |    |  7   | 0 |
    |   Add1  | ADD.D |     | 3.0 | 5  |    |  6   |   |
    |  Mult1  |       |     |     |    |    |      |   |
    |   ALU1  |       |     |     |    |    |      |   |
    | Branch1 |       |     |     |    |    |      |   |
    +---------+-------+-----+-----+----+----+------+---+
    Registres:
    +------+------+-----+-----+-----+------+-----+-----+-----+-----+-----+
    |      |  0   |  1  |  2  |  3  |  4   |  5  |  6  |  7  |  8  |  9  |
    +------+------+-----+-----+-----+------+-----+-----+-----+-----+-----+
    | ROB# |      |  3  |     |     |      |     |     |     |     |     |
    |  R0  |  0   | 192 |  -8 |  0  |  0   |  0  |  0  |  0  |  0  |  0  |
    | ROB# |      |     |     |     |      |     |  X  |  X  |  X  |  X  |
    | R10  |  0   |  0  |  0  |  0  |  0   |  0  |  X  |  X  |  X  |  X  |
    | ROB# |  5   |     |     |     |  6   |     |     |     |     |     |
    |  F0  | 10.0 | 0.0 | 3.0 | 0.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
    | ROB# |      |     |     |     |      |     |  X  |  X  |  X  |  X  |
    | F10  | 0.0  | 0.0 | 0.0 | 0.0 | 0.0  | 0.0 |  X  |  X  |  X  |  X  |
    +------+------+-----+-----+-----+------+-----+-----+-----+-----+-----+
    ROB:
    +--------+-------------+---------------------+------+-------+--------+
    | Entrée | Instruction |                     | État | Dest. | Valeur |
    +--------+-------------+---------------------+------+-------+--------+
    |   2    |     S.D     |   ['F4', '0(R1)']   |  3   |       |  13.0  |
    |   3    |    DADDI    | ['R1', 'R1', '#-8'] |  3   |   R1  |  184   |
    |   4    |     BNE     |  ['R1', 'R2', '#0'] |  3   |       |  True  |
    |   5    |     L.D     |   ['F0', '0(R1)']   |  1   |   F0  |        |
    |   6    |    ADD.D    |  ['F4', 'F0', 'F2'] |  1   |   F4  |        |
    |   7    |     S.D     |   ['F4', '0(R1)']   |  1   |       |        |
    +--------+-------------+---------------------+------+-------+--------+


## Architecture du simulateur

Le module `simulator.py` est le module principal du simulateur. Une fois l'initialisation terminée, le simulateur contient une table de registres dans `self.regs`, un tampon de reordonnancement dans `self.ROB` et deux listes imbriquées contenant les stations de réservations dans `self.RS` (une première liste séparant les unités fonctionnelles par type, et dans chacune une autre liste).

Les fonctions les plus importantes de la classe `Simulator` sont les suivantes :

* `go`: démarre la simulation, boucle tant que le programme n'est pas complètement exécuté.

* `step`: fonction appelée à chaque itération dans la fonction `go`. Exécute les grands blocs de la simulation dans l'ordre suivant :
    * sanctionner l'instruction à la tête du ROB (fonction `commit`),
    * décrémenter le temps dans les unités fonctionnelles en exécution et terminer leur exécution lorsqu'applicable (`decrement_time`),
    * incrémenter le *program counter* (PC) ou bien le mettre à la valeur précisée par un branchement précédent, puis lancer une nouvelle instruction (`issue`)
    * terminer l'exécution si le PC pointe vers la dernière instruction et le ROB est vide.

*N.B. : l'ordre des opérations est inversé pour ne pas lancer une instruction et l'exécuter dans le même cycle.

* `load_config`: charge la configuration en mémoire et créé les éléments requis en fonction de ce qui est chargé. Par exemple, le nombre et le type des unités fonctionnelles varie selon ce qui est écrit dans la configuration et c'est cette fonction qui est en cause.

L'autre module qui sera important pour votre projet est le module `components.py`, contenant les définitions des composantes du simulateur. Les classes pour le ROB, les registres et les unités fonctionnelles se trouvent dans ce module. Vous aurez sans doute à y ajouter un ou des éléments.

### Composantes

Dans cette section, un peu d'information est fournie sur les composantes du simulateur. Il est bien possible que cette information ne vous soit pas directement utile, mais elle pourra vous aider à développer une meilleure compréhension de la structure interne du simulateur.

La structure des *registres* est une classe encapsulant un dictionnaire de type `OrderedDict`. Tous les éléments de la mémoire sont accessibles avec les opérateurs crochets et une string, par exemple :

    :::python
    # Lecture
    F0 = self.regs['F0']
    # Écriture
    self.regs['F1'] = 8

Les registres contiennent également une deuxième liste (dans `regs.stat`) qui indique l'état de chaque registre. Si le champ d'un registre dans cette table contient une valeur, cela veut dire que le registre est occupé et en attente d'une valeur. Il ne faudra donc pas y écrire directement.

Le *reorder buffer* est une tampon circulaire, donc il ne faut pas l'accéder directement. On itère sur celui-ci de la manière suivante :

    :::python
    for entry in self.ROB:
        # Vérification

Pour accéder à la tête du ROB, on peut utiliser la variable `ROB.start`:

    :::python
    head_entry = self.ROB[self.ROB.start]
    if head_entry.state == ... # etc.

Finalement, les stations de réservations sont contenues dans deux listes imbriquées, plus précisément un `OrderedDict` contenant des listes.

    :::python
    #Stations de réservation de type Load
    load_units = self.RS['Load']
    for l in load_units:
        # faire quelque chose avec l

### Aide

L'aide d'utilisation fournie par le programme avec le drapeau `-h` est la suivante :

    :::text
    usage: mipssim.py [-h] [-L LATEX_TRACE_FILE] [-d]
                      config_file source_file [trace_file]

    Simulateur de MIPS en Python (2.7+). Testé avec Python 2.7 et 3.3.

    positional arguments:
      config_file          Fichier xml contenant la configuration du simulateur.
      source_file          Fichier contenant le code source à exécuter.
      trace_file           Ficher dans lequel sera écrit l'état du simulateur à
                           tous les pas de temps. (default: None)

    optional arguments:
      -h, --help           show this help message and exit
      -L LATEX_TRACE_FILE  Fichier pour écrire une trace sous format LaTeX
                           (surtout les tableaux). (default: None)
      -d                   Force l'impression de davantage d'information à chaque
                           étape de l'exécution dans la ligne de commande.
                           (default: False)

