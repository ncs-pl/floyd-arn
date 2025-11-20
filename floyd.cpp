// Copyright (C) 2025 Nicolas Paul <nicolas.paul1@etu.univ-orleans.fr> and
// Tolunay Akkaya <tolunay.akkaya@etu.univ-orleans.fr>.
//
// This file is part of Projet Floyd.
//
// Projet Floyd is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Ce fichier propose une parallélisation de l'algorithme de Floyd-Warshall
// en utilisant MPI.

// mpirun -n N ./floyd example.dot > output.txt


#include <fstream>
#include <iostream>

#include <mpi.h>

int
main(int argc, char **argv)
{
    // Initialisation et récupération des arguments du programme.

    MPI_Init(&argc, &argv);

    int worker_id;
    int workers_count;
    MPI_Comm_rank(MPI_COMM_WORLD, &worker_id);
    MPI_Comm_size(MPI_COMM_WORLD, &workers_count);

    std::string exe = argv[0];
    if (argc != 3) {
        std::cerr << "Utilisation : " << exe << " <root> <input>" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    int root_id = 0;
    {
        char* end = nullptr;
        long value = std::strtol(argv[1], &end, 10);
        if (*end != '\0') {
            std::cerr << exe << " : \"" << argv[1] << "\" n'est pas un entier valide." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        if (value < 0) {
            std::cerr << exe << " : root doit être supérieur ou égal à 0." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        if (value > std::numeric_limits<int>::max()) {
            std::cerr << exe << " : root doit être strictement inférieur à " << std::numeric_limits<int>::max() << "." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return 1;
        }

        root_id = static_cast<int>(value);
    }

    std::string input = argv[2];
    std::istream file(input);
    if (!file.good()) {
        std::cerr << exe << " : le fichier \"" << filename << "\" est introuvable." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return 1;
    }

    // Compiler le fichier DOT en une matrice adjacente.

    // Repartition des données sur tous les travailleurs disponibles.

    // Algorithme de Floyd-Warshall en parallèle.

    // Unifier la matrice de distance sur le travailleur principal.

    // Afficher la matrice de distance.

    // Nettoyer la mémoire allouée

    MPI_Finalize();
    return 0;
}
