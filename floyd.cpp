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
// en utilisant OpenMPI 5+ et ISO C++ 11.

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <map>

#include <graphviz/cgraph.h>
#include <mpi.h>

constexpr int kInfinity = std::numeric_limits<int>::max();
constexpr int kTrue = 1;
constexpr int kFalse = 0;

int
main(int argc, char **argv)
{
  int status = 0; // pas EXIT_SUCCESS...
  int i, j, k; // itérateurs de boucle

  int pid, nprocs;
  int root;
  int b; // longueur d'un bloc
  char* input = nullptr;
  int *A = nullptr;

  FILE* fd = nullptr; // file descriptor d'input
  std::map<std::string, int> index;
  Agraph_t *G;
  Agnode_t *u, *v;
  Agedge_t *e;

  int *D = nullptr;
  int *KC = nullptr; // colonne pivots
  int *KR = nullptr; // ligne pivots

  int n; // nombre de sommets.
  int q; // nombre de blocs

  MPI_Comm ccart, crow, ccol;
  int *co = nullptr; // (ligne, colonne)
  int *dims = nullptr;
  int *pds = nullptr;

  // Récupération des arguments du programme.

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc != 4) {
    std::cout << "usage: " << argv[0] << " <root> <b> <input>" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  root = std::stoi(argv[1], nullptr, 10);

  if (root < 0 || root >= nprocs) {
    std::cerr << argv[0] << ": root doit être un processeur valide" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  b = std::stoi(argv[2], nullptr, 10);
  if (b <= 0) {
    std::cerr << argv[0] << ": b doit être positif" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  input = argv[3];

  // Initialisation de la matrice A.


  if (pid == root) {
    // Lecture et compilation.

    fd = std::fopen(input, "r");
    G = agread(fd, nullptr);
    fclose(fd);

    n = agnnodes(G);

    i = 0;
    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u))
    {
      index.emplace(agnameof(u), i++);
    }

    // Initialisation de A à l'infini sauf sur les diagonales.

    A = new int[n*n];

    for (i = 0; i < n*n; i++) {
      A[i] = kInfinity;
    }

    for (i = 0; i < n; i++) {
      A[i*n+i] = 0;
    }

    // Construction de la matrice adjacente.

    for (u = agfstnode(G); u; u = agnxtnode(G, u)) {
      i = index[agnameof(u)];
      for (e = agfstout(G, u); e; e = agnxtout(G, e)) {
        v = aghead(e);
        j = index[agnameof(v)];
        A[i*n+j] = A[j*n+i] = std::stoi(agget(e, (char*)"weight"), nullptr, 10);
      }
    }

    agclose(G);

    // Affichage de A.

    std::cout << "Matrice adjacente :" << std::endl;

    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        std::cout << "    "
	          << (A[i*n+j] == kInfinity ? "∞" : std::to_string(A[i*n+j]))
		  << " ";
      }
      std::cout << std::endl;
    }
  }

  // Découpage de A en blocs locaux D.

  MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);

  D = new int[b*b]; // hypothèses dans le sujet.
  MPI_Scatter(A, b*b, MPI_INT, D, b*b, MPI_INT, root, MPI_COMM_WORLD);

  // Création de la topologie cartésienne.

  // TODO(nico): vérifier (n/b)^2 = nprocs
  dims = new int[2];
  dims[0] = dims[1] = n/b;

  pds = new int[2];
  pds[0] = pds[1] = kTrue;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, kTrue, &ccart);

  co = new int[2];
  MPI_Cart_coords(ccart, pid, 2, co);

  MPI_Comm_split(ccart, co[0], co[1], &crow);
  MPI_Comm_split(ccart, co[1], co[0], &ccol);

  // Algorithme de Floyd-Warshall.

  KR = new int[b];
  KC = new int[b];

  for (k = 0; k < n; k++) {
    // Identifier les pivots.

    if (co[0] == k/b) {
      for (i = 0; i < b; i++) {
        KR[i] = D[(k%b)*b+i];
      }
    }

    if (co[1] == k/b) {
      for (i = 0; i < b; i++) {
        KC[i] = D[i*b+(k%b)];
      }
    }

    // Partage des pivots.

    // TODO: comment bien décider du "root" ?
    MPI_Bcast(KR, b, MPI_INT, ??, crow);
    MPI_Bcast(KC, b, MPI_INT, ??, ccol);

    // Mise à jour des blocs locaux.

    for (i = 0; i < b; i++) {
      for (j = 0; j < n; j++) {
        D[i*b+j] = std::min(D[i*b+j], KC[i] + KR[j]);
      }
    }
  }

  // Récupération des blocs locaux.

  MPI_Gather(D, b*b, MPI_INT, A, b*b, MPI_INT, root, MPI_COMM_WORLD);

  // Affichage final.

  if (pid == root) {
    std::cout << "Matrice finale :" << std::endl;

    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        std::cout << "    " << A[i*n+j] << " ";
      }
      std::cout << std::endl;
    }
  }

  // Nettoyage du programe.

cleanup:
  delete[] coords;
  delete[] periods;
  delete[] dims;
  delete[] KR;
  delete[] KC;
  delete[] D;
  delete[] A;

  MPI_Comm_free(&comm_col);
  MPI_Comm_free(&comm_row);
  MPI_Comm_free(&comm_cart);

  MPI_Finalize();
  return status;
}

