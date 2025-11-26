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

int
main(int argc, char **argv)
{
  int status = EXIT_SUCCESS;

  int *A = nullptr;
  int *D = nullptr;
  int *Q = nullptr;
  char* input = nullptr;

  // Récupération des arguments du programme.

  MPI_Init(&argc, &argv);

  int pid, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (argc != 4)
  {
    std::cout << "usage: " << argv[0] << " <root> <b> <input>" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  int root;
  try
  {
    root = std::stoi(argv[1], nullptr, 10);

    if (root < 0 || root >= nprocs) {
      std::cerr << argv[0] << ": root doit être un processeur valide" << std::endl;
      status = EXIT_FAILURE;
      goto cleanup;
    }
  }
  catch (std::invalid_argument const& _ex)
  {
    std::cerr << argv[0] << ": root doit être un entier" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }
  catch (std::out_of_range const& _ex)
  {
    std::cerr << argv[0] << ": root doit être une valeur raisonnable" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  int b;
  try
  {
    b = std::stoi(argv[2], nullptr, 10);

    if (b <= 0) {
      std::cerr << argv[0] << ": b doit être positif" << std::endl;
      status = EXIT_FAILURE;
      goto cleanup;
    }
  }
  catch (std::invalid_argument const& _ex)
  {
    std::cerr << argv[0] << ": b doit être un entier" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }
  catch (std::out_of_range const& _ex)
  {
    std::cerr << argv[0] << ": b doit être une valeur raisonnable" << std::endl;
    status = EXIT_FAILURE;
    goto cleanup;
  }

  input = argv[3];

  // Initialisation de la matrice A.

  int n;

  if (pid == root)
  {
    // Lecture et compilation.

    FILE *fd = std::fopen(input, "r");
    if(!fd)
    {
      std::cerr << argv[0] << ": \"" << input << "\" ne peut pas être ouvert" << std::endl;
      status = EXIT_FAILURE;
      goto cleanup;
    }

    Agraph_t *G = agread(fd, nullptr);
    if (!G)
    {
      std::cerr << argv[0] << ": \"" << input << "\" ne peut pas être compilé" << std::endl;
      status = EXIT_FAILURE;
      goto cleanup;
    }

    fclose(fd);

    n = agnnodes(G);
    std::map<std::string, int> index;

    int t = 0;
    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u))
    {
      index.emplace(agnameof(u), t++);
    }

    // Initialisation de A à l'infini sauf sur les diagonales.

    A = new int[n*n];

    for (int i = 0; i < n*n; i++)
    {
      A[i] = kInfinity;
    }

    for (int i = 0; i < n; i++)
    {
      A[i*n+i] = 0;
    }

    // Construction de la matrice adjacente.

    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u))
    {
      int const i = index[agnameof(u)];
      for (Agedge_t *e = agfstout(G, u); e; e = agnxtout(G, e))
      {
        Agnode_t *v = aghead(e);
        int const j = index[agnameof(v)];

        int w;
        try
        {
	  std::string attr = agget(e, (char*)"weight");
          w = std::stoi(attr, nullptr, 10);
        }
        catch (std::invalid_argument const& _ex)
        {
          std::cerr << argv[0] << ": le weight d'une arête doit être un entier" << std::endl;
          status = EXIT_FAILURE;
          goto cleanup;
        }
        catch (std::out_of_range const& _ex)
        {
          std::cerr << argv[0] << ": le weight d'une arête doit être une valeur raisonnable" << std::endl;
          status = EXIT_FAILURE;
          goto cleanup;
        }

        A[i*n+j] = A[j*n+i] = w;
      }
    }

    agclose(G);

    // Affichage de A.

    std::cout << "Matrice adjacente :" << std::endl;

    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        std::cout << "    " << A[i*n+j] << " ";
      }

      std::cout << std::endl;
    }
  }

  // Découpage de A en blocs locaux D.

  D = new int[b*b]; // hypothèses dans le sujet.
  MPI_Bcast(&n, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Scatter(A, b*b, MPI_INT, D, b*b, MPI_INT, root, MPI_COMM_WORLD);

  // Algorithme de Floyd-Warshall.

  Q = new int[n];

  for (int l = 0; l < n; l++)
  {
    if (pid == l/b)
    {
      for (int i = 0; i < n; i++)
      {
        Q[i] = D[(l%b)*n+i];
      }
    }

    MPI_Bcast(Q, n, MPI_INT, l/b, MPI_COMM_WORLD);

    for (int i = 0; i < b; i++)
    {
      for (int j = 0; j < n; j++)
      {
	D[i*n+j] = std::min(D[i*n+j], D[i*n+l] + Q[j]);
      }
    }
  }

  // Récupération des blocs locaux.

  MPI_Gather(D, b*b, MPI_INT, A, b*b, MPI_INT, root, MPI_COMM_WORLD);

  // Affichage final.

  if (pid == root)
  {
    std::cout << "Matrice finale :" << std::endl;

    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < n; j++)
      {
        std::cout << "    " << A[i*n+j] << " ";
      }

      std::cout << std::endl;
    }
  }

  // Nettoyage du programe.

 cleanup:
   delete[] Q;
   delete[] D;
   delete[] A;

  MPI_Finalize();
  return status;
}

#if 0
  int qrows, qcols;
  int *qrow = nullptr, *qcol = nullptr; // Données de pivots.
  int q; // Coordonnée de ligne/colonne du bloc pivot.
  int d; // Nouvelle valeur calculée pour un élément de D.

  qrow = new int[b];
  qcol = new int[b];
  for (l = 0; l < n; ++l) {

    // Déterminer les blocs aux valeurs pivots et les récupérer,
    // ainsi que partager nos valeurs pivots aux autres.

    q = l/b;
    MPI_Bcast(qrow, b, MPI_INT, q*nb+q, MPI_COMM_WORLD);
    MPI_Bcast(qcol, b, MPI_INT, q*nb+q, MPI_COMM_WORLD);

    // Calculer D(l).

    for (i = 0; i < b; ++i) {
      for (j = 0; j < b; ++j) {
        d = qrow[i] + qcol[j];
	D[i*b+j] = d <= D[i*b+j] ? d : D[i*b+j];
      }
    }
  }


cleanup:
  delete[] qrow;
  delete[] qcol;
#endif
