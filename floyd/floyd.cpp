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

// mpirun -np 4 ./floyd 0 2 bytequest.dot
// mpirun -np 4 ./floyd 0 4 petit.dot
// mpirun -np 16 ./floyd 0 5 grand.dot

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <graphviz/cgraph.h>
#include <mpi.h>
#include <limits>

constexpr int kInfinity = std::numeric_limits<int>::max();
constexpr int kScatterTag = 1;
constexpr int kGatherTag = 2;

class Matrix {
private:
  std::size_t m_rows, m_columns;
  std::vector<int> m_buffer;

public:
  Matrix(std::size_t n, std::size_t m) : m_rows(n), m_columns(m), m_buffer(n*m)
  {}

  int&
  operator()(std::size_t i, std::size_t j)
  {
    return m_buffer[i*m_columns+j];
  }

  int
  operator()(std::size_t i, std::size_t j) const
  {
    return m_buffer[i*m_columns+j];
  }

  std::size_t
  rows() const
  {
    return m_rows;
  }

  std::size_t
  cols() const
  {
    return m_columns;
  }

  std::size_t
  size() const
  {
    return m_rows * m_columns;
  }

  int *
  data()
  {
    return m_buffer.data();
  }

  const int *
  data() const
  {
    return m_buffer.data();
  }
};

std::ostream& operator<<(std::ostream& s, const Matrix& M) {
  for (std::size_t i = 0; i < M.rows(); ++i) {
    for (std::size_t j = 0; j < M.cols(); ++j) {
      int v = M(i, j);
      s << (v == kInfinity ? "∞" : std::to_string(v)) << ' ';
    }

    if (i + 1 < M.rows()) s << std::endl;
  }

  return s;
}

int
main(int argc, char **argv)
{
  int np = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &np);


  if (argc != 4) {
    std::cout << argv[0] << " root b input" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // TODO(nico): check for contracts?
  int root = std::stoi(argv[1], nullptr, 10);
  std::size_t b = static_cast<std::size_t>(std::stoi(argv[2], nullptr, 10));
  std::string input = argv[3];

  // Création de la topologie.

  MPI_Comm cart_comm, rows_comm, cols_comm;

  int dims[2] = {0, 0};
  MPI_Dims_create(np, 2, dims);

  int periods[2] = {false, false};
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &cart_comm);

  int pid = 0;
  std::array<int, 2> coords{0, 0};
  MPI_Comm_rank(cart_comm, &pid);
  MPI_Cart_coords(cart_comm, pid, 2, coords.data());

  int rdims[2] = {false, true};
  int cdims[2] = {true, false};
  MPI_Cart_sub(cart_comm, rdims, &rows_comm);
  MPI_Cart_sub(cart_comm, cdims, &cols_comm);

  // Compilation GraphViz et création de la matrice adjacente.

  std::size_t n = 0;
  Matrix A(0, 0);

  if (pid == root) {
    FILE *fd = std::fopen(input.c_str(), "r");
    Agraph_t *G = agread(fd, nullptr);
    fclose(fd);

    n = static_cast<std::size_t>(agnnodes(G));
        std::cout << "Nombre de sommets lus par Graphviz : " << n << std::endl;

    A = Matrix(n, n);

    // NOTE(nico): toujours s'assurer de la construction.
    std::fill(A.data(), A.data() + A.size(), kInfinity);
    for (std::size_t i = 0; i < A.rows(); ++i) A(i, i) = 0;


    std::vector<std::string> labels; // noms des sommets.
    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u)) {
      // NOTE(nico): ordre non-déterministe ^^^^^^^^^^^^^^^
      labels.push_back(agnameof(u));
    }

    // NOTE(nico): tri ici parce que GraphViz ne garantit pas
    // l'ordre des sommets, même si parfois il respectera celui
    // défini dans le fichier.
    std::sort(labels.begin(), labels.end());

    std::map<std::string, int> index;
    for (std::size_t i = 0; i < labels.size(); ++i) index[labels[i]] = i;

    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u)) {
      const int i = index[agnameof(u)];

      for (Agedge_t *e = agfstout(G, u); e; e = agnxtout(G, e)) {
        Agnode_t *v = aghead(e);
        const int j = index[agnameof(v)];
        const int w = std::stoi(agget(e, (char*)"weight"));
        A(i, j) = w; A(j, i) = w;
      }
    }

    agclose(G);
    std::cout << "Matrice adjacente :" << std::endl;
    std::cout << A << std::endl;
  }

  MPI_Bcast(&n, 1, MPI_INT, root, cart_comm);

  // Division en blocs et répartition des blocs.

  Matrix D(b, b);
  std::fill(D.data(), D.data() + D.size(), kInfinity);

  if (pid == root) {
    for (std::size_t i = 0; i < A.rows() / D.rows(); ++i) {
      for (std::size_t j = 0; j < A.cols() / D.cols(); ++j) {
        Matrix T(D.rows(), D.cols());

        for (std::size_t r = 0; r < D.rows(); ++r) {
          const int oft = (i*D.rows() + r)*A.cols() + j*D.cols();
          std::copy(A.data()+oft,
                    A.data()+oft+D.cols(),
                    T.data()+r*D.cols());
        }

        int dst_co[2] = {(int)i, (int)j};
        int dst = 0;
        MPI_Cart_rank(cart_comm, dst_co, &dst);

        if (dst == root) {
          D = T;
        } else {
          // TODO: non-blocking IO
          MPI_Send(T.data(),
                   T.size(),
                   MPI_INT,
                   dst,
                   kScatterTag,
                   cart_comm);
        }
      }
    }
  } else {
    // TODO: non-blocking IO
    MPI_Recv(D.data(),
             D.size(),
             MPI_INT,
             root,
             kScatterTag,
             cart_comm,
             MPI_STATUS_IGNORE);
  }


  // Algorithme de Floyd-Warshall.

  for (std::size_t k = 0; k < n; ++k) {
    Matrix R(b, b); // ligne pivot pour notre colonne
    std::fill(R.data(), R.data() + R.size(), kInfinity);

    Matrix C(b, b); // colonne pivot pour notre ligne
    std::fill(C.data(), C.data() + C.size(), kInfinity);

    if (static_cast<std::size_t>(coords[1]) == k/b) {
      for (std::size_t i = 0; i < b; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
          C(i, j) = D(i, j);
        }
      }
    }

    MPI_Bcast(C.data(), C.size(), MPI_INT, k/b, rows_comm);

    if (static_cast<std::size_t>(coords[0]) == k/b) {
      for (std::size_t i = 0; i < b; ++i) {
        for (std::size_t j = 0; j < b; ++j) {
          R(i, j) = D(i, j);
        }
      }
    }

    MPI_Bcast(R.data(), R.size(), MPI_INT, k/b, cols_comm);

    for (std::size_t i = 0; i < b; ++i) {
      for (std::size_t j = 0; j < b; ++j) {
        int ik = C(i, k%b);
        int kj = R(k%b, j);

        if (ik != kInfinity && kj != kInfinity) {
          D(i, j) = std::min(D(i, j), ik + kj);
        }
      }
    }
  }

  // Récupération des blocs.

  if (pid == root) {
    for (std::size_t i = 0; i < A.rows() / D.rows(); ++i) {
      for (std::size_t j = 0; j < A.cols() / D.cols(); ++j) {
        Matrix T(D.rows(), D.cols());

        int src_co[2] = {(int)i, (int)j};
        int src = 0;
        MPI_Cart_rank(cart_comm, src_co, &src);

        if (src == root) {
          T = D;
        } else {
          // TODO: non-blocking IO
          MPI_Recv(T.data(),
                   T.size(),
                   MPI_INT,
                   src,
                   kGatherTag,
                   cart_comm,
                   MPI_STATUS_IGNORE);
        }


        for (std::size_t r = 0; r < T.rows(); ++r) {
          std::copy(T.data()+r*T.cols(),
                    T.data()+(r+1)*T.cols(),
                    A.data()+(i*T.rows()+r)*A.cols()+j*T.cols());
        }
      }
    }


  } else {
    // TODO: non-blocking IO
    MPI_Send(D.data(),
             D.size(),
             MPI_INT,
             root,
             kGatherTag,
             cart_comm);
  }

  if(pid == root){
    std::cout << "Matrice des distances :" << std::endl;
    std::cout << A << std::endl;

  }

  MPI_Comm_free(&cols_comm);
  MPI_Comm_free(&rows_comm);
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();
  return 0;
}

// vim: ft=cpp expandtab ts=2 sw=2 sts=2