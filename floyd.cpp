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

constexpr int kInfinity = std::numeric_limits<int>::max();
constexpr int kScatterTag = 1;
constexpr int kGatherTag = 2;

class Matrix {
private:
  size_t m_rows, m_columns;
  std::vector<int> m_buffer;

public:
  Matrix(size_t rows, size_t columns) : m_rows(rows), m_columns(columns), m_buffer(rows * columns) {}

  int& operator()(size_t i, size_t j) { return m_buffer[i * m_columns + j]; }
  int operator()(size_t i, size_t j) const { return m_buffer[i * m_columns + j]; }

  size_t rows() const { return m_rows; }
  size_t cols() const { return m_columns; }
  size_t size() const { return m_rows * m_columns; }
  int *data() { return m_buffer.data(); }
  const int *data() const { return m_buffer.data(); }
};

std::ostream& operator<<(std::ostream& os, const Matrix& M) {
  for (size_t i = 0; i < M.rows(); ++i) {
    for (size_t j = 0; j < M.cols(); ++j) {
      int v = M(i, j);

      if (v == kInfinity) {
        os << "∞";
      } else {
        os << v;
      }

      if (j + 1 < M.cols()) {
        os << ' ';
      }
    }

    if (i + 1 < M.rows()) {
      os << '\n';
    }
  }

  return os;
}

void
CreateTopology(MPI_Comm wld, MPI_Comm *cart, MPI_Comm *rows, MPI_Comm *cols)
{
  int np = 0;
  MPI_Comm_size(wld, &np);

  int dims[2] = {0, 0};
  MPI_Dims_create(np, 2, dims);

  int pds[2] = {false, false};
  MPI_Cart_create(wld, 2, dims, pds, true, cart);

  int rdims[2] = {false, true};
  int cdims[2] = {true, false};
  MPI_Cart_sub(*cart, rdims, rows);
  MPI_Cart_sub(*cart, cdims, cols);
}

void
CompileGraphFile(std::string& input, Matrix& A, int pid, int root)
{
  if (pid == root) {
    FILE *fd = std::fopen(input.c_str(), "r");
    Agraph_t *G = agread(fd, nullptr);
    fclose(fd);

    int n = agnnodes(G);

    A = Matrix(n, n);
    std::fill(A.data(), A.data() + A.size(), kInfinity);
    for (size_t i = 0; i < A.rows(); ++i) {
      A(i, i) = 0;
    }

    std::map<std::string, int> index;
    
    std::vector<std::string> nodeNames;
    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u)) {
      nodeNames.push_back(agnameof(u));
    }
    
    // NOTE(nico): tri ici parce que GraphViz ne garantit pas
    // l'ordre des sommets, même si parfois il respectera celui
    // défini dans le fichier.
    std::sort(nodeNames.begin(), nodeNames.end());
    
    for (size_t i = 0; i < nodeNames.size(); ++i) {
      index[nodeNames[i]] = i;
    }

    for (Agnode_t *u = agfstnode(G); u; u = agnxtnode(G, u)) {
      const int i = index[agnameof(u)];

      for (Agedge_t *e = agfstout(G, u); e; e = agnxtout(G, e)) {
        Agnode_t *v = aghead(e);
        const int j = index[agnameof(v)];
        const int weight = std::stoi(agget(e, (char*)"weight"));
        A(i, j) = weight;
        A(j, i) = weight;
      }
    }

    agclose(G);
    std::cout << "Matrice adjacente :" << std::endl;
    std::cout << A << std::endl;
  }
}

void
ScatterBlocks(Matrix& A, Matrix& D, int pid, int root, MPI_Comm comm)
{
  if (pid == root) {
    for (size_t i = 0; i < A.rows() / D.rows(); ++i) {
      for (size_t j = 0; j < A.cols() / D.cols(); ++j) {
        Matrix T(D.rows(), D.cols());
        for (size_t r = 0; r < D.rows(); ++r) {
          const int oft = (i*D.rows() + r)*A.cols() + j*D.cols();
          std::copy(A.data() + oft, A.data() + oft + D.cols(), T.data() + r*D.cols());
        }

        int dst_co[2] = {(int)i, (int)j};
        int dst = 0;
        MPI_Cart_rank(comm, dst_co, &dst);

        if (dst == root) {
          D = T;
        } else {
          // TODO: non-blocking IO
          MPI_Send(T.data(), T.size(), MPI_INT, dst, kScatterTag, comm);
        }
      }
    }
  } else {
    // TODO: non-blocking IO
    MPI_Recv(D.data(), D.size(), MPI_INT, root, kScatterTag, comm, MPI_STATUS_IGNORE);
  }
}


void
FloydWarshall(Matrix& D, size_t n, size_t row, size_t col, MPI_Comm rows_comm,
  MPI_Comm cols_comm)
{
  size_t b = D.rows();
  
  for (size_t k = 0; k < n; ++k) {
    Matrix R(b, b); // ligne pivot pour notre colonne
    std::fill(R.data(), R.data() + R.size(), kInfinity);

    Matrix C(b, b); // colonne pivot pour notre ligne
    std::fill(C.data(), C.data() + C.size(), kInfinity);
    
    if (col == k/b) {
      for (size_t i = 0; i < b; ++i) {
        for (size_t j = 0; j < b; ++j) {
          C(i, j) = D(i, j);
        }
      }
    }
    
    MPI_Bcast(C.data(), C.size(), MPI_INT, k/b, rows_comm);
    
    if (row == k/b) {
      for (size_t i = 0; i < b; ++i) {
        for (size_t j = 0; j < b; ++j) {
          R(i, j) = D(i, j);
        }
      }
    }
    
    MPI_Bcast(R.data(), R.size(), MPI_INT, k/b, cols_comm);
    
    for (size_t i = 0; i < b; ++i) {
      for (size_t j = 0; j < b; ++j) {
        int ik = C(i, k%b);
        int kj = R(k%b, j);
        
        if (ik != kInfinity && kj != kInfinity) {
          D(i, j) = std::min(D(i, j), ik + kj);
        }
      }
    }
  }
}

void
GatherBlocks(Matrix& A, Matrix& D, int pid, int root, MPI_Comm comm)
{
  if (pid == root) {
    for (size_t i = 0; i < A.rows() / D.rows(); ++i) {
      for (size_t j = 0; j < A.cols() / D.cols(); ++j) {
        Matrix T(D.rows(), D.cols());
        int src_co[2] = {(int)i, (int)j};
        int src = 0;
        MPI_Cart_rank(comm, src_co, &src);

        if (src == root) {
          T = D;
        } else {
          // TODO: non-blocking IO
          MPI_Recv(T.data(), T.size(), MPI_INT, src, kGatherTag, comm, MPI_STATUS_IGNORE);
        }


        for (size_t r = 0; r < T.rows(); ++r) {
          std::copy(T.data() + r * T.cols(), T.data() + (r + 1) * T.cols(), A.data() + (i * T.rows() + r) * A.cols() + j * T.cols());
        }
      }
    }

    std::cout << "Matrice des distances :" << std::endl;
    std::cout << A << std::endl;
  } else {
    // TODO: non-blocking IO
    MPI_Send(D.data(), D.size(), MPI_INT, root, kGatherTag, comm);
  }
}

int
main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  if (argc != 4) {
    std::cout << argv[0] << " root b input" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int root = std::stoi(argv[1], nullptr, 10);
  int b = std::stoi(argv[2], nullptr, 10);
  std::string input = argv[3];

  MPI_Comm cart_comm, rows_comm, cols_comm;
  CreateTopology(MPI_COMM_WORLD, &cart_comm, &rows_comm, &cols_comm);
  int pid = 0;
  std::array<int, 2> coords{0, 0};
  MPI_Comm_rank(cart_comm, &pid);
  MPI_Cart_coords(cart_comm, pid, 2, coords.data());

  Matrix A(0, 0);
  CompileGraphFile(input, A, pid, root);

  int n = A.rows();
  MPI_Bcast(&n, 1, MPI_INT, root, cart_comm);

  Matrix D(b, b);
  std::fill(D.data(), D.data() + D.size(), kInfinity);
  ScatterBlocks(A, D, pid, root, cart_comm);

  FloydWarshall(D, n, coords[0], coords[1], rows_comm, cols_comm);

  GatherBlocks(A, D, pid, root, cart_comm);

  MPI_Comm_free(&cart_comm);
  MPI_Comm_free(&rows_comm);
  MPI_Comm_free(&cols_comm);
  MPI_Finalize();
  return 0;
}

// vim: ft=cpp expandtab ts=2 sw=2 sts=2
