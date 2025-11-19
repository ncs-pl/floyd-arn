// Une implémentation parallèle de l'algorithme de Floyd-Warshall en
// utilisant OpenMPI 5.0 dans le cadre d'un projet en Master ARIAS à
// l'Université d'Orléans.  Utilise ISO C++ 11.
//
// Copyright (C) 2025 Nicolas Paul <nicolas.paul1@etu.univ-orleans.fr> and
// Tolunay Akkaya <akkatolunay@icloud.com>.
// 
// This program is free software: you can redistribute it and/or modify
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

#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <string_view>
#include <tuple>
#include <stdexcept>
#include <cmath>

#include <mpi.h>
#include <graphviz/cgraph.h>


/// @brief Lecture du graphe avec GraphViz et génération de la matrice d'adjacence adaptée.
/// @param filename Le nom du fichier contenant le graphe.
/// @return std::tuple<std::vector<int>, std::map<std::string, int>, int> La matrice d'adjacence non adaptée, les sommets du graphe, et le nombre de sommets.
std::tuple<std::vector<int>, std::map<std::string, int>, int>
ReadGraph(std::string_view filename)
{
  // TODO: use C++'s file system feature
  FILE *file = fopen(filename.data(), "r");
  if (!file) throw std::runtime_error("Erreur d’ouverture du fichier .dot");
    
  Agraph_t *graph = agread(file, nullptr);
  fclose(file);
  if (!graph) throw std::runtime_error("Erreur de lecture GraphViz");
  
  int nodes_count = agnnodes(graph);
  std::map<std::string, int> nodes;
  nodes.reserve(static_cast<std::size_t>(nb_nodes));
    
  int i = 0;
  for (Agnode_t* node = agfstnode(graph); node; node = agnxtnode(graph, node)) nodes.emplace(agnameof(n), i++);
  
  std::vector<int> adjacency_matrix(nodes_count * nodes_count, 0);
    
  for (Agnode_t* node = agfstnode(graph); node; node = agnxtnode(graph, node)) {
    int const i = nodes.at(agnameof(node));
    for (Agedge_t* edge = agfstout(graph, node); edge; edge = agnxtout(graph, edge)) {
      int const j = nodes.at(agnameof(aghead(edge)));
      char* const w = agget(edge, (char*)"weight");
      int const weight = w ? std::stoi(w) : 1;
      adjacency_matrix[i * nb_nodes + j] = weight;
      adjacency_matrix[j * nb_nodes + i] = weight;
    }
  }
    
  agclose(graph);
  return std::make_tuple(std::move(adjacency_matrix), std::move(nodes), nodes_count);
}

int
main(int argc, char* argv)
{
  MPI_Init(&argc, &argv);
  
  int rank;
  int processors_count;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &processors_count);
  
  if (argc < 3)
  {
    std::cout << "Usage: " << argv[0] << "<file> <root>" << std::endl;
    MPI_Finalize();
    return 1;
  }
  
  std::string_view filename = argv[1];
  int root = std::atoi(argv[2]);
  
  if(rank == root) {
    auto [adjacency_matrix, nodes, nodes_count] = ReadGraph(filename);
  }
  // TODO: adjancency matrix est définie dans le if mais tout le monde en a besoin pour Scatterv!
  
  int b = compute_block_size(nodes_count, processors_count);
  auto [sendcounts, displs] = prepare_scatterv(nodes_count, processors_count, b);
  std::vector<int> local_block(sendcounts[rank]);
  MPI_Scatterv(adj.data(), sendcounts.data(), displs.data(),
               MPI_INT, local_block.data(), sendcounts[rank], MPI_INT,
               0, MPI_COMM_WORLD);
  return local_block;
  
  
  MPI_Finalize();
  return 0;
}


/// @brief Retourne la taille de bloc b pour découper la matrice en approx `processors_count` blocs
inline int compute_block_size(int nodes_count, int processors_count) {
    return static_cast<int>(std::ceil(nodes_count / std::sqrt(static_cast<double>(processors_count))));
}

/// @brief Retourne les indices de blocs (i,j) pour la matrice
inline auto generate_block_indices(int blocks_per_dim) {
    return std::views::iota(0, blocks_per_dim * blocks_per_dim);
}

/// @brief Calcule le nombre d'éléments dans un bloc (en gérant les bords)
inline int block_size(int bi, int bj, int b, int nodes_count) {
    int rows = std::min(b, nodes_count - bi * b);
    int cols = std::min(b, nodes_count - bj * b);
    return rows * cols;
}

/// @brief Prépare sendcounts et displs pour MPI_Scatterv
std::tuple<std::vector<int>, std::vector<int>> prepare_scatterv(int nodes_count, int processors_count, int b) {
    int blocks_per_dim = (nodes_count + b - 1) / b;
    int total_blocks = blocks_per_dim * blocks_per_dim;
    int blocks_per_proc = (total_blocks + processors_count - 1) / processors_count;

    std::vector<int> sendcounts(processors_count, 0);
    std::vector<int> displs(processors_count, 0);

    auto blocks = generate_block_indices(blocks_per_dim);
    for (int rank = 0; rank < processors_count; ++rank) {
        int start = rank * blocks_per_proc;
        int end = std::min(start + blocks_per_proc, total_blocks);
        sendcounts[rank] = std::accumulate(
            blocks | std::views::drop(start) | std::views::take(end - start),
            0,
            [blocks_per_dim, b, nodes_count](int sum, int idx) {
                int bi = idx / blocks_per_dim;
                int bj = idx % blocks_per_dim;
                return sum + block_size(bi, bj, b, nodes_count);
            });
        displs[rank] = (rank == 0 ? 0 : displs[rank - 1] + sendcounts[rank - 1]);
    }
    return {sendcounts, displs};
}
