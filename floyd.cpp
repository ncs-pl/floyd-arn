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

/**
 * @file floyd_mpi.c
 * @brief Parallélisation de l'algorithme de Floyd-Warshall par blocs avec MPI.
 *
 * Hypothèses :
 * - n est divisible par b
 * - nprocs = (n / b) * (n / b) et est un carré parfait
 *
 * Usage :
 *   mpirun -np P ./floyd_mpi n b input.txt output.txt
 *
 * input.txt :
 *   n
 *   n lignes de n entiers (matrice d'adjacence / distances)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define INF 1000000000

/**
 * @brief Alloue un tableau d'entiers de taille n.
 */
static int *alloc_int_array(int n) {
    int *ptr = (int *)malloc(n * sizeof(int));
    if (!ptr) {
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return ptr;
}

/**
 * @brief Lit une matrice n x n depuis un fichier (sur le rang 0 uniquement).
 *
 * @param filename Nom du fichier
 * @param n        Taille de la matrice
 * @return int*    Pointeur vers la matrice allouée (taille n*n)
 */
static int *read_full_matrix(const char *filename, int n, int rank) {
    if (rank != 0) return NULL;

    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Impossible d'ouvrir le fichier %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n_file;
    if (fscanf(f, "%d", &n_file) != 1 || n_file != n) {
        fprintf(stderr, "Taille n incohérente dans le fichier\n");
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int *mat = alloc_int_array(n * n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (fscanf(f, "%d", &mat[i * n + j]) != 1) {
                fprintf(stderr, "Erreur de lecture de la matrice\n");
                fclose(f);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }

    fclose(f);
    return mat;
}

/**
 * @brief Écrit la matrice n x n dans un fichier (rang 0 seulement).
 */
static void write_full_matrix(const char *filename, int *mat, int n, int rank) {
    if (rank != 0) return;

    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Impossible d'ouvrir le fichier %s en écriture\n", filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(f, "%d\n", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(f, "%d ", mat[i * n + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

/**
 * @brief Copie un bloc b x b de la matrice globale vers un buffer local.
 *
 * @param global  matrice globale n x n (rang 0)
 * @param n       taille globale
 * @param block   buffer de taille b*b
 * @param b       taille de bloc
 * @param br      indice de bloc en ligne
 * @param bc      indice de bloc en colonne
 */
static void copy_block_from_global(int *global, int n,
                                   int *block, int b,
                                   int br, int bc) {
    int row_offset = br * b;
    int col_offset = bc * b;

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) {
            block[i * b + j] = global[(row_offset + i) * n + (col_offset + j)];
        }
    }
}

/**
 * @brief Copie un bloc local b x b dans la matrice globale.
 */
static void copy_block_to_global(int *global, int n,
                                 int *block, int b,
                                 int br, int bc) {
    int row_offset = br * b;
    int col_offset = bc * b;

    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) {
            global[(row_offset + i) * n + (col_offset + j)] = block[i * b + j];
        }
    }
}

/**
 * @brief Phase 1 : Floyd-Warshall sur un bloc diagonal b x b.
 */
static void floyd_phase1_diagonal(int *block, int b) {
    for (int k = 0; k < b; ++k) {
        for (int i = 0; i < b; ++i) {
            int dik = block[i * b + k];
            if (dik == INF) continue;
            for (int j = 0; j < b; ++j) {
                int k_j = block[k * b + j];
                if (k_j == INF) continue;
                int cand = dik + k_j;
                if (cand < block[i * b + j]) {
                    block[i * b + j] = cand;
                }
            }
        }
    }
}

/**
 * @brief Phase 2 (ligne k) : met à jour un bloc (k, j) avec le bloc diagonal diag.
 *
 * block : bloc (k, j)
 * diag  : bloc (k, k)
 */
static void floyd_phase2_row(int *block, int *diag, int b) {
    for (int kk = 0; kk < b; ++kk) {
        for (int i = 0; i < b; ++i) {
            int dik = diag[i * b + kk];
            if (dik == INF) continue;
            for (int j = 0; j < b; ++j) {
                int k_j = block[kk * b + j];
                if (k_j == INF) continue;
                int cand = dik + k_j;
                if (cand < block[i * b + j]) {
                    block[i * b + j] = cand;
                }
            }
        }
    }
}

/**
 * @brief Phase 2 (colonne k) : met à jour un bloc (i, k) avec le bloc diagonal diag.
 *
 * block : bloc (i, k)
 * diag  : bloc (k, k)
 */
static void floyd_phase2_col(int *block, int *diag, int b) {
    for (int kk = 0; kk < b; ++kk) {
        for (int i = 0; i < b; ++i) {
            int i_k = block[i * b + kk];
            if (i_k == INF) continue;
            for (int j = 0; j < b; ++j) {
                int k_j = diag[kk * b + j];
                if (k_j == INF) continue;
                int cand = i_k + k_j;
                if (cand < block[i * b + j]) {
                    block[i * b + j] = cand;
                }
            }
        }
    }
}

/**
 * @brief Phase 3 : met à jour un bloc (i, j) avec les blocs (i, k) (colBlock) et (k, j) (rowBlock).
 */
static void floyd_phase3(int *block, int *colBlock, int *rowBlock, int b) {
    for (int kk = 0; kk < b; ++kk) {
        for (int i = 0; i < b; ++i) {
            int i_k = colBlock[i * b + kk];
            if (i_k == INF) continue;
            for (int j = 0; j < b; ++j) {
                int k_j = rowBlock[kk * b + j];
                if (k_j == INF) continue;
                int cand = i_k + k_j;
                if (cand < block[i * b + j]) {
                    block[i * b + j] = cand;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 5) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s n b input.txt output.txt\n", argv[0]);
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int n = atoi(argv[1]);
    int b = atoi(argv[2]);
    const char *input_file = argv[3];
    const char *output_file = argv[4];

    if (n % b != 0) {
        if (rank == 0) {
            fprintf(stderr, "Erreur: n doit être divisible par b\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int p = n / b; // nombre de blocs par dimension
    if (p * p != nprocs) {
        if (rank == 0) {
            fprintf(stderr, "Erreur: nprocs doit être égal à (n / b) * (n / b)\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int q = (int)sqrt((double)nprocs);
    if (q * q != nprocs || q != p) {
        if (rank == 0) {
            fprintf(stderr, "Erreur: nprocs doit être un carré parfait et égal à (n / b)^2\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Création d'une grille cartésienne 2D p x p
    int dims[2] = {p, p};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int cart_rank;
    MPI_Comm_rank(cart_comm, &cart_rank);

    int coords[2];
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    int pr = coords[0]; // indice de bloc en ligne
    int pc = coords[1]; // indice de bloc en colonne

    // Communicateurs de ligne et de colonne
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];

    // Communicateur de ligne : on garde la 2ème dimension (colonnes)
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

    // Communicateur de colonne : on garde la 1ère dimension (lignes)
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    // Rang 0 lit la matrice complète
    int *full_matrix = read_full_matrix(input_file, n, rank);

    // Distribution des blocs b x b vers chaque processus
    int *local_block = alloc_int_array(b * b);

    if (rank == 0) {
        // Envoi des blocs
        for (int br = 0; br < p; ++br) {
            for (int bc = 0; bc < p; ++bc) {
                int dest_coords[2] = {br, bc};
                int dest_rank;
                MPI_Cart_rank(cart_comm, dest_coords, &dest_rank);

                if (dest_rank == 0) {
                    copy_block_from_global(full_matrix, n, local_block, b, br, bc);
                } else {
                    int *tmp_block = alloc_int_array(b * b);
                    copy_block_from_global(full_matrix, n, tmp_block, b, br, bc);
                    MPI_Send(tmp_block, b * b, MPI_INT, dest_rank, 0, cart_comm);
                    free(tmp_block);
                }
            }
        }
    } else {
        MPI_Recv(local_block, b * b, MPI_INT, 0, 0, cart_comm, MPI_STATUS_IGNORE);
    }

    // Buffers pour les phases
    int *diag_block = alloc_int_array(b * b);
    int *row_block  = alloc_int_array(b * b);
    int *col_block  = alloc_int_array(b * b);

    // Boucle principale sur les blocs diagonaux k = 0..p-1
    for (int k = 0; k < p; ++k) {
        // PHASE 1: bloc diagonal (k, k)
        if (pr == k && pc == k) {
            floyd_phase1_diagonal(local_block, b);
            // Copie dans diag_block pour diffusion
            for (int i = 0; i < b * b; ++i) {
                diag_block[i] = local_block[i];
            }
        }

        // Diffusion du bloc diagonal dans la ligne k
        if (pr == k) {
            MPI_Bcast(diag_block, b * b, MPI_INT, k, row_comm);
        }

        // Diffusion du bloc diagonal dans la colonne k
        if (pc == k) {
            MPI_Bcast(diag_block, b * b, MPI_INT, k, col_comm);
        }

        // PHASE 2: mise à jour des blocs de la ligne k et de la colonne k
        if (pr == k && pc != k) {
            // Bloc (k, j), j != k
            floyd_phase2_row(local_block, diag_block, b);
        }
        if (pc == k && pr != k) {
            // Bloc (i, k), i != k
            floyd_phase2_col(local_block, diag_block, b);
        }

        // PHASE 3: mise à jour des blocs (i, j) avec i != k, j != k
        // On diffuse pour chaque ligne i le bloc (i, k) sur row_comm
        // et pour chaque colonne j le bloc (k, j) sur col_comm.

        // 1) Diffusion des blocs de colonne (i, k) le long des lignes
        if (pc == k) {
            // Ce processus possède le bloc (pr, k)
            // On le met dans col_block puis on le diffuse dans sa ligne
            for (int i = 0; i < b * b; ++i) {
                col_block[i] = local_block[i];
            }
        }
        // Tous les processus de la même ligne reçoivent col_block
        MPI_Bcast(col_block, b * b, MPI_INT, k, row_comm);

        // 2) Diffusion des blocs de ligne (k, j) le long des colonnes
        if (pr == k) {
            // Ce processus possède le bloc (k, pc)
            // On le met dans row_block puis on le diffuse dans sa colonne
            for (int i = 0; i < b * b; ++i) {
                row_block[i] = local_block[i];
            }
        }
        // Tous les processus de la même colonne reçoivent row_block
        MPI_Bcast(row_block, b * b, MPI_INT, k, col_comm);

        // 3) Mise à jour des blocs (i, j) avec i != k, j != k
        if (pr != k && pc != k) {
            floyd_phase3(local_block, col_block, row_block, b);
        }

        MPI_Barrier(cart_comm);
    }

    // Rassemblement des blocs vers le rang 0
    if (rank == 0) {
        for (int br = 0; br < p; ++br) {
            for (int bc = 0; bc < p; ++bc) {
                int src_coords[2] = {br, bc};
                int src_rank;
                MPI_Cart_rank(cart_comm, src_coords, &src_rank);

                if (src_rank == 0) {
                    copy_block_to_global(full_matrix, n, local_block, b, br, bc);
                } else {
                    int *tmp_block = alloc_int_array(b * b);
                    MPI_Recv(tmp_block, b * b, MPI_INT, src_rank, 1, cart_comm, MPI_STATUS_IGNORE);
                    copy_block_to_global(full_matrix, n, tmp_block, b, br, bc);
                    free(tmp_block);
                }
            }
        }
    } else {
        MPI_Send(local_block, b * b, MPI_INT, 0, 1, cart_comm);
    }

    // Rang 0 écrit la matrice résultat
    write_full_matrix(output_file, full_matrix, n, rank);

    // Nettoyage
    free(local_block);
    free(diag_block);
    free(row_block);
    free(col_block);

    if (rank == 0) {
        free(full_matrix);
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);

    MPI_Finalize();
    return 0;
}
