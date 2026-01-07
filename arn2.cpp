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
// en utilisant OpenMPI 5+, OpenMP 4.0 et ISO C++ 11.

// mpirun -np np ./arn root b input  k
// mpirun -np 4 ./arn 0 50 dataset_100seq.fa  4
// mpirun -np 4 ./arn 0 250 dataset_500seq.fa  4
// mpirun -np 4 ./arn 0 1000 dataset_2000seq.fa  4

#include <algorithm>
#include <array>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cctype>   // std::isspace
#include <fstream>  // in dans liresequences


#include <graphviz/cgraph.h>
#include <mpi.h>
#include <limits>
#include <cstring>

constexpr int kInfinity = std::numeric_limits<int>::max();
constexpr int kScatterTag = 1;
constexpr int kGatherTag = 2;

/**
 * @brief Matrice dense en stockage ligne-major (row-major).
 *
 * Cette classe représente une matrice de dimensions @f$ m\_rows \times m\_columns @f$
 * stockée dans un seul buffer contigu (std::vector<int>) en ordre ligne-major.
 * L'accès aux éléments se fait via l'opérateur ().
 */
class Matrix {
private:
  std::size_t m_rows, m_columns;
  std::vector<int> m_buffer;

public:
  /**
   * @brief Construit une matrice de taille n x m.
   *
   * Les éléments ne sont pas initialisés à une valeur particulière
   * (le std::vector est simplement alloué).
   *
   * @param n Nombre de lignes.
   * @param m Nombre de colonnes.
   */
  Matrix(std::size_t n, std::size_t m) : m_rows(n), m_columns(m), m_buffer(n*m)
  {}

  /**
   * @brief Accès en écriture à l'élément (i, j) (version non-const).
   *
   * @param i Indice de ligne (0 ≤ i < rows()).
   * @param j Indice de colonne (0 ≤ j < cols()).
   * @return Référence modifiable sur l'élément (i, j).
   */
  int&
  operator()(std::size_t i, std::size_t j)
  {
    return m_buffer[i*m_columns+j];
  }

   /**
   * @brief Accès en lecture à l'élément (i, j) (version const).
   *
   * @param i Indice de ligne (0 ≤ i < rows()).
   * @param j Indice de colonne (0 ≤ j < cols()).
   * @return Valeur de l'élément (i, j).
   */
  int
  operator()(std::size_t i, std::size_t j) const
  {
    return m_buffer[i*m_columns+j];
  }

  /**
   * @brief Renvoie le nombre de lignes de la matrice.
   *
   * @return Nombre de lignes.
   */
  std::size_t
  rows() const
  {
    return m_rows;
  }

  /**
   * @brief Renvoie le nombre de colonnes de la matrice.
   *
   * @return Nombre de colonnes.
   */
  std::size_t
  cols() const
  {
    return m_columns;
  }

  /**
   * @brief Renvoie le nombre total d'éléments de la matrice.
   *
   * Équivaut à rows() * cols().
   *
   * @return Taille logique du buffer (nombre d'éléments).
   */
  std::size_t
  size() const
  {
    return m_rows * m_columns;
  }

  /**
   * @brief Accès au buffer sous-jacent (version non-const).
   *
   * @return Pointeur brut sur les données internes.
   */
  int *
  data()
  {
    return m_buffer.data();
  }

  /**
   * @brief Accès au buffer sous-jacent (version const).
   *
   * @return Pointeur brut constant sur les données internes.
   */
  
  const int *
  data() const
  {
    return m_buffer.data();
  }
};

/**
 * @brief Affiche la matrice sur un flux de sortie.
 *
 * Les éléments sont séparés par des espaces et chaque ligne est terminée
 * par un saut de ligne. Les valeurs égales à kInfinity sont affichées
 * sous forme du symbole "∞".
 *
 * @param s Flux de sortie (std::ostream).
 * @param M Matrice à afficher.
 * @return Référence sur le flux de sortie (pour chaînage).
 */
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

//////////// PAM /////////////// 

/**
 * @brief Représentation d'un résultat de clustering PAM.
 * medoids : indices des k médoïdes (taille k)
 * assignment : pour chaque sommet i, indice du médoïde qui lui est associé
 */
struct PAMResult {
    std::vector<int> medoids;
    std::vector<int> assignment; // même taille que n, optionnellement rempli
};


// Calcul du coût local pour un ensemble de médoïdes (lignes locales uniquement)
static long long compute_cost_local(int n,
                                    const int* D_local,
                                    int local_n,
                                    const std::vector<int>& medoids)
{
    long long local_cost = 0;
    for (int i = 0; i < local_n; ++i) {
        int best = kInfinity;
        for (int m : medoids) {
            int d = D_local[i * n + m];
            if (d < best) best = d;
        }
        local_cost += best;
    }
    return local_cost;
}

/**
 * @brief Version parallèle de PAM avec MPI, sur la matrice de distances.
 *
 * Hypothèses :
 *  - n est divisible par le nombre de processus
 *  - D_global est la matrice n×n sur le rang 0 (en ligne-major)
 *  - Sur les autres rangs, D_global peut être nullptr
 *  - L'algorithme distribue les lignes de D entre les processus,
 *    réplique les médoïdes sur tous, et utilise des MPI_Allreduce
 *
 * @param n nombre de sommets
 * @param D_global matrice de distances sur le rang 0 (peut être modifiée ou non)
 * @param k nombre de médoïdes
 * @param comm communicateur MPI
 * @return PAMResult (valide uniquement sur le rang 0 ; sur les autres rangs,
 *         medoids/assignment peuvent être vides)
 */
PAMResult pam(int n, int* D_global, int k, MPI_Comm comm){
    PAMResult result;

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Diffuser n et k (au cas où)
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    MPI_Bcast(&k, 1, MPI_INT, 0, comm);

    if (k <= 0 || k > n) {
        if (rank == 0) {
            std::cerr << "Erreur : k invalide dans pam." << std::endl;
        }
        return result;
    }

    if (n % size != 0) {
        if (rank == 0) {
            std::cerr << "Erreur : n (" << n
                      << ") n'est pas divisible par le nombre de processus ("
                      << size << ") dans pam." << std::endl;
        }
        MPI_Abort(comm, 1);
    }

    int local_n    = n / size;
    int local_size = local_n * n;

    int* D_local = new int[local_size];

    // Répartition des lignes de D_global vers tous les processus
    MPI_Scatter(
        D_global,    // sendbuf (rang 0)
        local_size,  // sendcount par processus
        MPI_INT,
        D_local,     // recvbuf local
        local_size,  // recvcount
        MPI_INT,
        0,
        comm
    );

    // Initialisation simple des médoïdes : {0, 1, ..., k-1}
    std::vector<int> medoids(k);
    for (int i = 0; i < k; ++i) {
        medoids[i] = i;
    }

    // Calcul du coût initial (parallèle)
    long long local_cost = compute_cost_local(n, D_local, local_n, medoids);
    long long current_cost = 0;
    MPI_Allreduce(&local_cost, &current_cost, 1, MPI_LONG_LONG, MPI_SUM, comm);

    bool global_improved = true;

    while (global_improved) { // tout le monde teste le meme echange mais chacun calcule sa ligne
        global_improved = false;

        long long best_cost = current_cost;
        std::vector<int> best_medoids = medoids;

        // Tous les processus parcourent les mêmes candidats (mi, o)
        for (int mi = 0; mi < k; ++mi) {
            int m = medoids[mi];

            for (int o = 0; o < n; ++o) {
                // Sauter si o est deja medoid
                bool isMedoid = false;
                for (int x : medoids) {
                    if (x == o) { isMedoid = true; break; }
                }
                if (isMedoid) continue;
                if (o == m) continue;

                // Construire les médoïdes candidats M'
                std::vector<int> candidate = medoids;
                candidate[mi] = o;

                // Calcul du coût local pour ce candidat
                long long cand_local_cost = compute_cost_local(n, D_local, local_n, candidate);
                long long cand_global_cost = 0;

                // Tous les processus participent à ce Allreduce
                MPI_Allreduce(&cand_local_cost, &cand_global_cost, 1,
                              MPI_LONG_LONG, MPI_SUM, comm);

                // Seul le rang 0 décide si c'est un meilleur candidat
                if (rank == 0 && cand_global_cost < best_cost) {
                    best_cost = cand_global_cost;
                    best_medoids = candidate;
                    global_improved = true;
                }

                // Tous les rangs doivent garder la même valeur de global_improved
                int local_flag = global_improved ? 1 : 0;
                int global_flag = 0;
                MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_LOR, comm);
                global_improved = (global_flag != 0);

                // On ne peut pas sortir des boucles ici sans casser l'alignement
                // des Allreduce, donc on continue jusqu'à la fin, mais on
                // n'utilisera que best_medoids/best_cost à la sortie.
            }
        }

        // Maintenant, seul le rang 0 sait quels sont les meilleurs médoïdes finaux
        // On vérifie s'il y a une amélioration globale
        int improved_int = 0;
        if (rank == 0) {
            if (best_cost < current_cost) {
                improved_int = 1;
                medoids = best_medoids;
                current_cost = best_cost;
            } else {
                improved_int = 0;
            }
        }

        // Diffuser la décision d'amélioration
        MPI_Bcast(&improved_int, 1, MPI_INT, 0, comm);
        global_improved = (improved_int != 0); 

        if (global_improved) {
            // Diffuser les nouveaux médoïdes et le nouveau coût à tous
            MPI_Bcast(medoids.data(), k, MPI_INT, 0, comm);
            MPI_Bcast(&current_cost, 1, MPI_LONG_LONG, 0, comm);
        }
    }

    // Remplissage du résultat sur le rang 0
    if (rank == 0) {
        result.medoids = medoids;
        result.assignment.resize(n, -1); // initialisé avec -1

        // Assignation finale en utilisant D_global complet
        for (int i = 0; i < n; ++i) {
            int best = kInfinity; // distance minimale pour un médoide
            int best_idx = -1; // l'indice du médoide le plus proche
            for (int mi = 0; mi < k; ++mi) {
                int m = medoids[mi];
                int d = D_global[i * n + m]; //distance entre le sommet i et le médoide m
                if (d < best) {
                    best = d;
                    best_idx = mi;
                }
            }
            result.assignment[i] = best_idx; // on dit que le sommet i appartient au cluster best_idx
        }
    }

    delete[] D_local;
    return result;
}

//////////// ARN /////////////// 


/**
 * @brief Lit un fichier de séquences au format FASTA simplifié :
 *        >id
 *        SEQUENCE...
 *
 * Hypothèses :
 *  - chaque séquence commence par une ligne '>'
 *  - les lignes suivantes (jusqu'au prochain '>') sont concaténées
 *  - toutes les séquences ont la même longueur (ex : 100)
 *
 * @param filename chemin du fichier
 * @param seq_length longueur attendue des séquences (0 = pas de vérification stricte)
 * @return std::vector<std::string> liste des séquences
 */
std::vector<std::string> lire_sequences(const std::string& filename, int seq_length = 0){ 
    std::vector<std::string> sequences;
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Erreur : impossible d'ouvrir le fichier " << filename << std::endl;
        return sequences;
    }

    std::string line;
    std::string current_seq;

    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '>') {
            // Nouvelle séquence : on sauvegarde l'ancienne si elle existe
            if (!current_seq.empty()) {
                if (seq_length > 0 && (int)current_seq.size() != seq_length) {
                    std::cerr << "Avertissement : sequence de longueur "
                              << current_seq.size()
                              << " differente de " << seq_length << std::endl;
                }
                sequences.push_back(current_seq);
                current_seq.clear();
            }
            // On ignore le header (">0", ">1", etc.)
        } else {
            // Ligne de séquence : on enlève les espaces et on concatène
            for (char c : line) {
                if (!std::isspace(static_cast<unsigned char>(c))) {
                    current_seq.push_back(c);
                }
            }
        }
    }

    // Ne pas oublier la dernière séquence
    if (!current_seq.empty()) {
        if (seq_length > 0 && (int)current_seq.size() != seq_length) {
            std::cerr << "Avertissement : sequence de longueur "
                      << current_seq.size()
                      << " differente de " << seq_length << std::endl;
        }
        sequences.push_back(current_seq);
    }

    return sequences;
}

/**
 * @brief Calcule le score d’alignement global entre deux séquences
 *        avec l’algorithme de Needleman–Wunsch.
 *
 * @param u Première séquence
 * @param v Deuxième séquence
 * @return Score d’alignement global optimal
 */
int
needleman_wunsch(const std::string& u, const std::string& v)
{
  std::size_t n = u.size() + 1;
  std::size_t m = v.size() + 1;
  Matrix F(n, m);
  for(std::size_t i = 0; i<n;i++) F(i, 0) = -3*i;
  for(std::size_t i = 0; i<m;i++) F(0, i) = -3*i;

  for(std::size_t i = 1; i<n; i++) {
    for(std::size_t j = 1; j<m; j++) {
      int match = F(i-1, j-1) + (u[i-1] == v[j-1] ? 1 : -1);
      int deletion = F(i-1, j) -3;
      int insertion = F(i, j-1) -3;
      F(i, j) = std::max({ match, deletion, insertion });
    }
  }

  return F(n-1,m-1);
}


/**
 * @brief Construit la matrice de similarité entre des séquences d'ARN.
 *
 * Calcul :
 *  - Le score d’alignement est calculé par la fonction needleman_wunsch()
 *  - Le calcul est parallélisé avec OpenMP
 *
 * @param sequences Vecteur contenant les séquences d’ARN
 * @return Matrix Matrice n × n des scores d’alignement
 */
Matrix construire_matrice_adjacence_arn(const std::vector<std::string>& sequences,int th){

    Matrix mat(sequences.size(),sequences.size());
    #pragma omp parallel for schedule(dynamic, 1)  num_threads(th)
    for (std::size_t i = 0; i < sequences.size(); ++i) {
            mat(i, i) = needleman_wunsch(sequences[i], sequences[i]); // diagonale

      for (std::size_t j = i + 1; j < sequences.size(); j++) {
        mat(i, j) = mat(j, i) = needleman_wunsch(sequences[i], sequences[j]);
      }
    }
    return mat;
}


int
main(int argc, char **argv)
{
  int np = 0;
  int provided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  std::cout << "provided = " << provided << std::endl;
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (argc != 5) {
    std::cout << argv[0] << " root b input k" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // TODO(nico): check for contracts?
  int root = std::stoi(argv[1], nullptr, 10);
  std::size_t b = static_cast<std::size_t>(std::stoi(argv[2], nullptr, 10));
  std::string input = argv[3];
  int k = std::stoi(argv[4]);

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
  std::vector<std::string> sequences; // pour afficher les ARN (rang 0)

  if (pid == root) {
    sequences = lire_sequences(input, 0);
    n = sequences.size();
    Matrix similarity = construire_matrice_adjacence_arn(sequences, provided);
    
    // Conversion : distance = max_score - score
    int max_score = 100; // score diagonal (auto-alignement)
    A = Matrix(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = max_score - similarity(i, j);
        }
        A(i, i) = 0; // distance à soi-même = 0
    }
  }
  

  MPI_Bcast(&n, 1, MPI_INT, root, cart_comm);


  if (root == pid){
    std::cout << "Matrice adjacente :" << std::endl;
    std::cout << A << "\n" << std::endl;
}

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
    for (std::size_t i = 0; i < (std::size_t)dims[0]; ++i) {
      for (std::size_t j = 0; j < (std::size_t)dims[1]; ++j){
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
    std::cout << "Matrice des distances (Floyd) :" << std::endl;
    std::cout << A << std::endl;
  }
  
  /// PAM ///

  PAMResult res = pam(n, A.data(), k, cart_comm);

  if (pid == root) {
        if (res.medoids.empty()) {
            std::cerr << "Erreur : pam a echoue ou retourne un resultat vide."
                      << std::endl;
            MPI_Finalize();
            return EXIT_FAILURE;
        }

        std::cout << "\nMedoids retenus :" << std::endl;
    for (std::size_t idx = 0; idx < res.medoids.size(); ++idx) {
        int m = res.medoids[idx];
        std::cout << "  Medoid " << idx << " ->"
             << " (ARN : " << m << ")";
        std::cout << std::endl;
    }

    // Affichage des clusters 

    if (res.assignment.size() == n) {
        std::cout << "\nPartitions (clusters) :" << std::endl;
        for (int c = 0; c < k; ++c) {
            int med = res.medoids[c];
            std::cout << "Cluster " << c
                 << " (ARN #" << med << ") : \n";

            bool first = true;
            for (std::size_t i = 0; i < n; ++i) {
                if (res.assignment[i] == c) {
                    if (!first) std::cout << ", ";
                    std::cout << "Arn";
                    if (i < n) {
                        std::cout << " (" << i << ")";
                    }
                    first = false;
                }
            }
            std::cout << std::endl;
        }
    }
  }


  MPI_Comm_free(&cols_comm);
  MPI_Comm_free(&rows_comm);
  MPI_Comm_free(&cart_comm);

  MPI_Finalize();
  return 0;
}
