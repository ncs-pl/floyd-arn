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

// Ce fichier propose une parallélisation de l'algorithme de Needleman-Wunsch
// en utilisant OpenMP ? et ISO C++ 11.


#include <vector>
#include <fstream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <iostream>

constexpr int kInfinity = std::numeric_limits<int>::max();


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
 *        avec l’algorithme de Needleman–Wunsch (pénalités affines).
 *
 * Schéma de score :
 *  - match      : +1
 *  - mismatch   : -1
 *  - ouverture de gap : gap_open
 *  - extension de gap : gap_extend
 *
 * On ne reconstruit PAS l’alignement, on calcule uniquement le score optimal.
 *
 * @param s1 Première séquence
 * @param s2 Deuxième séquence
 * @param gap_open Pénalité d’ouverture de gap (ex : -3)
 * @param gap_extend Pénalité d’extension de gap (ex : -1)
 * @return Score d’alignement global optimal
 */
long long needleman_wunsch_score(const std::string& s1,
                                 const std::string& s2,
                                 int gap_open = -3,
                                 int gap_extend = -1)
{
    const int n = static_cast<int>(s1.size());
    const int m = static_cast<int>(s2.size());

    // Valeur représentant -inf
    const long long NEG_INF = std::numeric_limits<long long>::min() / 4;

    // M  : fin par match / mismatch
    // Ix : fin par gap dans s2
    // Iy : fin par gap dans s1
    std::vector<std::vector<long long>> M(n + 1, std::vector<long long>(m + 1, NEG_INF));
    std::vector<std::vector<long long>> Ix(n + 1, std::vector<long long>(m + 1, NEG_INF));
    std::vector<std::vector<long long>> Iy(n + 1, std::vector<long long>(m + 1, NEG_INF));

    // Initialisation
    M[0][0] = 0;

    // Première colonne : gaps dans s2
    for (int i = 1; i <= n; ++i) {
        if (i == 1)
            Ix[i][0] = gap_open;
        else
            Ix[i][0] = Ix[i - 1][0] + gap_extend;
    }

    // Première ligne : gaps dans s1
    for (int j = 1; j <= m; ++j) {
        if (j == 1)
            Iy[0][j] = gap_open;
        else
            Iy[0][j] = Iy[0][j - 1] + gap_extend;
    }

    // Remplissage des matrices
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {

            // Score de substitution
            int sub = (s1[i - 1] == s2[j - 1]) ? 1 : -1;

            // Match / mismatch
            M[i][j] = std::max({ M[i - 1][j - 1],
                                 Ix[i - 1][j - 1],
                                 Iy[i - 1][j - 1] }) + sub;

            // Gap dans s2
            Ix[i][j] = std::max({ M[i - 1][j]  + gap_open,
                                  Iy[i - 1][j] + gap_open,
                                  Ix[i - 1][j] + gap_extend });

            // Gap dans s1
            Iy[i][j] = std::max({ M[i][j - 1]  + gap_open,
                                  Ix[i][j - 1] + gap_open,
                                  Iy[i][j - 1] + gap_extend });
        }
    }

    // Score final
    return std::max({ M[n][m], Ix[n][m], Iy[n][m] });
}

/**
 * @brief Construit la matrice des scores Needleman–Wunsch pour toutes les séquences.
 *
 * La matrice est symétrique : score(i,j) = score(j,i).
 *
 * @param filename Fichier contenant les séquences.
 * @return Matrix NxN contenant les scores d'alignement global.
 */
Matrix needleman_score_matrix(const std::string& filename)
{
  std::vector<std::string> sequences = lire_sequences(filename);
  const std::size_t N = sequences.size();
  Matrix S(N, N);

  for (std::size_t i = 0; i < N; ++i) {
    // Diagonale : score de la séquence avec elle-même (souvent = longueur * match)
    S(i, i) = needleman_wunsch_score(sequences[i], sequences[i]);

    for (std::size_t j = i + 1; j < N; ++j) {
      const int sc = needleman_wunsch_score(sequences[i], sequences[j]);
      S(i, j) = sc;
      S(j, i) = sc; // symétrie
    }
  }

  return S;
}


int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <fichier_sequences>\n";
    return 1;
  }

  const std::string filename = argv[1];

  try {
    Matrix S = needleman_score_matrix(filename);

    std::cout << "Matrice des scores Needleman-Wunsch (" 
              << S.rows() << " x " << S.cols() << ")\n";
    std::cout << S << std::endl;   
  }
  catch (const std::exception& e) {
    std::cerr << "Erreur: " << e.what() << "\n";
    return 2;
  }

  return 0;
}