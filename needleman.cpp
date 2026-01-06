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
#include<omp.h>


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
 *        avec l’algorithme de Needleman–Wunsch.
 *
 * Schéma de score :
 *  - match      : +1
 *  - mismatch   : -1
 *  - ouverture de gap : gap_open
 *
 * On ne reconstruit pas l’alignement, on calcule uniquement le score optimal.
 *
 * @param u Première séquence
 * @param v Deuxième séquence
 * @return Score d’alignement global optimal
 */
int
needleman_wunsch(const std::string& u, const std::string& v)
{
  int n = u.size();
  Matrix F(n+1, n+1);
  for(int i = 0; i<=n;i++) F(i, 0) = -3*i;
  for(int i = 0; i<=n;i++) F(0, i) = -3*i;

  for(int i = 1; i<=n; i++) {
    for(int j = 1; j<=n; j++) {
      int c1 = F(i-1, j-1) + (u[i] == v[j] ? 1 : -1);
      int c2 = F(i-1, j) -3;
      int c3 = F(i, j-1) -3;
      F(i, j) = std::max({ c1, c2, c3 });
    }
  }

  return F(n,n);
}

int
main(int argc, char **argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <fichier_sequences>\n";
    return 1;
  }

  const std::string filename = argv[1];
  std::vector<std::string> seq = lire_sequences(filename);
  Matrix S(seq.size(), seq.size());

  #pragma omp parallel for schedule(dynamic, 1) 
  for (std::size_t i = 0; i < seq.size(); ++i) {
    for (std::size_t j = 0; j < seq.size(); j++) {
      S(i, j) = S(j, i) = needleman_wunsch(seq[i], seq[j]);
    }
  }

  std::cout << "Matrice des scores Needleman-Wunsch\n" << S << std::endl;
  return 0;
}