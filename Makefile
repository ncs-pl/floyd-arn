# Projet Floyd-ARN.
# Copyright (C) 2025 Nicolas Paul <nicolas.paul1@etu.univ-orleans.fr> and
# Tolunay Akkaya <akkatolunay@etu.univ-orleans.fr>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

floyd: floyd.cpp
	mpicxx       \
	  -std=c++11 \
	  -Wall      \
	  -Wextra    \
	  -pedantic  \
	  -I/opt/homebrew/include \
	  -L/opt/homebrew/lib \
	  -O3 \
	  floyd.cpp \
	  -lcgraph \
	  -o floyd

arn: arn.cpp
	mpicxx       \
	  -std=c++11 \
	  -Wall      \
	  -Wextra    \
	  -pedantic  \
	  -I/opt/homebrew/include \
	  -L/opt/homebrew/lib \
	  -O3 \
	  -g \
	  arn.cpp \
	  -o arn   \
	  -lcgraph 
	   
needleman: needleman.cpp
	g++ \
	  -std=c++11 \
	  -Wall \
	  -Wextra \
	  -pedantic \
	  -O3 \
	  -fopenmp \
	  needleman.cpp \
	  -o needleman


arn2: arn2.cpp
	mpicxx       \
	  -std=c++11 \
	  -Wall      \
	  -Wextra    \
	  -pedantic  \
	  -I/opt/homebrew/include \
	  -L/opt/homebrew/lib \
	  -O3 \
	  -fopenmp \
	  -g \
	  arn2.cpp \
	  -o arn2   \
	  -lcgraph