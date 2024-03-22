
import numpy as np
import pandas as pd

class SudokuSolver():
    def __init__(self, sudoku=None) -> None:
        self.sudoku = sudoku

    def add_sudoku(self, sudoku):
        self.sudoku = sudoku

    def solve(self):
        find = self.find_empty()
        if not find:
            return True
        else:
            row, col = find

        for i in range(1,10):
            if self.valid(str(i), (row, col)):
                self.sudoku[row][col] = str(i)

                if self.solve():
                    return True

                self.sudoku[row][col] = ' '

        return False


    def valid(self, num, pos):
        # Check row
        for i in range(len(self.sudoku[0])):
            if self.sudoku[pos[0]][i] == num and pos[1] != i:
                return False

        # Check column
        for i in range(len(self.sudoku)):
            if self.sudoku[i][pos[1]] == num and pos[0] != i:
                return False

        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3

        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if self.sudoku[i][j] == num and (i,j) != pos:
                    return False

        return True


    def print_board(self):
        for i in range(len(self.sudoku)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - -")

            for j in range(len(self.sudoku[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")

                if j == 8:
                    print(self.sudoku[i][j])
                elif (j == 2) or (j == 5):
                    print(self.sudoku[i][j], end = '')
                else:
                    print(str(self.sudoku[i][j]) + " ", end="")


    def find_empty(self):
        for i in range(len(self.sudoku)):
            for j in range(len(self.sudoku[0])):
                if self.sudoku[i][j] == ' ':
                    return (i, j)  # row, col

    def as_array(self):
        return np.array(self.sudoku).reshape(9, 9)